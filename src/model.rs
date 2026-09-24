//! Device-resident weights: one VRAM arena holding every container payload,
//! plus the name -> (address, length) table the kernels use. On the CPU backend
//! the "device" addresses are pointers into the mmap'd container instead.

use crate::container::{Container, Dtype};
use crate::cuda;
use crate::cuda::cuda_ffi::{self, CUdeviceptr};

/// Aligned on-device q8_0 stride: the ggml layout is 34 bytes (2-byte f16 scale
/// + 32 int8), so every 4-byte word of the payload is misaligned and the GEMMs
/// have to assemble words with byte loads and shifts (~8 instructions per word,
/// i.e. ~64 per 32-element block against only 8 dp4a). On the device each block
/// is repacked to 36 bytes with the payload at offset 4, which is 4-byte
/// aligned, so the inner loops use plain int loads - measured 3.1x faster
/// (2048x11008x297: 469 -> 1464 GFLOP/s).
pub const Q8_0_BLOCK_IN: usize = 34;
pub const Q8_0_BLOCK_DEV: usize = 36;

/// One tensor on the device (or, on the CPU, in the mmap'd container): the
/// address the kernels read and the payload length.
pub struct DeviceTensor {
    pub addr: CUdeviceptr,
    pub nbytes: usize,
}

pub struct DeviceWeights {
    pub arena: CUdeviceptr,
    pub tensors: std::collections::HashMap<String, DeviceTensor>,
}

impl DeviceWeights {
    /// Allocate one arena sized to the container's payload footprint and copy
    /// every payload into it. Aliases resolve to their target's address, so tied
    /// weights cost nothing extra.
    pub fn upload(c: &Container) -> Result<DeviceWeights, String> {
        let d = cuda_ffi::driver()?;

        // Q8_0 payloads are repacked from the container's 34-byte block stride
        // to the aligned 36-byte device stride, so the device offsets do not
        // mirror the container's any more; lay out the arena here instead
        // (4096-aligned per tensor).
        let mut dev_off: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        let mut cursor = 0usize;
        for name in &c.order {
            let t = &c.tensors[name];
            if t.alias_of.is_some() {
                continue;
            }
            cursor = (cursor + 4095) & !4095usize;
            dev_off.insert(name.clone(), cursor);
            cursor += dev_size(t)?;
        }
        let arena_bytes = (cursor + 4095) & !4095usize;
        if arena_bytes == 0 {
            return Err("container has no payloads".into());
        }

        let (mut free0, mut total) = (0usize, 0usize);
        cuda_ffi::chk(d.cuMemGetInfo(&mut free0, &mut total), "cuMemGetInfo")?;
        if arena_bytes + (256 << 20) > free0 {
            return Err(format!(
                "not enough VRAM: need {:.3} GB + overhead, have {:.3} GB free",
                arena_bytes as f64 / 1e9,
                free0 as f64 / 1e9
            ));
        }

        let mut arena: CUdeviceptr = 0;
        cuda_ffi::chk(d.cuMemAlloc(&mut arena, arena_bytes), "cuMemAlloc(arena)")?;

        let t0 = std::time::Instant::now();
        // One plain pageable copy per tensor.  Measured on a Gen3 x4 host link:
        // a single cudaMemcpyHtoD runs at 3.21-3.29 GB/s, while staging through a
        // page-locked buffer with async copies measured 2.69-2.99 GB/s (the extra
        // host memcpy and the per-chunk syncs cost more than pinning saves), and
        // page-locking the whole model costs host memory proportional to its size.
        // The plain copy is therefore both the fastest option in the common case
        // and the one with no extra footprint.
        let mut host_buf = lightgpu::vm::Staging::chunked();
        let mut uploaded = 0usize;
        let mut tensors = std::collections::HashMap::new();
        // Upload in container order; payload copies are contiguous, so this is a
        // straight sequential stream into VRAM.
        for name in &c.order {
            let t = &c.tensors[name];
            let (addr, nbytes) = match &t.alias_of {
                Some(target) => {
                    let tt = c
                        .tensors
                        .get(target)
                        .ok_or(format!("{name}: bad alias {target}"))?;
                    let off = arena
                        + *dev_off
                            .get(target)
                            .ok_or(format!("{name}: alias target {target} unplaced"))?
                            as u64;
                    (off, dev_size(tt)?)
                }
                None => {
                    let src = c.raw(name)?;
                    let off = arena + *dev_off.get(name).ok_or(format!("{name}: unplaced"))? as u64;
                    let out_bytes = dev_size(t)?;
                    match t.dtype {
                        Dtype::Q8_0 => {
                            // The 34 -> 36-byte stride change is a real transform, so
                            // each chunk is repacked into the staging buffer and copied
                            // out of it (see lightgpu::vm::Staging: reusing one buffer
                            // and keeping the repack cache-hot is worth ~10%, while
                            // page-locking it is worth nothing on this link).
                            let per = (host_buf.len() / Q8_0_BLOCK_DEV) * Q8_0_BLOCK_IN;
                            let mut s_off = 0usize;
                            let mut d_off = 0usize;
                            while s_off < src.len() {
                                let n = per.min(src.len() - s_off);
                                let nout = (n / Q8_0_BLOCK_IN) * Q8_0_BLOCK_DEV;
                                let st = &mut host_buf.as_mut_bytes()[..nout];
                                repack_q8_0_36_into(&src[s_off..s_off + n], st)?;
                                cuda_ffi::chk(
                                    d.cuMemcpyHtoD(
                                        off + d_off as u64,
                                        st.as_ptr() as *const std::ffi::c_void,
                                        nout,
                                    ),
                                    "cuMemcpyHtoD(q8_0 chunked)",
                                )?;
                                s_off += n;
                                d_off += nout;
                            }
                        }
                        _ => {
                            // The payload is copied straight out of the mapped
                            // container - it needs no transform, so bouncing it
                            // through the staging buffer would only add a second
                            // pass over memory.
                            lightgpu::vm::copy_htod(off, src)?;
                        }
                    }
                    uploaded += out_bytes;
                    (off, out_bytes)
                }
            };
            tensors.insert(name.clone(), DeviceTensor { addr, nbytes });
        }
        cuda_ffi::chk(d.cuCtxSynchronize(), "sync(final)")?;
        let dt = t0.elapsed().as_secs_f64();
        let (mut free1, _) = (0usize, 0usize);
        cuda_ffi::chk(d.cuMemGetInfo(&mut free1, &mut total), "cuMemGetInfo")?;
        eprintln!(
            "upload    : {:.3} GB into a {:.3} GB arena in {:.2} s ({:.2} GB/s), VRAM free {:.2} -> {:.2} GiB",
            uploaded as f64 / 1e9,
            arena_bytes as f64 / 1e9,
            dt,
            uploaded as f64 / 1e9 / dt.max(1e-9),
            free0 as f64 / (1u64 << 30) as f64,
            free1 as f64 / (1u64 << 30) as f64,
        );

        Ok(DeviceWeights { arena, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&DeviceTensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("no device tensor `{name}`"))
    }

    /// Backend-aware load: one VRAM arena on the GPU, in-place mmap on the CPU.
    pub fn load(c: &Container) -> Result<DeviceWeights, String> {
        if cuda::cpu_mode() {
            DeviceWeights::from_container(c)
        } else {
            DeviceWeights::upload(c)
        }
    }

    /// CPU mode: point every tensor straight at its payload inside the mmap'd
    /// container. Nothing is copied and nothing is repacked - the container
    /// already stores native 34-byte q8_0 blocks, which is exactly what the CPU
    /// GEMM reads, so the process footprint stays at the mapping size.
    pub fn from_container(c: &Container) -> Result<DeviceWeights, String> {
        let mut tensors = std::collections::HashMap::new();
        for name in &c.order {
            let t = &c.tensors[name];
            // raw() resolves aliases (e.g. lm.output.weight -> lm.tok_embd.weight)
            // so tied weights share the target's address.
            let addr = c.raw(name)?.as_ptr() as u64;
            tensors.insert(
                name.clone(),
                DeviceTensor {
                    addr,
                    nbytes: t.nbytes,
                },
            );
        }
        let bytes = c.payload_bytes();
        eprintln!(
            "weights   : {} tensors mapped in place ({:.3} GB, no upload)",
            tensors.len(),
            bytes as f64 / 1e9
        );
        Ok(DeviceWeights { arena: 0, tensors })
    }
}

/// On-device payload size: Q8_0 grows from the 34-byte to the 36-byte stride.
fn dev_size(t: &crate::container::TensorInfo) -> Result<usize, String> {
    match t.dtype {
        Dtype::Q8_0 => {
            if t.shape.len() != 2 || t.shape[0] % 32 != 0 {
                return Err(format!(
                    "{}: unexpected q8_0 shape {:?} (ne0 must be a multiple of 32)",
                    t.name, t.shape
                ));
            }
            Ok((t.shape[0] / 32) * Q8_0_BLOCK_DEV * t.shape[1])
        }
        _ => Ok(t.nbytes),
    }
}

/// Repack ggml 34-byte q8_0 blocks (2-byte f16 scale + 32 int8) into the
/// aligned 36-byte device stride (f16 scale, 2 pad bytes, 32 int8), into a
/// caller-provided destination (the destination must be nblk * 36 long).
fn repack_q8_0_36_into(src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    if src.len() % Q8_0_BLOCK_IN != 0 {
        return Err(format!(
            "q8_0 payload {} is not a multiple of 34",
            src.len()
        ));
    }
    let nblk = src.len() / Q8_0_BLOCK_IN;
    if dst.len() < nblk * Q8_0_BLOCK_DEV {
        return Err("repack destination too small".into());
    }
    // The payload is dense int8; the only change is the 34 -> 36 byte stride
    // with a 2-byte pad. Copy in 8-byte words where alignment allows.
    for b in 0..nblk {
        let s = &src[b * Q8_0_BLOCK_IN..(b + 1) * Q8_0_BLOCK_IN];
        let o = &mut dst[b * Q8_0_BLOCK_DEV..(b + 1) * Q8_0_BLOCK_DEV];
        o[..2].copy_from_slice(&s[..2]);
        o[2] = 0;
        o[3] = 0;
        o[4..36].copy_from_slice(&s[2..34]);
    }
    Ok(())
}

impl Drop for DeviceWeights {
    fn drop(&mut self) {
        if let Ok(d) = cuda_ffi::driver() {
            let _ = d.cuMemFree(self.arena);
        }
    }
}
