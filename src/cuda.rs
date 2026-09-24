//! Engine layer over the shared toolkit (`lightgpu`).
//!
//! What is *engine-specific* and lives here:
//!   * the `CPU_MODE` switch, which makes `Buffer` a host allocation so the whole
//!     graph can run without a GPU (and so the CPU path is the reference the GPU
//!     path is diffed against);
//!   * `Buffer`, whose address is a `CUdeviceptr` on the GPU and a host address
//!     on the CPU, so every caller treats addresses uniformly;
//!
//! What moved to the toolkit: the driver bindings (`lightgpu::ffi`, re-exported
//! below as `cuda_ffi`), the primary-context bring-up, module load with a
//! memoized kernel-handle cache, device buffers (`lightgpu::vm`), argument
//! marshalling and launch (`lightgpu::vm::{Args, Launch}`), and the fatbin for
//! the kernel set this engine used to own.

/// The driver bindings, under the name the rest of the crate already uses.
pub use lightgpu::ffi as cuda_ffi;

/// Module load/query with the kernel-handle cache: the toolkit's `Module`.
pub use lightgpu::vm::Module;

/// Initialise the driver and attach to device 0's primary context. Returns the
/// error so the caller can fall back to the CPU.
pub fn init() -> Result<(), String> {
    lightgpu::vm::init()
}

/// Backend selection. `ptr` is a CUdeviceptr on the GPU and a plain host
/// address on the CPU, so every caller treats addresses uniformly.
use std::sync::atomic::{AtomicBool, Ordering};

static CPU_MODE: AtomicBool = AtomicBool::new(false);

pub fn set_cpu_mode(on: bool) {
    CPU_MODE.store(on, Ordering::Relaxed);
}

#[inline]
pub fn cpu_mode() -> bool {
    CPU_MODE.load(Ordering::Relaxed)
}

/// What the Buffer must keep alive: the host allocation in CPU mode, or the
/// toolkit's device buffer on the GPU. The payload is deliberately never read -
/// `ptr` is the address callers use - but both variants own their storage and
/// free it on drop, so the field exists purely to tie that lifetime to the
/// Buffer.
#[allow(dead_code)]
enum Store {
    Host(Box<[u8]>),
    Dev(lightgpu::vm::DevBuf),
}

pub struct Buffer {
    pub ptr: u64,
    pub bytes: usize,
    /// Allocation keeper; see `Store`. Never read.
    #[allow(dead_code)]
    store: Option<Store>,
}

impl Buffer {
    /// Zeroed scratch: device memory on the GPU, anonymous host memory on the CPU.
    pub fn alloc(bytes: usize) -> Result<Buffer, String> {
        if cpu_mode() {
            let v = vec![0u8; bytes].into_boxed_slice();
            let ptr = v.as_ptr() as u64;
            return Ok(Buffer {
                ptr,
                bytes,
                store: Some(Store::Host(v)),
            });
        }
        let buf = lightgpu::vm::DevBuf::zeros(bytes)?;
        Ok(Buffer {
            ptr: buf.ptr,
            bytes,
            store: Some(Store::Dev(buf)),
        })
    }

    pub fn upload(&self, data: &[u8]) -> Result<(), String> {
        assert!(data.len() <= self.bytes);
        if cpu_mode() {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr as *mut u8, data.len())
            };
            return Ok(());
        }
        lightgpu::vm::copy_htod(self.ptr, data)
    }

    pub fn download(&self, out: &mut [u8]) -> Result<(), String> {
        assert!(out.len() <= self.bytes);
        if cpu_mode() {
            unsafe {
                std::ptr::copy_nonoverlapping(self.ptr as *const u8, out.as_mut_ptr(), out.len())
            };
            return Ok(());
        }
        lightgpu::vm::copy_dtoh(out, self.ptr)
    }
}

// Both stores free themselves: the host box on drop, the toolkit DevBuf through
// its own Drop (`lightgpu::vm`), so this type needs no raw driver calls at all.


/// The kernel set's fatbin, compiled by the toolkit's build script. Absent in a
/// build without the `cuda` feature (`cargo build --no-default-features`),
/// which is what a machine with no CUDA toolkit uses.
#[cfg(feature = "cuda")]
pub fn embed_fatbin() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/la_kernels.fatbin"))
}
