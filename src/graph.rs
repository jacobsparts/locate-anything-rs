//! Forward passes built from the validated kernels: the Qwen2 LM and (later)
//! the MoonViT encoder plus connector.
//!
//! Layout discipline (mirrors ggml so the weights can be used verbatim):
//!   * weight matrices are [ne1 rows][ne0 cols] with ne0 = input/reduction dim,
//!     ne1 = output dim, rows contiguous over ne0.
//!   * activations are [ne0, nrows] with ne0 contiguous (element (i,t) at
//!     i + ne0*t), i.e. ggml's [ne0, ntok].
//!   * attention tensors are [hd, nh, ntok] - token stride nh*hd.

use crate::cpu;
use crate::cuda::{self, Buffer, Module};
use lightgpu::vm::{Args, Launch};
use crate::cuda::cuda_ffi::{self, CUdeviceptr};
use crate::model::DeviceWeights;

pub struct Lm {
    pub n_layers: usize,
    pub hidden: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

/// Reusable per-step buffers for incremental decoding, so decode_step does not
/// re-allocate (and re-zero) a batch-sized scratch, a position buffer and a
/// vocab-sized logits staging buffer on every token.
pub struct DecodeCtx {
    pub s: Scratch,
    pub pos_b: Buffer,
    pub tmp: Buffer,
    /// Device-side argmax outputs (i32 index + f32 value) for the token loop.
    pub idx: Buffer,
    pub val: Buffer,
}

impl DecodeCtx {
    pub fn new(lm: &Lm) -> Result<DecodeCtx, String> {
        Ok(DecodeCtx {
            s: Scratch::new(lm, 1)?,
            pos_b: Buffer::alloc(4)?,
            tmp: Buffer::alloc(lm.vocab * 4)?,
            idx: Buffer::alloc(4)?,
            val: Buffer::alloc(4)?,
        })
    }
}

pub struct Scratch {
    pub x: Buffer,  // [hidden, seq]
    pub xn: Buffer, // [hidden, seq]
    pub q: Buffer,  // [hd, n_heads, seq]
    pub k: Buffer,
    pub v: Buffer,
    pub attn_out: Buffer, // [hidden, seq]
    pub gate: Buffer,     // [inter, seq]
    pub up: Buffer,       // [inter, seq]
    pub ffn_out: Buffer,  // [hidden, seq]
    pub mask: Buffer,     // [seq_kv, seq_q]
    pub qs: Buffer,       // int8 [seq][inter]  (dp4a activation quantization)
    pub sc: Buffer,       // f32  [seq][inter/32] (per-32-block scales, column-major over blocks)
}

impl Scratch {
    pub fn new(lm: &Lm, seq: usize) -> Result<Scratch, String> {
        Ok(Scratch {
            x: Buffer::alloc(lm.hidden * seq * 4)?,
            xn: Buffer::alloc(lm.hidden * seq * 4)?,
            q: Buffer::alloc(lm.head_dim * lm.n_heads * seq * 4)?,
            k: Buffer::alloc(lm.head_dim * lm.n_kv_heads * seq * 4)?,
            v: Buffer::alloc(lm.head_dim * lm.n_kv_heads * seq * 4)?,
            attn_out: Buffer::alloc(lm.hidden * seq * 4)?,
            gate: Buffer::alloc(lm.inter * seq * 4)?,
            up: Buffer::alloc(lm.inter * seq * 4)?,
            ffn_out: Buffer::alloc(lm.hidden * seq * 4)?,
            mask: Buffer::alloc(seq * seq * 4)?,
            qs: Buffer::alloc(lm.inter * seq)?,
            sc: Buffer::alloc((lm.inter / 32) * seq * 4)?,
        })
    }
}

/// Launch helper bundle: every kernel call goes through these, so the argument
/// marshalling is the toolkit's `vm::Args` (owned, typed) rather than a local
/// erased array.
pub struct K {
    /// None on the CPU backend (no fatbin is loaded at all).
    pub module: Option<Module>,
}

impl K {
    /// Build the kernel bundle. LA_CPU=1 forces the CPU backend; otherwise the
    /// CUDA driver is initialised and any failure (no GPU, no driver, driver
    /// too old) falls back to the CPU backend, so the engine always runs.
    pub fn new() -> Result<K, String> {
        // A build without the `cuda` feature carries no kernels at all.
        #[cfg(not(feature = "cuda"))]
        {
            cuda::set_cpu_mode(true);
            eprintln!("backend   : CPU (built without the `cuda` feature)");
            return Ok(K { module: None });
        }
        #[cfg(feature = "cuda")]
        {
            if std::env::var("LA_CPU").is_ok() {
                cuda::set_cpu_mode(true);
            } else if let Err(e) = cuda::init() {
                eprintln!("cuda      : {e}");
                eprintln!("cuda      : falling back to the CPU backend (LA_CPU=1 forces it)");
                cuda::set_cpu_mode(true);
            }
            if cuda::cpu_mode() {
                eprintln!("backend   : CPU");
                return Ok(K { module: None });
            }
            let module = Module::load(cuda::embed_fatbin())?;
            eprintln!("backend   : CUDA");
            Ok(K {
                module: Some(module),
            })
        }
    }

    /// The module a kernel lives in; only reachable on the CUDA path. `Args::launch`
    /// takes a module rather than a bare handle, which is also what makes a kernel
    /// name resolvable at launch time.
    fn module_of(&self, name: &str) -> Result<&Module, String> {
        match &self.module {
            Some(m) => Ok(m),
            None => Err(format!("kernel `{name}` requested in CPU mode")),
        }
    }

    pub fn sync(&self) -> Result<(), String> {
        if cuda::cpu_mode() {
            return Ok(());
        }
        let d = cuda_ffi::driver()?;
        cuda_ffi::chk(d.cuCtxSynchronize(), "cuCtxSynchronize")
    }

    /// dst[0..bytes] <- src[0..bytes] (device-to-device).
    pub fn copy_bytes(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::copy_bytes(dst, src, bytes);
        }
        let d = cuda_ffi::driver()?;
        cuda_ffi::chk(d.cuMemcpyDtoD(dst, src, bytes), "cuMemcpyDtoD")
    }

    /// host[0..bytes] <- device src.
    pub fn read_bytes(&self, dst: *mut u8, src: CUdeviceptr, bytes: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::read_bytes(dst, src, bytes);
        }
        let d = cuda_ffi::driver()?;
        cuda_ffi::chk(
            d.cuMemcpyDtoH(dst as *mut std::ffi::c_void, src, bytes),
            "cuMemcpyDtoH",
        )
    }

    /// y[ne1, ncols] = W_q8_0[ne1,ne0] * x[ne0,ncols] (scalar reference kernel).
    pub fn gemm_q8(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        self.gemm_q8_scalar_36(w, x, y, ne0, ne1, ncols)
    }

    /// Quantize activations to int8 (32-block, per-column scales) for the dp4a path.
    pub fn quantize_q8_0(
        &self,
        x: CUdeviceptr,
        qs: CUdeviceptr,
        sc: CUdeviceptr,
        ne0: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::quantize_q8_0(x, qs, sc, ne0, ncols);
        }
        let mut aa = Args::new();
        aa.ptr(x).ptr(qs).ptr(sc).i32(ne0 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_quantize_q8_0")?, "lg_quantize_q8_0", Launch::new((((ne0 / 32) as u32), ncols as u32, 1), (32, 1, 1)).shared(0))
    }

    /// Decode GEMV (warp per output row) with an optional bias folded into the
    /// epilogue, so the separate per-layer bias launch disappears.
    pub fn gemm_q8_gemv5(
        &self,
        w: CUdeviceptr,
        qs: CUdeviceptr,
        sc: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
        bias: Option<CUdeviceptr>,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::q8_gemm(w, qs, sc, y, ne0, ne1, ncols, bias);
        }
        let mut aa = Args::new();
        aa.ptr(w).ptr(qs).ptr(sc).ptr(y).i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32).ptr(bias.unwrap_or(0));
        aa.launch(self.module_of("lg_q8_0_gemv")?, "lg_q8_0_gemv", Launch::new((((ne1 + 7) / 8) as u32, ncols as u32, 1), (32 * 8, 1, 1)).shared(ne0 as u32))
    }

    /// v16: v8 tiling over the aligned 36-byte q8_0 layout.
    pub fn gemm_q8_dp4a6(
        &self,
        w: CUdeviceptr,
        qs: CUdeviceptr,
        sc: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::q8_gemm(w, qs, sc, y, ne0, ne1, ncols, None);
        }
        let mut aa = Args::new();
        aa.ptr(w).ptr(qs).ptr(sc).ptr(y).i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_q8_0_gemm_dp4a")?, "lg_q8_0_gemm_dp4a", Launch::new((((ne1 + 63) / 64) as u32, ((ncols + 31) / 32) as u32, 1), (256, 1, 1)).shared(0))
    }

    /// q8_0 GEMM with the fastest available kernel: quantize the activation into
    /// `qs`/`sc` and run the dp4a GEMM over the aligned 36-byte weight layout.
    pub fn gemm_q8_fast(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        qs: CUdeviceptr,
        sc: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        self.quantize_q8_0(x, qs, sc, ne0, ncols)?;
        self.gemm_q8_dp4a6(w, qs, sc, y, ne0, ne1, ncols)
    }

    /// v8-style f32 GEMM (64 rows x 32 columns per block). Requires ne0 % 4 == 0.
    pub fn gemm_f32_tiled2(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::f32_gemm(w, x, y, ne0, ne1, ncols);
        }
        let mut aa = Args::new();
        aa.ptr(w).ptr(x).ptr(y).i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_f32_gemm_tiled")?, "lg_f32_gemm_tiled", Launch::new((((ne1 + 63) / 64) as u32, ((ncols + 31) / 32) as u32, 1), (256, 1, 1)).shared(0))
    }

    /// y[ne1, ncols] = W_f32[ne1,ne0] * x[ne0,ncols]
    pub fn gemm_f32(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        // The tiled kernel uses float4 loads, which need ne0 % 4 == 0 and
        // 16-byte-aligned row starts; fall back to the scalar kernel otherwise
        // (e.g. the 588-wide patch-embed weight).
        if ne0 % 4 != 0 {
            return self.gemm_f32_scalar(w, x, y, ne0, ne1, ncols);
        }
        self.gemm_f32_tiled2(w, x, y, ne0, ne1, ncols)
    }

    /// Scalar f32 GEMM: one row of outputs per block, used when the tiled
    /// kernel's alignment requirement (ne0 % 4 == 0) does not hold.
    pub fn gemm_f32_scalar(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::f32_gemm(w, x, y, ne0, ne1, ncols);
        }
        let mut aa = Args::new();
        aa.ptr(w).ptr(x).ptr(y).i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_f32_gemm")?, "lg_f32_gemm", Launch::new((((ne1 as u32) + 63) / 64, ncols as u32, 1), (64, 1, 1)).shared(0))
    }

    pub fn rms_norm(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        nrows: usize,
        eps: f32,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::rms_norm(x, w, y, ne0, nrows, eps);
        }
        let mut aa = Args::new();
        aa.ptr(x).ptr(w).ptr(y).i32(ne0 as i32).i32(nrows as i32).f32(eps);
        aa.launch(self.module_of("lg_rms_norm")?, "lg_rms_norm", Launch::new((nrows as u32, 1, 1), (256, 1, 1)).shared(0))
    }

    /// x[ne0,nrows] += b[ne0]
    pub fn add_bias(
        &self,
        x: CUdeviceptr,
        b: CUdeviceptr,
        ne0: usize,
        nrows: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::add_bias(x, b, ne0, nrows);
        }
        // lg_row_affine(x, scale, shift, ne0, nrows) computes x[i] = x[i]*scale[i]
        // + shift[i]. A plain bias add is therefore a NULL SCALE and the bias as
        // the SHIFT - passing the bias as the scale would multiply the activation
        // by it instead of adding it.
        let mut aa = Args::new();
        aa.ptr(x).ptr(0).ptr(b).i32(ne0 as i32).i32(nrows as i32);
        aa.launch(self.module_of("lg_row_affine")?, "lg_row_affine", Launch::new((((ne0 as u32) + 255) / 256, nrows as u32, 1), (256, 1, 1)).shared(0))
    }

    /// a += b elementwise over n floats.
    pub fn add_inplace(&self, a: CUdeviceptr, b: CUdeviceptr, n: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::add_inplace(a, b, n);
        }
        let mut aa = Args::new();
        aa.ptr(a).ptr(b).i32(n as i32);
        aa.launch(self.module_of("lg_add_inplace")?, "lg_add_inplace", Launch::new((((n as u32) + 255) / 256, 1, 1), (256, 1, 1)).shared(0))
    }

    pub fn rope_neox(
        &self,
        x: CUdeviceptr,
        pos: CUdeviceptr,
        hd: usize,
        nh: usize,
        ntok: usize,
        theta: f32,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::rope_neox(x, pos, hd, nh, ntok, theta);
        }
        let mut aa = Args::new();
        aa.ptr(x).ptr(pos).i32(hd as i32).i32(nh as i32).i32(ntok as i32).f32(theta);
        aa.launch(self.module_of("lg_rope_neox")?, "lg_rope_neox", Launch::new(((((hd / 2) as u32) + 63) / 64, nh as u32, ntok as u32), (64, 1, 1)).shared(0))
    }

    pub fn attn_gqa(
        &self,
        q: CUdeviceptr,
        k: CUdeviceptr,
        v: CUdeviceptr,
        mask: CUdeviceptr,
        out: CUdeviceptr,
        hd: usize,
        n_qh: usize,
        n_kvh: usize,
        ntq: usize,
        ntk: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::attn(q, k, v, mask, out, hd, n_qh, n_kvh, ntq, ntk);
        }
        let scale = 1.0f32 / (hd as f32).sqrt();
        let jobs = (n_qh * ntq) as u32;
        let warps = 8u32;
        let mut aa = Args::new();
        aa.ptr(q).ptr(k).ptr(v).ptr(mask).ptr(out).i32(hd as i32).i32(n_qh as i32).i32(n_kvh as i32).i32(ntq as i32).i32(ntk as i32).f32(scale);
        aa.launch(self.module_of("lg_attn_gqa")?, "lg_attn_gqa", Launch::new((((jobs) + warps - 1) / warps, 1, 1), (warps * 32, 1, 1)).shared(0))
    }

    /// gate += silu(gate) * up  (in place on gate, elementwise over n)
    pub fn silu_mul(&self, gate: CUdeviceptr, up: CUdeviceptr, n: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::silu_mul(gate, up, n);
        }
        let mut aa = Args::new();
        aa.ptr(gate).ptr(up).i32(n as i32);
        aa.launch(self.module_of("lg_silu_mul")?, "lg_silu_mul", Launch::new((((n as u32) + 255) / 256, 1, 1), (256, 1, 1)).shared(0))
    }

    /// Dequantize ONE row of an aligned 36-byte q8_0 table; row id by value
    /// (no ids upload). grid = (ceil(ne0/256), 1), block = 256.
    pub fn get_row_q8_36(
        &self,
        w: CUdeviceptr,
        row: usize,
        dst: CUdeviceptr,
        ne0: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::get_row_q8(w, row, dst, ne0);
        }
        let mut aa = Args::new();
        aa.ptr(w).i32(row as i32).ptr(dst).i32(ne0 as i32);
        aa.launch(self.module_of("lg_get_row_q8_0_aligned")?, "lg_get_row_q8_0_aligned", Launch::new((((ne0 as u32) + 255) / 256, 1, 1), (256, 1, 1)).shared(0))
    }

    /// Async device-to-device row copy (KV cache append), in 16-byte chunks.
    pub fn copy_row(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::copy_bytes(dst, src, bytes);
        }
        let n4 = (bytes + 15) / 16;
        let mut aa = Args::new();
        aa.ptr(src).ptr(dst).i32(n4 as i32);
        aa.launch(self.module_of("lg_copy_row")?, "lg_copy_row", Launch::new(((((n4 as u32) + 255) / 256).max(1), 1, 1), (256, 1, 1)).shared(0))
    }

    /// Write one i32 into a device buffer (no pageable H2D copy).
    pub fn set_i32(&self, dst: CUdeviceptr, value: i32) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::set_i32(dst, value);
        }
        let mut aa = Args::new();
        aa.ptr(dst).i32(value);
        aa.launch(self.module_of("lg_set_i32")?, "lg_set_i32", Launch::new((1, 1, 1), (1, 1, 1)).shared(0))
    }

    /// Scalar f32-activation GEMM over the aligned 36-byte q8_0 layout.
    pub fn gemm_q8_scalar_36(
        &self,
        w: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        ne1: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::q8_gemm_f32(w, x, y, ne0, ne1, ncols);
        }
        let mut aa = Args::new();
        aa.ptr(w).ptr(x).ptr(y).i32(ne0 as i32).i32(ne1 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_q8_0_gemm_aligned")?, "lg_q8_0_gemm_aligned", Launch::new((((ne1 + 63) / 64) as u32, ((ncols + 31) / 32) as u32, 1), (256, 1, 1)).shared(0))
    }

    pub fn argmax(
        &self,
        x: CUdeviceptr,
        idx: CUdeviceptr,
        val: CUdeviceptr,
        ne0: usize,
        ncols: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::argmax(x, idx, val, ne0, ncols);
        }
        let mut aa = Args::new();
        aa.ptr(x).ptr(idx).ptr(val).i32(ne0 as i32).i32(ncols as i32);
        aa.launch(self.module_of("lg_argmax")?, "lg_argmax", Launch::new((ncols as u32, 1, 1), (256, 1, 1)).shared(0))
    }
    /// GELU tanh approximation (ViT MLP), elementwise in place.
    pub fn gelu_tanh(&self, x: CUdeviceptr, n: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::gelu_tanh(x, n);
        }
        // The toolkit's gelu is out-of-place: (x, y, n). An in-place apply is
        // the same pointer twice, which is safe elementwise (each element is
        // read before it is written, with no cross-element dependence).
        let mut aa = Args::new();
        aa.ptr(x).ptr(x).i32(n as i32);
        aa.launch(self.module_of("lg_gelu_tanh")?, "lg_gelu_tanh", Launch::new((((n as u32) + 255) / 256, 1, 1), (256, 1, 1)).shared(0))
    }

    /// y = 0.5 x (1 + erf(x/sqrt2)), elementwise in place.
    pub fn gelu_erf(&self, x: CUdeviceptr, n: usize) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::gelu_erf(x, n);
        }
        // Out-of-place (x, y, n); in-place is the same pointer twice. See the
        // note on gelu_tanh above.
        let mut aa = Args::new();
        aa.ptr(x).ptr(x).i32(n as i32);
        aa.launch(self.module_of("lg_gelu_erf")?, "lg_gelu_erf", Launch::new((((n as u32) + 255) / 256, 1, 1), (256, 1, 1)).shared(0))
    }

    /// dst [nrows, ntok] <- src [src_stride, ntok] (first nrows rows).
    pub fn extract_rows(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        src_stride: usize,
        nrows: usize,
        ntok: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::extract_rows(dst, src, src_stride, nrows, ntok);
        }
        // One block per token (coalesced reads and writes), threads spanning rows.
        let mut aa = Args::new();
        aa.ptr(dst).ptr(src).i32(src_stride as i32).i32(nrows as i32).i32(ntok as i32);
        aa.launch(self.module_of("lg_extract_rows")?, "lg_extract_rows", Launch::new((ntok as u32, (((nrows as u32) + 255) / 256).max(1), 1), (256, 1, 1)).shared(0))
    }

    /// 2x2 patch merge: out [4C, m] <- vf [C, gh*gw].
    pub fn merge_2x2(
        &self,
        out: CUdeviceptr,
        vf: CUdeviceptr,
        gh: usize,
        gw: usize,
        c: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::merge_2x2(out, vf, gh, gw, c);
        }
        let m = (gh / 2) * (gw / 2);
        let mut aa = Args::new();
        aa.ptr(out).ptr(vf).i32(gh as i32).i32(gw as i32).i32(c as i32);
        aa.launch(self.module_of("lg_merge_2x2")?, "lg_merge_2x2", Launch::new(((((4 * c) as u32) + 255) / 256, m as u32, 1), (256, 1, 1)).shared(0))
    }

    /// 2-D RoPE with host-built pair-major tables (cos[k*ntok + t]).
    pub fn rope_2d(
        &self,
        x: CUdeviceptr,
        cos: CUdeviceptr,
        sin: CUdeviceptr,
        hd: usize,
        nh: usize,
        ntok: usize,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::rope_2d(x, cos, sin, hd, nh, ntok);
        }
        let np = hd / 2;
        let mut aa = Args::new();
        aa.ptr(x).ptr(cos).ptr(sin).i32(hd as i32).i32(nh as i32).i32(ntok as i32);
        aa.launch(self.module_of("lg_rope_2d")?, "lg_rope_2d", Launch::new((((np as u32) + 63) / 64, nh as u32, ntok as u32), (64, 1, 1)).shared(0))
    }

    /// LayerNorm (w+b, eps) over rows of [ne0, nrows].
    pub fn layer_norm(
        &self,
        x: CUdeviceptr,
        w: CUdeviceptr,
        b: CUdeviceptr,
        y: CUdeviceptr,
        ne0: usize,
        nrows: usize,
        eps: f32,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::layer_norm(x, w, b, y, ne0, nrows, eps);
        }
        let mut aa = Args::new();
        aa.ptr(x).ptr(w).ptr(b).ptr(y).i32(ne0 as i32).i32(nrows as i32).f32(eps);
        aa.launch(self.module_of("lg_layer_norm")?, "lg_layer_norm", Launch::new((nrows as u32, 1, 1), (256, 1, 1)).shared(0))
    }

    /// Self-attention with K/V staged in shared memory (see lg_attn_flash).
    /// Same arithmetic as attn_gqa; a block owns (head, W query tokens) so each
    /// K/V element is read from DRAM once per block instead of once per query.
    pub fn attn_flash(
        &self,
        q: CUdeviceptr,
        k: CUdeviceptr,
        v: CUdeviceptr,
        out: CUdeviceptr,
        hd: usize,
        nh: usize,
        ntok: usize,
        mask: CUdeviceptr,
    ) -> Result<(), String> {
        if cuda::cpu_mode() {
            return cpu::attn(q, k, v, mask, out, hd, nh, nh, ntok, ntok);
        }
        let scale = 1.0f32 / (hd as f32).sqrt();
        // Query tokens per block and key chunk staged in shared memory.
        let w: usize = 16;
        let kc: usize = 32;
        let shared = (2 * kc * hd * 4) as u32;
        let mut aa = Args::new();
        aa.ptr(q).ptr(k).ptr(v).ptr(mask).ptr(out).i32(hd as i32).i32(nh as i32).i32(ntok as i32).i32(ntok as i32).f32(scale).i32(kc as i32);
        aa.launch(self.module_of("lg_attn_flash")?, "lg_attn_flash", Launch::new((nh as u32, (((ntok + w - 1) / w).max(1)) as u32, 1), ((w * 32) as u32, 1, 1)).shared(shared))
    }

    /// Attention over ALL keys (no mask): q,k,v [hd, nh, ntok].
    pub fn attn_full(
        &self,
        q: CUdeviceptr,
        k: CUdeviceptr,
        v: CUdeviceptr,
        out: CUdeviceptr,
        hd: usize,
        nh: usize,
        ntok: usize,
    ) -> Result<(), String> {
        // The naive warp-per-query kernel re-reads every K/V row for every query
        // token (measured 41 ms/layer vs 21 ms/layer here), so the shared-memory
        // staged kernel is the only path.
        self.attn_flash(q, k, v, out, hd, nh, ntok, 0)
    }
}

/// One Qwen2 decoder layer, mirroring src/qwen2.cpp of the reference engine:
///   xn = rms_norm(x, attn_norm)
///   q  = W_q xn + b_q   (reshaped [hd, n_heads, seq])
///   k  = W_k xn + b_k   ([hd, n_kv_heads, seq]); v likewise
///   NEOX RoPE on q and k
///   o  = attn(q, k, v, mask)      -> [hd, n_heads, seq]
///   o  = W_o o                    (no bias)
///   h  = x + o
///   hn = rms_norm(h, ffn_norm)
///   f  = W_down (silu(W_gate hn) * (W_up hn))
///   y  = h + f
pub struct LayerNames {
    pub attn_norm: String,
    pub wq: String,
    pub bq: String,
    pub wk: String,
    pub bk: String,
    pub wv: String,
    pub bv: String,
    pub wo: String,
    pub ffn_norm: String,
    pub w_gate: String,
    pub w_up: String,
    pub w_down: String,
}

pub fn lm_layer_names(i: usize) -> LayerNames {
    let p = format!("lm.blk.{i}.");
    LayerNames {
        attn_norm: format!("{p}attn_norm.weight"),
        wq: format!("{p}attn_q.weight"),
        bq: format!("{p}attn_q.bias"),
        wk: format!("{p}attn_k.weight"),
        bk: format!("{p}attn_k.bias"),
        wv: format!("{p}attn_v.weight"),
        bv: format!("{p}attn_v.bias"),
        wo: format!("{p}attn_o.weight"),
        ffn_norm: format!("{p}ffn_norm.weight"),
        w_gate: format!("{p}ffn_gate.weight"),
        w_up: format!("{p}ffn_up.weight"),
        w_down: format!("{p}ffn_down.weight"),
    }
}

impl Lm {
    /// Additive causal mask, absolute positions: mask[tq][tk] = 0 if tk <= tq else -1e30.
    /// Layout matches lg_attn_gqa: element (tq, tk) at tq*ntk + tk.
    pub fn causal_mask(ntq: usize, ntk: usize) -> Vec<f32> {
        let mut m = vec![0f32; ntq * ntk];
        for tq in 0..ntq {
            for tk in 0..ntk {
                if tk > tq {
                    m[tq * ntk + tk] = -1e30;
                }
            }
        }
        m
    }
}

/// Per-layer KV cache for incremental decoding.
pub struct KvCache {
    pub k: Vec<Buffer>, // [hd, n_kv_heads, max_seq] each
    pub v: Vec<Buffer>,
    pub mask: Buffer, // [max_seq] zeroed attention mask (decoding is unmasked)
    pub max_seq: usize,
    pub len: usize, // committed entries
}

impl KvCache {
    pub fn new(lm: &Lm, max_seq: usize) -> Result<KvCache, String> {
        let mut k = Vec::with_capacity(lm.n_layers);
        let mut v = Vec::with_capacity(lm.n_layers);
        let n = lm.head_dim * lm.n_kv_heads * max_seq * 4;
        for _ in 0..lm.n_layers {
            k.push(Buffer::alloc(n)?);
            v.push(Buffer::alloc(n)?);
        }
        let mask = Buffer::alloc(max_seq * 4)?; // Buffer::alloc zeroes
        Ok(KvCache {
            k,
            v,
            mask,
            max_seq,
            len: 0,
        })
    }
}

impl Lm {
    /// One incremental step: a single token embedding [hidden] at absolute
    /// position `pos`, with the past in `kv`. Writes the token's k/v into the
    /// cache at slot `pos` and returns logits ([vocab] on device).
    pub fn decode_step(
        &self,
        k: &K,
        w: &DeviceWeights,
        emb: &[f32],
        tok_id: usize,
        pos: usize,
        kv: &mut KvCache,
        logits: &Option<&Buffer>,
        ctx: &DecodeCtx,
    ) -> Result<(), String> {
        if emb.len() != self.hidden {
            return Err("bad embedding row".into());
        }
        if pos >= kv.max_seq {
            return Err("kv cache overflow".into());
        }
        let s = &ctx.s;
        // The embedding row is gathered on the device: a pageable H2D copy per
        // step would flush the pipeline (see lg_get_row_q8_0_aligned).
        let te = w.get("lm.tok_embd.weight")?;
        k.get_row_q8_36(te.addr, tok_id, s.x.ptr, self.hidden)?;
        let pos_b = &ctx.pos_b;
        k.set_i32(pos_b.ptr, pos as i32)?;
        // Unmasked attention over the cache prefix (ntk = pos + 1): the kernel still
        // reads a mask row, so pass the cache's permanently-zeroed buffer (sizing
        // s.mask for ntk here would overflow the seq=1 scratch allocation).
        let ntk = pos + 1;

        for i in 0..self.n_layers {
            let n = lm_layer_names(i);
            let an = w.get(&n.attn_norm)?;
            k.rms_norm(s.x.ptr, an.addr, s.xn.ptr, self.hidden, 1, self.eps)?;
            let wq = w.get(&n.wq)?;
            let bq = w.get(&n.bq)?;
            let wk = w.get(&n.wk)?;
            let bk = w.get(&n.bk)?;
            let wv = w.get(&n.wv)?;
            let bv = w.get(&n.bv)?;
            // Coalesced warp-per-row GEMV: quantize the 2048-wide normalized
            // activations once and reuse them for q, k and v.
            k.quantize_q8_0(s.xn.ptr, s.qs.ptr, s.sc.ptr, self.hidden, 1)?;
            k.gemm_q8_gemv5(
                wq.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.q.ptr,
                self.hidden,
                self.head_dim * self.n_heads,
                1,
                Some(bq.addr),
            )?;
            k.gemm_q8_gemv5(
                wk.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.k.ptr,
                self.hidden,
                self.head_dim * self.n_kv_heads,
                1,
                Some(bk.addr),
            )?;
            k.gemm_q8_gemv5(
                wv.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.v.ptr,
                self.hidden,
                self.head_dim * self.n_kv_heads,
                1,
                Some(bv.addr),
            )?;
            k.rope_neox(
                s.q.ptr,
                pos_b.ptr,
                self.head_dim,
                self.n_heads,
                1,
                self.rope_theta,
            )?;
            k.rope_neox(
                s.k.ptr,
                pos_b.ptr,
                self.head_dim,
                self.n_kv_heads,
                1,
                self.rope_theta,
            )?;
            // append k/v into the cache slot for this position
            let row = self.head_dim * self.n_kv_heads * 4;
            let off = (pos * row) as u64;
            // Async kernel copy: a synchronous cuMemcpyDtoD per layer (72 per step)
            // flushes the pipeline and dominates the decode profile.
            k.copy_row(s.k.ptr, kv.k[i].ptr + off, row)?;
            k.copy_row(s.v.ptr, kv.v[i].ptr + off, row)?;

            k.attn_gqa(
                s.q.ptr,
                kv.k[i].ptr,
                kv.v[i].ptr,
                kv.mask.ptr,
                s.attn_out.ptr,
                self.head_dim,
                self.n_heads,
                self.n_kv_heads,
                1,
                ntk,
            )?;
            let wo = w.get(&n.wo)?;
            // attn_out is not the quantized activation, so quantize it here.
            k.quantize_q8_0(s.attn_out.ptr, s.qs.ptr, s.sc.ptr, self.hidden, 1)?;
            k.gemm_q8_gemv5(
                wo.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.xn.ptr,
                self.hidden,
                self.hidden,
                1,
                None,
            )?;
            k.add_inplace(s.x.ptr, s.xn.ptr, self.hidden)?;

            let fnm = w.get(&n.ffn_norm)?;
            k.rms_norm(s.x.ptr, fnm.addr, s.xn.ptr, self.hidden, 1, self.eps)?;
            let wg = w.get(&n.w_gate)?;
            let wu = w.get(&n.w_up)?;
            let wd = w.get(&n.w_down)?;
            k.quantize_q8_0(s.xn.ptr, s.qs.ptr, s.sc.ptr, self.hidden, 1)?;
            k.gemm_q8_gemv5(
                wg.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.gate.ptr,
                self.hidden,
                self.inter,
                1,
                None,
            )?;
            k.gemm_q8_gemv5(
                wu.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.up.ptr,
                self.hidden,
                self.inter,
                1,
                None,
            )?;
            k.silu_mul(s.gate.ptr, s.up.ptr, self.inter)?;
            k.quantize_q8_0(s.gate.ptr, s.qs.ptr, s.sc.ptr, self.inter, 1)?;
            k.gemm_q8_gemv5(
                wd.addr,
                s.qs.ptr,
                s.sc.ptr,
                s.ffn_out.ptr,
                self.inter,
                self.hidden,
                1,
                None,
            )?;
            k.add_inplace(s.x.ptr, s.ffn_out.ptr, self.hidden)?;
        }
        let on = w.get("lm.output_norm.weight")?;
        k.rms_norm(s.x.ptr, on.addr, s.xn.ptr, self.hidden, 1, self.eps)?;
        let head = w.get("lm.output.weight")?;
        let tmp = &ctx.tmp;
        k.quantize_q8_0(s.xn.ptr, s.qs.ptr, s.sc.ptr, self.hidden, 1)?;
        k.gemm_q8_gemv5(
            head.addr,
            s.qs.ptr,
            s.sc.ptr,
            tmp.ptr,
            self.hidden,
            self.vocab,
            1,
            None,
        )?;
        match logits {
            // Device-side argmax: the caller only needs the token id, so the
            // 610 KB logits round trip is replaced by an 8-byte readback.
            None => {
                k.argmax(tmp.ptr, ctx.idx.ptr, ctx.val.ptr, self.vocab, 1)?;
            }
            Some(lg) => {
                k.copy_bytes(lg.ptr, tmp.ptr, self.vocab * 4)?;
            }
        }
        if pos + 1 > kv.len {
            kv.len = pos + 1;
        }
        Ok(())
    }

    /// Prefill the whole prompt (positions 0..seq) in one batch AND fill the KV
    /// cache layer by layer, returning the last-position logits. This makes the
    /// incremental decode_step path usable.
    pub fn prefill_into_cache(
        &self,
        k: &K,
        w: &DeviceWeights,
        embeds: &[f32],
        seq: usize,
        kv: &mut KvCache,
        logits: &Buffer,
    ) -> Result<(), String> {
        if seq == 0 || embeds.len() != self.hidden * seq {
            return Err("bad embed buffer".into());
        }
        if seq > kv.max_seq {
            return Err("kv cache too small".into());
        }
        let s = Scratch::new(self, seq)?;
        s.x.upload(unsafe {
            std::slice::from_raw_parts(embeds.as_ptr() as *const u8, embeds.len() * 4)
        })?;
        let pos: Vec<i32> = (0..seq as i32).collect();
        let pos_b = Buffer::alloc(seq * 4)?;
        pos_b.upload(unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, seq * 4) })?;
        let mask = Lm::causal_mask(seq, seq);
        s.mask.upload(unsafe {
            std::slice::from_raw_parts(mask.as_ptr() as *const u8, seq * seq * 4)
        })?;
        let row = self.head_dim * self.n_kv_heads * 4;
        for i in 0..self.n_layers {
            let n = lm_layer_names(i);
            let an = w.get(&n.attn_norm)?;
            k.rms_norm(s.x.ptr, an.addr, s.xn.ptr, self.hidden, seq, self.eps)?;
            let wq = w.get(&n.wq)?;
            let bq = w.get(&n.bq)?;
            let wk = w.get(&n.wk)?;
            let bk = w.get(&n.bk)?;
            let wv = w.get(&n.wv)?;
            let bv = w.get(&n.bv)?;
            k.gemm_q8_fast(
                wq.addr,
                s.xn.ptr,
                s.q.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.head_dim * self.n_heads,
                seq,
            )?;
            k.add_bias(s.q.ptr, bq.addr, self.head_dim * self.n_heads, seq)?;
            k.gemm_q8_fast(
                wk.addr,
                s.xn.ptr,
                s.k.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.head_dim * self.n_kv_heads,
                seq,
            )?;
            k.add_bias(s.k.ptr, bk.addr, self.head_dim * self.n_kv_heads, seq)?;
            k.gemm_q8_fast(
                wv.addr,
                s.xn.ptr,
                s.v.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.head_dim * self.n_kv_heads,
                seq,
            )?;
            k.add_bias(s.v.ptr, bv.addr, self.head_dim * self.n_kv_heads, seq)?;
            k.rope_neox(
                s.q.ptr,
                pos_b.ptr,
                self.head_dim,
                self.n_heads,
                seq,
                self.rope_theta,
            )?;
            k.rope_neox(
                s.k.ptr,
                pos_b.ptr,
                self.head_dim,
                self.n_kv_heads,
                seq,
                self.rope_theta,
            )?;
            // whole-chunk copy into cache slots [0, seq)
            k.copy_bytes(kv.k[i].ptr, s.k.ptr, row * seq)?;
            k.copy_bytes(kv.v[i].ptr, s.v.ptr, row * seq)?;
            k.attn_gqa(
                s.q.ptr,
                kv.k[i].ptr,
                kv.v[i].ptr,
                s.mask.ptr,
                s.attn_out.ptr,
                self.head_dim,
                self.n_heads,
                self.n_kv_heads,
                seq,
                seq,
            )?;
            let wo = w.get(&n.wo)?;
            k.gemm_q8_fast(
                wo.addr,
                s.attn_out.ptr,
                s.xn.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.hidden,
                seq,
            )?;
            k.add_inplace(s.x.ptr, s.xn.ptr, self.hidden * seq)?;
            let fnm = w.get(&n.ffn_norm)?;
            k.rms_norm(s.x.ptr, fnm.addr, s.xn.ptr, self.hidden, seq, self.eps)?;
            let wg = w.get(&n.w_gate)?;
            let wu = w.get(&n.w_up)?;
            let wd = w.get(&n.w_down)?;
            k.gemm_q8_fast(
                wg.addr,
                s.xn.ptr,
                s.gate.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.inter,
                seq,
            )?;
            k.gemm_q8_fast(
                wu.addr,
                s.xn.ptr,
                s.up.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.inter,
                seq,
            )?;
            k.silu_mul(s.gate.ptr, s.up.ptr, self.inter * seq)?;
            k.gemm_q8_fast(
                wd.addr,
                s.gate.ptr,
                s.ffn_out.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.inter,
                self.hidden,
                seq,
            )?;
            k.add_inplace(s.x.ptr, s.ffn_out.ptr, self.hidden * seq)?;
        }
        let on = w.get("lm.output_norm.weight")?;
        k.rms_norm(s.x.ptr, on.addr, s.xn.ptr, self.hidden, seq, self.eps)?;
        let head = w.get("lm.output.weight")?;
        let last = s.xn.ptr + ((seq - 1) * self.hidden * 4) as u64;
        let tmp = Buffer::alloc(self.vocab * 4)?;
        k.gemm_q8(head.addr, last, tmp.ptr, self.hidden, self.vocab, 1)?;
        k.copy_bytes(logits.ptr, tmp.ptr, self.vocab * 4)?;
        kv.len = seq;
        Ok(())
    }
}

/// MoonViT encoder + 2x2 merge + connector (the image path of LocateAnything).
pub struct Vit {
    pub n_layers: usize,
    pub hidden: usize,   // 1152
    pub head_dim: usize, // 72
    pub n_heads: usize,  // 16
    pub inter: usize,    // 4304
    pub eps: f32,        // 1e-5
    pub gh: usize,
    pub gw: usize,
    pub rope_theta: f32, // 1e4
}

pub struct VitScratch {
    pub x: Buffer,   // [hidden, ntok] current activations
    pub xn: Buffer,  // [hidden, ntok] normalized
    pub qkv: Buffer, // [(3*hidden), ntok]
    pub q: Buffer,   // [hidden, ntok]
    pub k: Buffer,
    pub v: Buffer,
    pub attn: Buffer,   // [hidden, ntok]
    pub ff: Buffer,     // [inter, ntok]
    pub vf: Buffer,     // [hidden, ntok] post final_norm
    pub merged: Buffer, // [4*hidden, m]
    pub cos: Buffer,    // rope tables [npairs, ntok]
    pub sin: Buffer,
    pub posemb: Buffer, // [hidden, ntok] resized pos-emb
    pub pixels: Buffer, // [588, ntok]
    pub qs: Buffer,     // int8 [ntok][kmax] activation quantization scratch
    pub sc: Buffer,     // f32 [ntok][kmax/32]
}

impl VitScratch {
    pub fn new(v: &Vit) -> Result<VitScratch, String> {
        let ntok = v.gh * v.gw;
        let m = (v.gh / 2) * (v.gw / 2);
        let npairs = v.head_dim / 2;
        // largest reduction dim any ViT GEMM sees: fc0 (inter) and wqkv (3*hidden).
        let kmax = v.inter.max(3 * v.hidden);
        Ok(VitScratch {
            x: Buffer::alloc(v.hidden * ntok * 4)?,
            xn: Buffer::alloc(v.hidden * ntok * 4)?,
            qkv: Buffer::alloc(3 * v.hidden * ntok * 4)?,
            q: Buffer::alloc(v.hidden * ntok * 4)?,
            k: Buffer::alloc(v.hidden * ntok * 4)?,
            v: Buffer::alloc(v.hidden * ntok * 4)?,
            attn: Buffer::alloc(v.hidden * ntok * 4)?,
            ff: Buffer::alloc(v.inter * ntok * 4)?,
            vf: Buffer::alloc(v.hidden * ntok * 4)?,
            merged: Buffer::alloc(4 * v.hidden * m * 4)?,
            cos: Buffer::alloc(npairs * ntok * 4)?,
            sin: Buffer::alloc(npairs * ntok * 4)?,
            posemb: Buffer::alloc(v.hidden * ntok * 4)?,
            pixels: Buffer::alloc(588 * ntok * 4)?,
            qs: Buffer::alloc(kmax * ntok)?,
            sc: Buffer::alloc((kmax / 32) * ntok * 4)?,
        })
    }
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

impl Vit {
    pub fn ntok(&self) -> usize {
        self.gh * self.gw
    }
    pub fn nmerged(&self) -> usize {
        (self.gh / 2) * (self.gw / 2)
    }

    /// Run the encoder and return the merged features [4*hidden, m] on the host.
    pub fn forward(
        &self,
        k: &K,
        w: &DeviceWeights,
        pixel_values: &[f32],
    ) -> Result<Vec<f32>, String> {
        let ntok = self.ntok();
        let s = VitScratch::new(self)?;
        s.pixels.upload(f32_bytes(pixel_values))?;

        // ---- patch embedding: y = W[588,1152]^T x + b + pos_emb ----
        let pe_w = w.get("vit.patch_embed.weight")?; // [14,14,3,1152] -> [1152,588]
        let pe_b = w.get("vit.patch_embed.bias")?; // [1152]
        k.gemm_f32(pe_w.addr, s.pixels.ptr, s.x.ptr, 588, self.hidden, ntok)?;
        k.add_bias(s.x.ptr, pe_b.addr, self.hidden, ntok)?;

        // positional embedding: bicubic-resize the learned 64x64 map on the host
        let pos_src = w.get("vit.pos_emb.weight")?;
        let mut host_pos = vec![0f32; pos_src.nbytes / 4];
        let out = unsafe {
            std::slice::from_raw_parts_mut(host_pos.as_mut_ptr() as *mut u8, pos_src.nbytes)
        };
        k.read_bytes(out.as_mut_ptr(), pos_src.addr, pos_src.nbytes)?;
        // resized is already [tok][ch] with tok = h*gw + w and ch contiguous, which is
        // exactly this engine's activation layout (token-major, feature contiguous).
        let pos_tok =
            crate::image::bicubic_pos_emb(&host_pos, 64, 64, self.hidden, self.gh, self.gw);
        s.posemb.upload(f32_bytes(&pos_tok))?;
        k.add_inplace(s.x.ptr, s.posemb.ptr, self.hidden * ntok)?;

        // ---- 2-D RoPE tables (pair-major) ----
        let (cos, sin) =
            crate::image::rope_2d_tables(self.gh, self.gw, self.head_dim, self.rope_theta);
        s.cos.upload(f32_bytes(&cos))?;
        s.sin.upload(f32_bytes(&sin))?;

        for i in 0..self.n_layers {
            let p = format!("vit.blk.{i}.");
            let n0w = w.get(&format!("{p}norm0.weight"))?;
            let n0b = w.get(&format!("{p}norm0.bias"))?;
            k.layer_norm(
                s.x.ptr,
                n0w.addr,
                n0b.addr,
                s.xn.ptr,
                self.hidden,
                ntok,
                self.eps,
            )?;
            let wqkv = w.get(&format!("{p}wqkv.weight"))?;
            let bqkv = w.get(&format!("{p}wqkv.bias"))?;
            k.gemm_q8_fast(
                wqkv.addr,
                s.xn.ptr,
                s.qkv.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                3 * self.hidden,
                ntok,
            )?;
            k.add_bias(s.qkv.ptr, bqkv.addr, 3 * self.hidden, ntok)?;
            k.extract_rows(s.q.ptr, s.qkv.ptr, 3 * self.hidden, self.hidden, ntok)?;
            k.extract_rows(
                s.k.ptr,
                s.qkv.ptr + (self.hidden * 4) as u64,
                3 * self.hidden,
                self.hidden,
                ntok,
            )?;
            k.extract_rows(
                s.v.ptr,
                s.qkv.ptr + (2 * self.hidden * 4) as u64,
                3 * self.hidden,
                self.hidden,
                ntok,
            )?;
            k.rope_2d(
                s.q.ptr,
                s.cos.ptr,
                s.sin.ptr,
                self.head_dim,
                self.n_heads,
                ntok,
            )?;
            k.rope_2d(
                s.k.ptr,
                s.cos.ptr,
                s.sin.ptr,
                self.head_dim,
                self.n_heads,
                ntok,
            )?;
            k.attn_full(
                s.q.ptr,
                s.k.ptr,
                s.v.ptr,
                s.attn.ptr,
                self.head_dim,
                self.n_heads,
                ntok,
            )?;
            let wo = w.get(&format!("{p}wo.weight"))?;
            let wb = w.get(&format!("{p}wo.bias"))?;
            k.gemm_q8_fast(
                wo.addr,
                s.attn.ptr,
                s.xn.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.hidden,
                ntok,
            )?;
            k.add_bias(s.xn.ptr, wb.addr, self.hidden, ntok)?;
            k.add_inplace(s.x.ptr, s.xn.ptr, self.hidden * ntok)?;

            let n1w = w.get(&format!("{p}norm1.weight"))?;
            let n1b = w.get(&format!("{p}norm1.bias"))?;
            k.layer_norm(
                s.x.ptr,
                n1w.addr,
                n1b.addr,
                s.xn.ptr,
                self.hidden,
                ntok,
                self.eps,
            )?;
            let f0w = w.get(&format!("{p}fc0.weight"))?;
            let f0b = w.get(&format!("{p}fc0.bias"))?;
            k.gemm_q8_fast(
                f0w.addr,
                s.xn.ptr,
                s.ff.ptr,
                s.qs.ptr,
                s.sc.ptr,
                self.hidden,
                self.inter,
                ntok,
            )?;
            k.add_bias(s.ff.ptr, f0b.addr, self.inter, ntok)?;
            k.gelu_tanh(s.ff.ptr, self.inter * ntok)?;
            let f1w = w.get(&format!("{p}fc1.weight"))?;
            let f1b = w.get(&format!("{p}fc1.bias"))?;
            k.gemm_f32(f1w.addr, s.ff.ptr, s.xn.ptr, self.inter, self.hidden, ntok)?;
            k.add_bias(s.xn.ptr, f1b.addr, self.hidden, ntok)?;
            k.add_inplace(s.x.ptr, s.xn.ptr, self.hidden * ntok)?;
        }

        let fnw = w.get("vit.final_norm.weight")?;
        let fnb = w.get("vit.final_norm.bias")?;
        k.layer_norm(
            s.x.ptr,
            fnw.addr,
            fnb.addr,
            s.vf.ptr,
            self.hidden,
            ntok,
            self.eps,
        )?;

        let m = self.nmerged();
        k.merge_2x2(s.merged.ptr, s.vf.ptr, self.gh, self.gw, self.hidden)?;

        let mut out = vec![0f32; 4 * self.hidden * m];
        let ob =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        s.merged.download(ob)?;
        Ok(out)
    }
}

/// Connector: layernorm(proj.0 w+b) -> proj.1 + b -> GELU(erf) -> proj.3 + b.
pub fn connector(
    k: &K,
    w: &DeviceWeights,
    merged: &[f32],
    m: usize,
    hidden4: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, String> {
    let x = Buffer::alloc(hidden4 * m * 4)?;
    x.upload(f32_bytes(merged))?;
    let xn = Buffer::alloc(hidden4 * m * 4)?;
    let g0w = w.get("proj.0.weight")?;
    let g0b = w.get("proj.0.bias")?;
    k.layer_norm(x.ptr, g0w.addr, g0b.addr, xn.ptr, hidden4, m, eps)?;
    let p1w = w.get("proj.1.weight")?;
    let p1b = w.get("proj.1.bias")?;
    let y = Buffer::alloc(hidden * m * 4)?;
    // activation-quantization scratch for the dp4a path (largest ne0 = hidden4)
    let qs_c = Buffer::alloc(hidden4 * m)?;
    let sc_c = Buffer::alloc((hidden4 / 32) * m * 4)?;
    k.gemm_q8_fast(
        p1w.addr, xn.ptr, y.ptr, qs_c.ptr, sc_c.ptr, hidden4, hidden, m,
    )?;
    k.add_bias(y.ptr, p1b.addr, hidden, m)?;
    k.gelu_erf(y.ptr, hidden * m)?;
    let p3w = w.get("proj.3.weight")?;
    let p3b = w.get("proj.3.bias")?;
    let z = Buffer::alloc(hidden * m * 4)?;
    k.gemm_q8_fast(
        p3w.addr, y.ptr, z.ptr, qs_c.ptr, sc_c.ptr, hidden, hidden, m,
    )?;
    k.add_bias(z.ptr, p3b.addr, hidden, m)?;
    let mut out = vec![0f32; hidden * m];
    let ob = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    z.download(ob)?;
    Ok(out)
}
