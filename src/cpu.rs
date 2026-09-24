//! CPU backend: Rust implementations of every kernel the model path uses, so
//! the engine runs without a GPU.
//!
//! Addresses are `u64` exactly like `CUdeviceptr`, so `graph::K` can dispatch on
//! a single global flag. In CPU mode weights are NOT copied: `DeviceWeights`
//! points straight into the mmap'd `.laqt` payloads, which already hold native
//! ggml 34-byte q8_0 blocks (f16 scale + 32 int8). So there is no repacking, no
//! 36-byte device layout and no multi-gigabyte upload - peak RSS stays close to
//! the mapping.
//!
//! The q8_0 GEMM follows the same algorithm as the dp4a kernels (and ggml's
//! reference vec_dot): quantize the activation to int8 with one scale per
//! 32-element block, accumulate the integer dot product, then scale by
//! d_weight * d_activation.

use rayon::prelude::*;

/// Native ggml q8_0 block: 2-byte f16 scale followed by 32 int8 values.
const BLK: usize = 34;
const QOFF: usize = 2;

/// Wrapper making a raw address Send+Sync so it can be captured by rayon.
#[derive(Clone, Copy)]
struct P<T>(*mut T);

impl<T> P<T> {
    #[inline(always)]
    fn new(v: *const T) -> P<T> {
        P(v as *mut T)
    }

    /// Offset accessor: going through a method (rather than `.0`) keeps the
    /// closure capturing the whole wrapper, which is what makes it Send+Sync
    /// under Rust 2021's disjoint closure captures.
    #[inline(always)]
    unsafe fn at(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}
unsafe impl<T> Send for P<T> {}
unsafe impl<T> Sync for P<T> {}

#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            let (mut e, mut m) = (127i32 - 15 + 1, man);
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline(always)]
fn blk_scale(p: *const u8) -> f32 {
    f16_to_f32(u16::from_le_bytes(unsafe { [*p, *p.add(1)] }))
}

/// Round to nearest, ties to even (CUDA __float2int_rn).
#[inline(always)]
fn rn(v: f32) -> i32 {
    v.round_ties_even() as i32
}

#[inline(always)]
fn row_off(row: usize, ne0: usize) -> usize {
    (row * (ne0 / 32)) * BLK
}

// ------------------------------------------------------------------ copies

pub fn copy_bytes(dst: u64, src: u64, bytes: usize) -> Result<(), String> {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
    Ok(())
}

pub fn read_bytes(dst: *mut u8, src: u64, bytes: usize) -> Result<(), String> {
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst, bytes) };
    Ok(())
}

pub fn set_i32(dst: u64, value: i32) -> Result<(), String> {
    unsafe { *(dst as *mut i32) = value };
    Ok(())
}

// -------------------------------------------------------------------- GEMM

/// Quantize x[ncols][ne0] (ne0 contiguous) to int8 blocks with per-32 scales.
/// Scales land at sc[c * nb + b], the layout the GEMMs walk.
pub fn quantize_q8_0(x: u64, qs: u64, sc: u64, ne0: usize, ncols: usize) -> Result<(), String> {
    let nb = ne0 / 32;
    let (xp, qp, sp) = (
        P::new(x as *const f32),
        P::new(qs as *mut i8),
        P::new(sc as *mut f32),
    );
    (0..ncols).into_par_iter().for_each(|c| {
        for b in 0..nb {
            let base = c * ne0 + b * 32;
            let mut amax = 0f32;
            for l in 0..32 {
                let v = unsafe { *xp.at(base + l) }.abs();
                if v > amax {
                    amax = v;
                }
            }
            let d = amax / 127.0;
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            unsafe { *sp.at(c * nb + b) = d };
            for l in 0..32 {
                let v = unsafe { *xp.at(base + l) };
                unsafe { *qp.at(base + l) = rn(v * id) as i8 };
            }
        }
    });
    Ok(())
}

/// y[c][row] = sum_b d_w[b] * sc[c][b] * dot(q_w[b], q_x[c][b])  (+ bias[row])
pub fn q8_gemm(
    w: u64,
    qs: u64,
    sc: u64,
    y: u64,
    ne0: usize,
    ne1: usize,
    ncols: usize,
    bias: Option<u64>,
) -> Result<(), String> {
    let nb = ne0 / 32;
    let (xp, sp, yp) = (
        P::new(qs as *const i8),
        P::new(sc as *const f32),
        P::new(y as *mut f32),
    );
    let bp = bias.map(|b| P::new(b as *const f32));
    (0..ne1).into_par_iter().for_each(|row| {
        let wrow = unsafe { (w as *const u8).add(row_off(row, ne0)) };
        let mut acc = vec![0f32; ncols];
        for b in 0..nb {
            let blk = unsafe { wrow.add(b * BLK) };
            let d = blk_scale(blk);
            let q = unsafe { blk.add(QOFF) };
            let mut qw = [0i32; 32];
            for l in 0..32 {
                qw[l] = unsafe { *q.add(l) } as i8 as i32;
            }
            for (c, a) in acc.iter_mut().enumerate() {
                let xq = unsafe { xp.at(c * ne0 + b * 32) };
                let mut dot = 0i32;
                let mut l = 0;
                while l < 32 {
                    dot += qw[l] * unsafe { *xq.add(l) } as i32
                        + qw[l + 1] * unsafe { *xq.add(l + 1) } as i32
                        + qw[l + 2] * unsafe { *xq.add(l + 2) } as i32
                        + qw[l + 3] * unsafe { *xq.add(l + 3) } as i32;
                    l += 4;
                }
                *a += d * unsafe { *sp.at(c * nb + b) } * dot as f32;
            }
        }
        for (c, a) in acc.iter().enumerate() {
            let v = match bp {
                Some(p) => *a + unsafe { *p.at(row) },
                None => *a,
            };
            unsafe { *yp.at(c * ne1 + row) = v };
        }
    });
    Ok(())
}

/// y[c][row] = sum_k dequant(W_q8_0[row,k]) * x[c][k]  (f32 activations)
pub fn q8_gemm_f32(
    w: u64,
    x: u64,
    y: u64,
    ne0: usize,
    ne1: usize,
    ncols: usize,
) -> Result<(), String> {
    let (xp, yp) = (P::new(x as *const f32), P::new(y as *mut f32));
    (0..ne1).into_par_iter().for_each(|row| {
        let wrow = unsafe { (w as *const u8).add(row_off(row, ne0)) };
        let mut wf = vec![0f32; ne0];
        for b in 0..(ne0 / 32) {
            let blk = unsafe { wrow.add(b * BLK) };
            let d = blk_scale(blk);
            let q = unsafe { blk.add(QOFF) };
            for l in 0..32 {
                wf[b * 32 + l] = d * (unsafe { *q.add(l) } as i8 as f32);
            }
        }
        for c in 0..ncols {
            let xr = unsafe { xp.at(c * ne0) };
            let mut acc = 0f32;
            for k in 0..ne0 {
                acc += wf[k] * unsafe { *xr.add(k) };
            }
            unsafe { *yp.at(c * ne1 + row) = acc };
        }
    });
    Ok(())
}

/// y[c][row] = sum_k W_f32[row,k] * x[c][k]
pub fn f32_gemm(
    w: u64,
    x: u64,
    y: u64,
    ne0: usize,
    ne1: usize,
    ncols: usize,
) -> Result<(), String> {
    let (xp, yp) = (P::new(x as *const f32), P::new(y as *mut f32));
    (0..ne1).into_par_iter().for_each(|row| {
        let wr = unsafe { (w as *const f32).add(row * ne0) };
        for c in 0..ncols {
            let xr = unsafe { xp.at(c * ne0) };
            let mut acc = 0f32;
            for k in 0..ne0 {
                acc += unsafe { *wr.add(k) } * unsafe { *xr.add(k) };
            }
            unsafe { *yp.at(c * ne1 + row) = acc };
        }
    });
    Ok(())
}

// -------------------------------------------------------------- embeddings

pub fn get_row_q8(w: u64, row: usize, dst: u64, ne0: usize) -> Result<(), String> {
    // Keep the base as an integer: a raw pointer captured by the closure would
    // not be Send+Sync.
    let wr = w + row_off(row, ne0) as u64;
    let dp = P::new(dst as *mut f32);
    (0..(ne0 / 32)).into_par_iter().for_each(|b| {
        let blk = unsafe { (wr as *const u8).add(b * BLK) };
        let d = blk_scale(blk);
        let q = unsafe { blk.add(QOFF) };
        for l in 0..32 {
            unsafe { *dp.at(b * 32 + l) = d * (*q.add(l) as i8 as f32) };
        }
    });
    Ok(())
}

// ------------------------------------------------------------- elementwise

pub fn add_bias(x: u64, b: u64, ne0: usize, nrows: usize) -> Result<(), String> {
    let (xp, bp) = (P::new(x as *mut f32), P::new(b as *const f32));
    (0..nrows).into_par_iter().for_each(|r| {
        for i in 0..ne0 {
            unsafe { *xp.at(r * ne0 + i) += *bp.at(i) };
        }
    });
    Ok(())
}

pub fn add_inplace(a: u64, b: u64, n: usize) -> Result<(), String> {
    let (ap, bp) = (P::new(a as *mut f32), P::new(b as *const f32));
    chunks(n).into_par_iter().for_each(|(lo, hi)| {
        for i in lo..hi {
            unsafe { *ap.at(i) += *bp.at(i) };
        }
    });
    Ok(())
}

pub fn silu_mul(gate: u64, up: u64, n: usize) -> Result<(), String> {
    let (gp, up_) = (P::new(gate as *mut f32), P::new(up as *const f32));
    chunks(n).into_par_iter().for_each(|(lo, hi)| {
        for i in lo..hi {
            let g = unsafe { *gp.at(i) };
            unsafe { *gp.at(i) = (g / (1.0 + (-g).exp())) * *up_.at(i) };
        }
    });
    Ok(())
}

pub fn gelu_tanh(x: u64, n: usize) -> Result<(), String> {
    const C: f32 = 0.7978845608028654; // sqrt(2/pi)
    let xp = P::new(x as *mut f32);
    chunks(n).into_par_iter().for_each(|(lo, hi)| {
        for i in lo..hi {
            let v = unsafe { *xp.at(i) };
            unsafe { *xp.at(i) = 0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh()) };
        }
    });
    Ok(())
}

pub fn gelu_erf(x: u64, n: usize) -> Result<(), String> {
    let xp = P::new(x as *mut f32);
    chunks(n).into_par_iter().for_each(|(lo, hi)| {
        for i in lo..hi {
            let v = unsafe { *xp.at(i) };
            unsafe { *xp.at(i) = 0.5 * v * (1.0 + erf(v * 0.7071067811865476)) };
        }
    });
    Ok(())
}

/// erf via the Numerical Recipes erfc rational approximation
/// (|relative error| < 1.2e-7, which is below f32 resolution).
fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let erfc = t * (-z * z + poly).exp();
    if x >= 0.0 {
        1.0 - erfc
    } else {
        erfc - 1.0
    }
}

pub fn rms_norm(x: u64, w: u64, y: u64, ne0: usize, nrows: usize, eps: f32) -> Result<(), String> {
    let (xp, wp, yp) = (
        P::new(x as *const f32),
        P::new(w as *const f32),
        P::new(y as *mut f32),
    );
    (0..nrows).into_par_iter().for_each(|r| {
        let xr = unsafe { xp.at(r * ne0) };
        let yr = unsafe { yp.at(r * ne0) };
        let mut ss = 0f32;
        for i in 0..ne0 {
            let v = unsafe { *xr.add(i) };
            ss += v * v;
        }
        let scale = 1.0 / (ss / ne0 as f32 + eps).sqrt();
        for i in 0..ne0 {
            unsafe { *yr.add(i) = *xr.add(i) * scale * *wp.at(i) };
        }
    });
    Ok(())
}

pub fn layer_norm(
    x: u64,
    w: u64,
    b: u64,
    y: u64,
    ne0: usize,
    nrows: usize,
    eps: f32,
) -> Result<(), String> {
    let (xp, wp, bp, yp) = (
        P::new(x as *const f32),
        P::new(w as *const f32),
        P::new(b as *const f32),
        P::new(y as *mut f32),
    );
    (0..nrows).into_par_iter().for_each(|r| {
        let xr = unsafe { xp.at(r * ne0) };
        let yr = unsafe { yp.at(r * ne0) };
        let (mut s1, mut s2) = (0f32, 0f32);
        for i in 0..ne0 {
            let v = unsafe { *xr.add(i) };
            s1 += v;
            s2 += v * v;
        }
        let n = ne0 as f32;
        let mean = s1 / n;
        let var = s2 / n - mean * mean;
        let scale = 1.0 / (var.max(0.0) + eps).sqrt();
        for i in 0..ne0 {
            unsafe { *yr.add(i) = (*xr.add(i) - mean) * scale * *wp.at(i) + *bp.at(i) };
        }
    });
    Ok(())
}

pub fn argmax(x: u64, idx: u64, val: u64, ne0: usize, ncols: usize) -> Result<(), String> {
    let (xp, ip, vp) = (
        P::new(x as *const f32),
        P::new(idx as *mut i32),
        P::new(val as *mut f32),
    );
    (0..ncols).into_par_iter().for_each(|c| {
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0i32;
        for i in 0..ne0 {
            let v = unsafe { *xp.at(c * ne0 + i) };
            if v > best {
                best = v;
                bi = i as i32;
            }
        }
        unsafe { *ip.at(c) = bi };
        unsafe { *vp.at(c) = best };
    });
    Ok(())
}

/// dst[ntok][nrows] <- the first nrows entries of src[ntok][src_stride]
pub fn extract_rows(
    dst: u64,
    src: u64,
    src_stride: usize,
    nrows: usize,
    ntok: usize,
) -> Result<(), String> {
    let (dp, sp) = (P::new(dst as *mut f32), P::new(src as *const f32));
    (0..ntok).into_par_iter().for_each(|t| {
        for i in 0..nrows {
            unsafe { *dp.at(t * nrows + i) = *sp.at(t * src_stride + i) };
        }
    });
    Ok(())
}

/// 2x2 patch merge: out[m][4C] <- vf[gh*gw][C]
pub fn merge_2x2(out: u64, vf: u64, gh: usize, gw: usize, c: usize) -> Result<(), String> {
    let (op, vp) = (P::new(out as *mut f32), P::new(vf as *const f32));
    let mw = gw / 2;
    let m = (gh / 2) * mw;
    (0..m).into_par_iter().for_each(|mi| {
        let (a, cc) = (mi / mw, mi % mw);
        for e_idx in 0..4 {
            let (b, e) = (e_idx / 2, e_idx % 2);
            let tok = (2 * a + b) * gw + (2 * cc + e);
            for d in 0..c {
                unsafe { *op.at(mi * 4 * c + e_idx * c + d) = *vp.at(tok * c + d) };
            }
        }
    });
    Ok(())
}

// -------------------------------------------------------------------- rope

/// NEOX/rotate-half RoPE over [hd, nh, ntok] tensors (token stride nh*hd).
pub fn rope_neox(
    x: u64,
    pos: u64,
    hd: usize,
    nh: usize,
    ntok: usize,
    theta: f32,
) -> Result<(), String> {
    let xp = P::new(x as *mut f32);
    let pp = P::new(pos as *const i32);
    let half = hd / 2;
    (0..nh * ntok).into_par_iter().for_each(|job| {
        let (t, h) = (job / nh, job % nh);
        let p = unsafe { *pp.at(t) } as f32;
        let base = unsafe { xp.at((t * nh + h) * hd) };
        for i in 0..half {
            let ang = p * theta.powf(-2.0 * i as f32 / hd as f32);
            let (s, c) = ang.sin_cos();
            let x0 = unsafe { *base.add(i) };
            let x1 = unsafe { *base.add(i + half) };
            unsafe { *base.add(i) = x0 * c - x1 * s };
            unsafe { *base.add(i + half) = x0 * s + x1 * c };
        }
    });
    Ok(())
}

/// ViT 2-D RoPE with pair-major tables: cos/sin are [hd/2, ntok], pairs adjacent.
pub fn rope_2d(
    x: u64,
    cos: u64,
    sin: u64,
    hd: usize,
    nh: usize,
    ntok: usize,
) -> Result<(), String> {
    let (xp, cp, sp) = (
        P::new(x as *mut f32),
        P::new(cos as *const f32),
        P::new(sin as *const f32),
    );
    let np = hd / 2;
    (0..nh * ntok).into_par_iter().for_each(|job| {
        let (t, h) = (job / nh, job % nh);
        let base = unsafe { xp.at((t * nh + h) * hd) };
        for k in 0..np {
            let c = unsafe { *cp.at(k * ntok + t) };
            let s = unsafe { *sp.at(k * ntok + t) };
            let x0 = unsafe { *base.add(2 * k) };
            let x1 = unsafe { *base.add(2 * k + 1) };
            unsafe { *base.add(2 * k) = x0 * c - x1 * s };
            unsafe { *base.add(2 * k + 1) = x0 * s + x1 * c };
        }
    });
    Ok(())
}

// --------------------------------------------------------------- attention

/// out[tq][h][hd] = softmax_tk(q.k*scale + mask[tq][tk]) . v  (GQA aware).
/// q,k,v,out are [hd, nh, ntok] with token stride nh*hd; `mask` is an additive
/// [ntq][ntk] table or 0 for none.
#[allow(clippy::too_many_arguments)]
pub fn attn(
    q: u64,
    k: u64,
    v: u64,
    mask: u64,
    out: u64,
    hd: usize,
    n_qh: usize,
    n_kvh: usize,
    ntq: usize,
    ntk: usize,
) -> Result<(), String> {
    let (qp, kp, vp, mp, op) = (
        P::new(q as *const f32),
        P::new(k as *const f32),
        P::new(v as *const f32),
        P::new(mask as *const f32),
        P::new(out as *mut f32),
    );
    let scale = 1.0 / (hd as f32).sqrt();
    let group = n_qh / n_kvh;
    (0..n_qh * ntq).into_par_iter().for_each(|job| {
        let (tq, h) = (job / n_qh, job % n_qh);
        let hkv = h / group;
        let qrow = unsafe { qp.at((tq * n_qh + h) * hd) };
        let mut scores = vec![0f32; ntk];
        let mut m = f32::NEG_INFINITY;
        for (tk, sc) in scores.iter_mut().enumerate() {
            let krow = unsafe { kp.at((tk * n_kvh + hkv) * hd) };
            let mut dot = 0f32;
            for d in 0..hd {
                dot += unsafe { *qrow.add(d) } * unsafe { *krow.add(d) };
            }
            let mut s = dot * scale;
            if mask != 0 {
                s += unsafe { *mp.at(tq * ntk + tk) };
            }
            *sc = s;
            if s > m {
                m = s;
            }
        }
        let mut l = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            l += *s;
        }
        let orow = unsafe { op.at((tq * n_qh + h) * hd) };
        for d in 0..hd {
            unsafe { *orow.add(d) = 0.0 };
        }
        for tk in 0..ntk {
            let p = scores[tk];
            if p == 0.0 {
                continue;
            }
            let vrow = unsafe { vp.at((tk * n_kvh + hkv) * hd) };
            for d in 0..hd {
                unsafe { *orow.add(d) += p * *vrow.add(d) };
            }
        }
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for d in 0..hd {
            unsafe { *orow.add(d) *= inv };
        }
    });
    Ok(())
}

/// Split [0, n) into ~64k-element chunks for the flat elementwise loops.
fn chunks(n: usize) -> Vec<(usize, usize)> {
    const CH: usize = 1 << 14;
    (0..(n + CH - 1) / CH)
        .map(|i| (i * CH, ((i + 1) * CH).min(n)))
        .collect()
}
