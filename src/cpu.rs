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

fn cpu_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::is_x86_feature_detected!("avx2");
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}


#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn q8_row_panel_avx2(w:*const u8,x:*const i8,sc:*const f32,out:*mut f32,nb:usize,ncols:usize,stride:usize,row:usize,ne1:usize,bias:f32) {
 use std::arch::x86_64::*;
 let mut acc=vec![0f32;ncols];
 for b in 0..nb {
  let blk=w.add(b*BLK);let d=blk_scale(blk);
  let a0=_mm256_cvtepi8_epi16(_mm_loadu_si128(blk.add(2) as *const __m128i));
  let a1=_mm256_cvtepi8_epi16(_mm_loadu_si128(blk.add(18) as *const __m128i));
  for c0 in (0..ncols).step_by(4) {
   let mut vv=[_mm256_setzero_si256();4];
   for j in 0..4.min(ncols-c0) {
    let p=x.add((b*stride+c0+j)*32);
    let b0=_mm256_cvtepi8_epi16(_mm_loadu_si128(p as *const __m128i));
    let b1=_mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(16) as *const __m128i));
    vv[j]=_mm256_add_epi32(_mm256_madd_epi16(a0,b0),_mm256_madd_epi16(a1,b1));
   }
   for j in 0..4.min(ncols-c0) {
    let v=_mm_add_epi32(_mm256_castsi256_si128(vv[j]),_mm256_extracti128_si256(vv[j],1));
    let v=_mm_add_epi32(v,_mm_shuffle_epi32(v,0x4e));
    let dot=_mm_cvtsi128_si32(_mm_add_epi32(v,_mm_shuffle_epi32(v,0xb1)));
    acc[c0+j]+=d * *sc.add(b*stride+c0+j) * dot as f32;
   }
  }

 }
 for c in 0..ncols {*out.add(c*ne1+row)=acc[c]+bias;}
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn q8_two_rows_avx2(w:*const u8,x:*const i8,sc:*const f32,out:*mut f32,nb:usize,n:usize,stride:usize,r:usize,m:usize,bias0:f32,bias1:f32) {
 use std::arch::x86_64::*;
 let mut acc0=vec![0f32;n];let mut acc1=vec![0f32;n];
 for b in 0..nb {
  let w0=w.add(b*34);let w1=w.add((nb+b)*34);
  let a0=_mm256_cvtepi8_epi16(_mm_loadu_si128(w0.add(2) as *const __m128i));
  let a1=_mm256_cvtepi8_epi16(_mm_loadu_si128(w0.add(18) as *const __m128i));
  let b0=_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(2) as *const __m128i));
  let b1=_mm256_cvtepi8_epi16(_mm_loadu_si128(w1.add(18) as *const __m128i));
  let d0=blk_scale(w0);let d1=blk_scale(w1);
  for c in 0..n {
   let p=x.add((b*stride+c)*32);
   let xx0=_mm256_cvtepi8_epi16(_mm_loadu_si128(p as *const __m128i));
   let xx1=_mm256_cvtepi8_epi16(_mm_loadu_si128(p.add(16) as *const __m128i));
   let dots=[_mm256_add_epi32(_mm256_madd_epi16(a0,xx0),_mm256_madd_epi16(a1,xx1)),_mm256_add_epi32(_mm256_madd_epi16(b0,xx0),_mm256_madd_epi16(b1,xx1))];
   let mut dd=[0i32;2];
   for j in 0..2 {let v=_mm_add_epi32(_mm256_castsi256_si128(dots[j]),_mm256_extracti128_si256(dots[j],1));let v=_mm_add_epi32(v,_mm_shuffle_epi32(v,0x4e));dd[j]=_mm_cvtsi128_si32(_mm_add_epi32(v,_mm_shuffle_epi32(v,0xb1)));}
   acc0[c]+=d0 * *sc.add(b*stride+c) * dd[0] as f32;
   acc1[c]+=d1 * *sc.add(b*stride+c) * dd[1] as f32;
  }
 }
 for c in 0..n {*out.add(c*m+r)=acc0[c]+bias0;*out.add(c*m+r+1)=acc1[c]+bias1;}
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
    #[cfg(target_arch="x86_64")]
    if cpu_avx2() && ncols>=8 {
        let nb=ne0/32;const PANEL:usize=128;let panels=(ncols+PANEL-1)/PANEL;
        let mut packed=vec![0i8;ne0*PANEL*panels];let mut scales=vec![0f32;nb*PANEL*panels];
        packed.par_chunks_mut(ne0*PANEL).zip(scales.par_chunks_mut(nb*PANEL)).enumerate().for_each(|(p,(dst,ds))|{
            for b in 0..nb {for c in 0..PANEL.min(ncols-p*PANEL) { unsafe {
                std::ptr::copy_nonoverlapping((qs as *const i8).add((p*PANEL+c)*ne0+b*32),dst.as_mut_ptr().add((b*PANEL+c)*32),32);
                ds[b*PANEL+c]=*(sc as *const f32).add((p*PANEL+c)*nb+b);
            }}}
        });
        let xp=P::new(packed.as_ptr());let sp=P::new(scales.as_ptr());let yp=P::new(y as *mut f32);
        (0..((ne1+7)/8)*panels).into_par_iter().for_each(|job|unsafe {
            let p=job%panels;let r0=job/panels*8;
            let mut r=r0;let end=(r0+8).min(ne1);
            while r+1<end {q8_two_rows_avx2((w as *const u8).add(row_off(r,ne0)),xp.at(p*ne0*PANEL),sp.at(p*nb*PANEL),yp.at(p*PANEL*ne1),nb,PANEL.min(ncols-p*PANEL),PANEL,r,ne1,bias.map(|ptr|*(ptr as *const f32).add(r)).unwrap_or(0.0),bias.map(|ptr|*(ptr as *const f32).add(r+1)).unwrap_or(0.0));r+=2;}
            if r<end {q8_row_panel_avx2((w as *const u8).add(row_off(r,ne0)),xp.at(p*ne0*PANEL),sp.at(p*nb*PANEL),yp.at(p*PANEL*ne1),nb,PANEL.min(ncols-p*PANEL),PANEL,r,ne1,bias.map(|ptr|*(ptr as *const f32).add(r)).unwrap_or(0.0));}

        });
        return Ok(());
    }
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

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn f32_row_avx2(w:*const f32,x:*const f32,out:*mut f32,k:usize,stride:usize,n:usize,row:usize,m:usize) {
 use std::arch::x86_64::*;
 let mut tmp=[0f32;8];
 for c in (0..n).step_by(32) {
  let mut sums=[_mm256_setzero_ps();4];
  for d in 0..k {
   let ww=_mm256_set1_ps(*w.add(d));
   for j in 0..4 {sums[j]=_mm256_add_ps(sums[j],_mm256_mul_ps(ww,_mm256_loadu_ps(x.add(d*stride+c+j*8))));}
  }
  for j in 0..4 { _mm256_storeu_ps(tmp.as_mut_ptr(),sums[j]);
   for t in 0..8 {if c+j*8+t<n {*out.add((c+j*8+t)*m+row)=tmp[t];}}
  }
 }

}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn f32_two_rows_avx2(w:*const f32,x:*const f32,out:*mut f32,k:usize,n:usize,r:usize,m:usize) {
 use std::arch::x86_64::*;
 for c in (0..n).step_by(32) {
  let mut a=[_mm256_setzero_ps();4];let mut b=[_mm256_setzero_ps();4];
  for d in 0..k {
   let wa=_mm256_set1_ps(*w.add(d));let wb=_mm256_set1_ps(*w.add(k+d));
   for j in 0..4 {let xx=_mm256_loadu_ps(x.add(d*64+c+j*8));a[j]=_mm256_add_ps(a[j],_mm256_mul_ps(wa,xx));b[j]=_mm256_add_ps(b[j],_mm256_mul_ps(wb,xx));}
  }
  for j in 0..4 {let mut aa=[0f32;8];let mut bb=[0f32;8];_mm256_storeu_ps(aa.as_mut_ptr(),a[j]);_mm256_storeu_ps(bb.as_mut_ptr(),b[j]);
   for t in 0..8 {if c+j*8+t<n {*out.add((c+j*8+t)*m+r)=aa[t];*out.add((c+j*8+t)*m+r+1)=bb[t];}}
  }
 }
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
    #[cfg(target_arch="x86_64")]
    if cpu_avx2() && ncols>=8 {
        const PANEL:usize=64;
        let panels=(ncols+PANEL-1)/PANEL;
        let mut xt=vec![0f32;ne0*PANEL*panels];
        xt.par_chunks_mut(ne0*PANEL).enumerate().for_each(|(p,dst)| {
            for d in 0..ne0 {for j in 0..PANEL.min(ncols-p*PANEL) {
                dst[d*PANEL+j]=unsafe{*(x as *const f32).add((p*PANEL+j)*ne0+d)};
            }}
        });
        let xp=P::new(xt.as_ptr());let yp=P::new(y as *mut f32);
        (0..((ne1+7)/8)*panels).into_par_iter().for_each(|job|unsafe{
            let p=job%panels;let r0=job/panels*8;
            let mut r=r0;
            while r+1<(r0+8).min(ne1) {f32_two_rows_avx2((w as *const f32).add(r*ne0),xp.at(p*ne0*PANEL),yp.at(p*PANEL*ne1),ne0,PANEL.min(ncols-p*PANEL),r,ne1);r+=2;}
            if r<(r0+8).min(ne1) {f32_row_avx2((w as *const f32).add(r*ne0),xp.at(p*ne0*PANEL),yp.at(p*PANEL*ne1),ne0,PANEL,PANEL.min(ncols-p*PANEL),r,ne1);}
        });
        return Ok(());
    }
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
        let mean=ss/ne0 as f32;
        let scale = 1.0 / (mean + eps).sqrt();
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
        for i in 0..ne0 {let v=unsafe{*xr.add(i)};s1+=v;s2+=v*v;}
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


#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn pv_fixed_avx2<const NV:usize>(v:*const f32,p:*const f32,out:*mut f32,ntk:usize,vs:usize,inv:f32) {
 use std::arch::x86_64::*;
 let mut sums=[_mm256_setzero_ps();NV];
 for t in 0..ntk {let pp=*p.add(t);if pp!=0.0 {let pp=_mm256_set1_ps(pp);for j in 0..NV {sums[j]=_mm256_add_ps(sums[j],_mm256_mul_ps(pp,_mm256_loadu_ps(v.add(t*vs+j*8))));}}}
 for j in 0..NV {_mm256_storeu_ps(out.add(j*8),_mm256_mul_ps(sums[j],_mm256_set1_ps(inv)));}
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn pv_avx2(v:*const f32,p:*const f32,out:*mut f32,hd:usize,ntk:usize,vs:usize,inv:f32) {
 use std::arch::x86_64::*;
 if hd==72 {pv_fixed_avx2::<9>(v,p,out,ntk,vs,inv);return;}
 if hd==128 {pv_fixed_avx2::<16>(v,p,out,ntk,vs,inv);return;}
 let mut d=0;
 while d+8<=hd {
  let mut sum=_mm256_setzero_ps();
  for t in 0..ntk {let pp=*p.add(t);if pp!=0.0 {sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_set1_ps(pp),_mm256_loadu_ps(v.add(t*vs+d))));}}
  _mm256_storeu_ps(out.add(d),_mm256_mul_ps(sum,_mm256_set1_ps(inv)));d+=8;
 }
 for d in d..hd {let mut sum=0f32;for t in 0..ntk {let pp=*p.add(t);if pp!=0.0 {sum+=pp * *v.add(t*vs+d);}}*out.add(d)=sum*inv;}

}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn scores_panel_avx2(q:*const f32,kt:*const f32,scores:*mut f32,hd:usize,stride:usize,n:usize,scale:f32,mask:*const f32) {
 use std::arch::x86_64::*;
 let mut tmp=[0f32;8];
 for t in (0..n).step_by(32) {
  let mut sums=[_mm256_setzero_ps();4];
  for d in 0..hd {let qq=_mm256_set1_ps(*q.add(d));for j in 0..4 {sums[j]=_mm256_add_ps(sums[j],_mm256_mul_ps(qq,_mm256_loadu_ps(kt.add(d*stride+t+j*8))));}}
  for j in 0..4 {_mm256_storeu_ps(tmp.as_mut_ptr(),_mm256_mul_ps(sums[j],_mm256_set1_ps(scale)));
   for l in 0..8 {let idx=t+j*8+l;if idx<n {*scores.add(idx)=if mask.is_null(){tmp[l]}else{tmp[l]+*mask.add(idx)};}}
  }
 }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn scores_pair_avx2(q0:*const f32,q1:*const f32,kt:*const f32,s0:*mut f32,s1:*mut f32,hd:usize,n:usize,scale:f32,mask0:*const f32,mask1:*const f32) {
 use std::arch::x86_64::*;
 for c in (0..n).step_by(32) {
  let mut a=[_mm256_setzero_ps();4];let mut b=a;
  for d in 0..hd {let qa=_mm256_set1_ps(*q0.add(d));let qb=_mm256_set1_ps(*q1.add(d));
   for j in 0..4 {let xx=_mm256_loadu_ps(kt.add(d*64+c+j*8));a[j]=_mm256_add_ps(a[j],_mm256_mul_ps(qa,xx));b[j]=_mm256_add_ps(b[j],_mm256_mul_ps(qb,xx));}
  }
  for j in 0..4 {let mut aa=[0f32;8];let mut bb=[0f32;8];_mm256_storeu_ps(aa.as_mut_ptr(),_mm256_mul_ps(a[j],_mm256_set1_ps(scale)));_mm256_storeu_ps(bb.as_mut_ptr(),_mm256_mul_ps(b[j],_mm256_set1_ps(scale)));
   for t in 0..8 {let idx=c+j*8+t;if idx<n {*s0.add(idx)=if mask0.is_null(){aa[t]}else{aa[t]+*mask0.add(idx)};*s1.add(idx)=if mask1.is_null(){bb[t]}else{bb[t]+*mask1.add(idx)};}}
  }
 }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn scores_four_avx2(q:*const f32,qstride:usize,kt:*const f32,s:*mut f32,sstride:usize,hd:usize,n:usize,scale:f32,mask:*const f32,mstride:usize) {
 use std::arch::x86_64::*;
 for c in (0..n).step_by(16) {
  let mut acc=[[_mm256_setzero_ps();2];4];
  for d in 0..hd {let x0=_mm256_loadu_ps(kt.add(d*64+c));let x1=_mm256_loadu_ps(kt.add(d*64+c+8));
   for r in 0..4 {let qq=_mm256_set1_ps(*q.add(r*qstride+d));acc[r][0]=_mm256_add_ps(acc[r][0],_mm256_mul_ps(qq,x0));acc[r][1]=_mm256_add_ps(acc[r][1],_mm256_mul_ps(qq,x1));}
  }
  for r in 0..4 {for j in 0..2 {let mut tmp=[0f32;8];_mm256_storeu_ps(tmp.as_mut_ptr(),_mm256_mul_ps(acc[r][j],_mm256_set1_ps(scale)));for t in 0..8 {let idx=c+j*8+t;if idx<n {*s.add(r*sstride+idx)=if mask.is_null(){tmp[t]}else{tmp[t]+*mask.add(r*mstride+idx)};}}}}
 }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn scores_eight_avx2(q:*const f32,qstride:usize,kt:*const f32,s:*mut f32,sstride:usize,hd:usize,n:usize,scale:f32,mask:*const f32,mstride:usize) {
 use std::arch::x86_64::*;
 for c in (0..n).step_by(8) {
  let mut acc=[_mm256_setzero_ps();8];
  for d in 0..hd {let xx=_mm256_loadu_ps(kt.add(d*64+c));
   for r in 0..8 {acc[r]=_mm256_add_ps(acc[r],_mm256_mul_ps(_mm256_set1_ps(*q.add(r*qstride+d)),xx));}
  }
  for r in 0..8 {let mut tmp=[0f32;8];_mm256_storeu_ps(tmp.as_mut_ptr(),_mm256_mul_ps(acc[r],_mm256_set1_ps(scale)));for t in 0..8.min(n-c) {*s.add(r*sstride+c+t)=if mask.is_null(){tmp[t]}else{tmp[t]+*mask.add(r*mstride+c+t)};}}
 }
}

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
    let simd = cpu_avx2();
    let stride = 64;
    let panels=(ntk+63)/64;
    // Transpose only current keys, not weights; bounded temporary (~25 MiB on fixture).
    let mut kt = if simd {
        vec![0f32; n_kvh * hd * stride * panels]
    } else {
        Vec::new()
    };
    if simd {
        kt.par_chunks_mut(hd * stride * panels)
            .enumerate()
            .for_each(|(h, dst)| {
                for d in 0..hd {
                    for t in 0..ntk {
                        dst[(t/64)*hd*stride+d*stride+t%64] = unsafe { *kp.at((t * n_kvh + h) * hd + d) };
                    }
                }
            });
    }

    let packed_v_enabled=simd;
    let mut packed_v=if packed_v_enabled {vec![0f32;n_kvh*ntk*hd]}else{Vec::new()};
    if packed_v_enabled {packed_v.par_chunks_mut(ntk*hd).enumerate().for_each(|(h,dst)|unsafe{
        for t in 0..ntk {std::ptr::copy_nonoverlapping(vp.at((t*n_kvh+h)*hd),dst.as_mut_ptr().add(t*hd),hd);}
    });}
    let pv_enabled=simd;
    let paired=simd;
    let qt=if paired {8}else{1};let nq=(ntq+qt-1)/qt;
    (0..n_qh*nq).into_par_iter().for_each(|job| {
        let h=job/nq;let tq0=job%nq*qt;let count=qt.min(ntq-tq0);
        let mut pair_scores=vec![0f32;count*ntk];
        if paired && count==8 {
            #[cfg(target_arch="x86_64")]
            unsafe {for p in 0..panels {scores_eight_avx2(qp.at((tq0*n_qh+h)*hd),n_qh*hd,kt.as_ptr().add(((h/group)*panels+p)*hd*stride),pair_scores.as_mut_ptr().add(p*64),ntk,hd,64.min(ntk-p*64),scale,if mask==0 {std::ptr::null()}else{mp.at(tq0*ntk+p*64)},ntk);}}
        }
        if paired && count==4 {
            #[cfg(target_arch="x86_64")]
            unsafe {for p in 0..panels {scores_four_avx2(qp.at((tq0*n_qh+h)*hd),n_qh*hd,kt.as_ptr().add(((h/group)*panels+p)*hd*stride),pair_scores.as_mut_ptr().add(p*64),ntk,hd,64.min(ntk-p*64),scale,if mask==0 {std::ptr::null()}else{mp.at(tq0*ntk+p*64)},ntk);}}
        }
        if paired && count==2 {
            #[cfg(target_arch="x86_64")]
            unsafe {for p in 0..panels {scores_pair_avx2(qp.at((tq0*n_qh+h)*hd),qp.at(((tq0+1)*n_qh+h)*hd),kt.as_ptr().add(((h/group)*panels+p)*hd*stride),pair_scores.as_mut_ptr().add(p*64),pair_scores.as_mut_ptr().add(ntk+p*64),hd,64.min(ntk-p*64),scale,if mask==0 {std::ptr::null()}else{mp.at(tq0*ntk+p*64)},if mask==0 {std::ptr::null()}else{mp.at((tq0+1)*ntk+p*64)});}}
        }
        for sub in 0..count {let tq=tq0+sub;
        let hkv = h / group;
        let qrow = unsafe { qp.at((tq * n_qh + h) * hd) };
        let scores=&mut pair_scores[sub*ntk..(sub+1)*ntk];
        let mut m = f32::NEG_INFINITY;
        if paired && (count==2 || count==4 || count==8) {for &sc in scores.iter() {if sc>m {m=sc;}}}
        else if simd {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                for p in 0..panels {
                    scores_panel_avx2(qrow,kt.as_ptr().add((hkv*panels+p)*hd*stride),scores.as_mut_ptr().add(p*64),hd,stride,64.min(ntk-p*64),scale,
                        if mask==0 {std::ptr::null()} else {mp.at(tq*ntk+p*64)});
                }
            }
            for &sc in scores.iter() {
                if sc > m {
                    m = sc;
                }
            }
        } else {
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
        }
        let mut l = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            l += *s;
        }
        let orow = unsafe { op.at((tq * n_qh + h) * hd) };
        if pv_enabled {
            #[cfg(target_arch="x86_64")]
            unsafe {pv_avx2(if packed_v_enabled {packed_v.as_ptr().add(hkv*ntk*hd)}else{vp.at(hkv*hd)},scores.as_ptr(),orow,hd,ntk,if packed_v_enabled {hd}else{n_kvh*hd},if l>0.0 {1.0/l} else {0.0});}
        } else {
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
        }
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

#[cfg(test)]
mod simd_tests {
    use super::*;
    #[test]
    fn eight_scores_exact() {
        #[cfg(target_arch="x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            for hd in [7usize,72,128] {for n in [1usize,7,8,31,63,64] {
                let q:Vec<f32>=(0..hd*8).map(|i|((i*17%53) as f32-26.0)/19.0).collect();
                let kt:Vec<f32>=(0..hd*64).map(|i|((i*23%71) as f32-35.0)/23.0).collect();
                let mask:Vec<f32>=(0..n*8).map(|i|if i%3==0 {f32::NEG_INFINITY}else{0.0}).collect();
                for masked in [false,true] {
                    let mut a=vec![f32::NAN;n*8];
                    unsafe {scores_eight_avx2(q.as_ptr(),hd,kt.as_ptr(),a.as_mut_ptr(),n,hd,n,0.125,if masked {mask.as_ptr()}else{std::ptr::null()},n);}
                    for r in 0..8 {for t in 0..n {let mut sum=0f32;for d in 0..hd {sum+=q[r*hd+d]*kt[d*64+t];}let want=sum*0.125+if masked {mask[r*n+t]}else{0.0};assert_eq!(want.to_bits(),a[r*n+t].to_bits());}}
                }
            }}
        }
    }

    #[test]
    fn four_scores_exact() {
        #[cfg(target_arch="x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            for hd in [7usize,72,128] {for n in [1usize,7,8,15,16,31,63,64] {
                let q:Vec<f32>=(0..hd*4).map(|i|((i*17%53) as f32-26.0)/19.0).collect();
                let kt:Vec<f32>=(0..hd*64).map(|i|((i*23%71) as f32-35.0)/23.0).collect();
                let mask:Vec<f32>=(0..n*4).map(|i|if i%3==0 {f32::NEG_INFINITY}else{0.0}).collect();
                for masked in [false,true] {
                    let mut a=vec![f32::NAN;n*4];
                    unsafe {scores_four_avx2(q.as_ptr(),hd,kt.as_ptr(),a.as_mut_ptr(),n,hd,n,0.125,if masked {mask.as_ptr()}else{std::ptr::null()},n);}
                    for r in 0..4 {for t in 0..n {let mut sum=0f32;for d in 0..hd {sum+=q[r*hd+d]*kt[d*64+t];}let want=sum*0.125+if masked {mask[r*n+t]}else{0.0};assert_eq!(want.to_bits(),a[r*n+t].to_bits());}}
                }
            }}
        }
    }

    #[test]
    fn paired_scores_exact() {
        #[cfg(target_arch="x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            for hd in [7usize,72,128] {for n in [1usize,7,8,31,32,63,64] {
                let q:Vec<f32>=(0..hd*2).map(|i|((i*17%53) as f32-26.0)/19.0).collect();
                let kt:Vec<f32>=(0..hd*64).map(|i|((i*23%71) as f32-35.0)/23.0).collect();
                let mask:Vec<f32>=(0..n*2).map(|i|if i%3==0 {f32::NEG_INFINITY}else{0.0}).collect();
                for masked in [false,true] {
                    let mut a=vec![f32::NAN;n];let mut b=a.clone();
                    unsafe {scores_pair_avx2(q.as_ptr(),q.as_ptr().add(hd),kt.as_ptr(),a.as_mut_ptr(),b.as_mut_ptr(),hd,n,0.125,if masked {mask.as_ptr()}else{std::ptr::null()},if masked {mask.as_ptr().add(n)}else{std::ptr::null()});}
                    for t in 0..n {for r in 0..2 {let mut sum=0f32;for d in 0..hd {sum+=q[r*hd+d]*kt[d*64+t];}let want=sum*0.125+if masked {mask[r*n+t]}else{0.0};assert_eq!(want.to_bits(),if r==0 {a[t]}else{b[t]}.to_bits());}}
                }
            }}
        }
    }

    #[test]
    fn two_row_tiles_exact() {
        #[cfg(target_arch="x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            for n in [1,7,8,31,32,63,64] {
                let k=96;let m=2;
                let x:Vec<f32>=(0..k*64).map(|i|((i*19%71) as f32-35.0)/23.0).collect();
                let w:Vec<f32>=(0..k*m).map(|i|((i*13%43) as f32-21.0)/19.0).collect();
                let mut got=vec![0f32;n*m];let mut want=vec![0f32;n*m];
                unsafe {f32_two_rows_avx2(w.as_ptr(),x.as_ptr(),got.as_mut_ptr(),k,n,0,m);
                    for r in 0..m {f32_row_avx2(w.as_ptr().add(r*k),x.as_ptr(),want.as_mut_ptr(),k,64,n,r,m);}}
                assert_eq!(got,want);
                let nb=3;let mut qw=vec![0u8;nb*34*m];
                for b in 0..nb*m {qw[b*34+1]=0x3c;for j in 0..32 {qw[b*34+2+j]=((b*37+j*17)%256) as u8;}}
                let qx:Vec<i8>=(0..nb*64*32).map(|i|((i*31)%256) as i8).collect();
                let scales:Vec<f32>=(0..nb*64).map(|i|(i+1) as f32/79.0).collect();
                unsafe {q8_two_rows_avx2(qw.as_ptr(),qx.as_ptr(),scales.as_ptr(),got.as_mut_ptr(),nb,n,64,0,m,0.25,-0.125);
                    for r in 0..m {q8_row_panel_avx2(qw.as_ptr().add(r*nb*34),qx.as_ptr(),scales.as_ptr(),want.as_mut_ptr(),nb,n,64,r,m,if r==0 {0.25}else{-0.125});}}
                assert_eq!(got,want);
            }
        }
    }

    #[test]
    fn packed_kernels_match_scalar() {
        #[cfg(target_arch="x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let k = 96; let n = 31; let m = 3;
            let x: Vec<f32> = (0..k * 64).map(|i| ((i * 17 % 53) as f32 - 26.0) / 19.0).collect();
            let w: Vec<f32> = (0..k * m).map(|i| ((i * 13 % 31) as f32 - 15.0) / 17.0).collect();
            let mut y = vec![f32::NAN; n * m];
            unsafe { for r in 0..m { f32_row_avx2(w.as_ptr().add(r * k), x.as_ptr(), y.as_mut_ptr(), k, 64, n, r, m); } }
            for c in 0..n { for r in 0..m {
                let mut sum = 0f32;
                for d in 0..k { sum += w[r * k + d] * x[d * 64 + c]; }
                assert_eq!(sum.to_bits(), y[c * m + r].to_bits());
            }}
        }
    }
}
