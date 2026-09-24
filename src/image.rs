//! Image preprocessing for the ViT path, mirroring the reference
//! src/image_io.cpp + pil_resize.hpp + vit_posemb.cpp.
//!
//! Pipeline: load RGB (png) -> optional pre-downscale -> PIL-equivalent bicubic
//! resize to a multiple of 28 -> normalize v = raw/127.5 - 1 -> patchify to
//! [588, n_tok] with pv[t*588 + c*196 + i*14 + j].

pub const PATCH: usize = 14;
pub const MERGE_W: usize = 2;
pub const IN_TOKEN_LIMIT: usize = 25600;

/// Cubic kernel, PIL's BICUBIC (a = -0.5) unless `a` is given.
fn cubic_w(t: f32, a: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
    } else if t < 2.0 {
        (((t - 5.0) * t + 8.0) * t - 4.0) * a
    } else {
        0.0
    }
}

/// PIL-equivalent bicubic resize of an RGB8 HWC image.
/// a = -0.5; on downscale the support is scaled (antialias), matching PIL.
pub fn pil_bicubic_resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    assert_eq!(src.len(), sw * sh * 3);
    let mut out = vec![0u8; dw * dh * 3];
    // scale factors: PIL uses filt_scale = max(1, src/dst) for the support
    let scale_w = sw as f32 / dw as f32;
    let scale_h = sh as f32 / dh as f32;
    let (filt_w, filt_h) = (scale_w.max(1.0), scale_h.max(1.0));
    // PIL's coefficients: for each output coordinate, center = (o + 0.5) * scale
    let mut cw: Vec<Vec<(usize, f32)>> = Vec::with_capacity(dw);
    for ox in 0..dw {
        let center = (ox as f32 + 0.5) * scale_w;
        let (start, len) = support(center, filt_w);
        let mut total = 0f32;
        let mut norm: Vec<(usize, f32)> = Vec::with_capacity(len);
        for i in start..(start + len as i32) {
            let w = cubic_w((i as f32 + 0.5 - center) / filt_w, -0.5);
            total += w;
            norm.push((i as usize, w));
        }
        for e in norm.iter_mut() {
            e.1 /= total;
        }
        cw.push(norm);
    }
    let mut ch: Vec<Vec<(usize, f32)>> = Vec::with_capacity(dh);
    for oy in 0..dh {
        let center = (oy as f32 + 0.5) * scale_h;
        let (start, len) = support(center, filt_h);
        let mut total = 0f32;
        let mut taps: Vec<(usize, f32)> = Vec::new();
        for i in start..(start + len as i32) {
            let w = cubic_w((i as f32 + 0.5 - center) / filt_h, -0.5);
            taps.push((i as usize, w));
            total += w;
        }
        ch.push(taps.into_iter().map(|(i, w)| (i, w / total)).collect());
    }

    // separable: horizontal pass into a temp buffer, then vertical
    let mut tmp = vec![0f32; dw * sh * 3];
    for y in 0..sh {
        for ox in 0..dw {
            for c in 0..3 {
                let mut acc = 0f32;
                for &(ix, w) in &cw[ox] {
                    let ix = ix.min(sw - 1);
                    acc += w * src[(y * sw + ix) * 3 + c] as f32;
                }
                tmp[(y * dw + ox) * 3 + c] = acc;
            }
        }
    }
    for oy in 0..dh {
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0f32;
                for &(iy, w) in &ch[oy] {
                    let iy = iy.min(sh - 1);
                    acc += w * tmp[(iy * dw + x) * 3 + c];
                }
                let v = acc.round().clamp(0.0, 255.0) as u8;
                out[(oy * dw + x) * 3 + c] = v;
            }
        }
    }
    out
}

/// Integer tap range and count for a bicubic filter centered at `center`
/// with support radius 2 * filt.
fn support(center: f32, filt: f32) -> (i32, usize) {
    let radius = 2.0 * filt;
    let start = (center - radius + 0.5).ceil() as i32;
    let end = (center + radius + 0.5).floor() as i32;
    (start, ((end - start).max(1)) as usize)
}

pub struct Preprocessed {
    pub gh: usize,
    pub gw: usize,
    pub target_w: usize,
    pub target_h: usize,
    pub pixel_values: Vec<f32>, // [588, n_tok] laid out pv[t*588 + ...]
}

/// (A) downscale when the patch grid would exceed the token limit.
/// (B) round each dimension up to a multiple of merge*patch (28).
pub fn target_size(w0: usize, h0: usize) -> (usize, usize) {
    let (mut w, mut h) = (w0, h0);
    if (w / PATCH) * (h / PATCH) > IN_TOKEN_LIMIT {
        let scale = (IN_TOKEN_LIMIT as f64 / ((w / PATCH) as f64 * (h / PATCH) as f64)).sqrt();
        w = (w0 as f64 * scale) as usize;
        h = (h0 as f64 * scale) as usize;
    }
    let pad = MERGE_W * PATCH; // 28
    let tw = ((w + pad - 1) / pad) * pad;
    let th = ((h + pad - 1) / pad) * pad;
    (tw, th)
}

pub fn preprocess(rgb: &[u8], w0: usize, h0: usize) -> Result<Preprocessed, String> {
    // stage A
    let mut w = w0;
    let mut h = h0;
    let mut cur = rgb.to_vec();
    if (w / PATCH) * (h / PATCH) > IN_TOKEN_LIMIT {
        let scale = (IN_TOKEN_LIMIT as f64 / ((w / PATCH) as f64 * (h / PATCH) as f64)).sqrt();
        let w1 = (w0 as f64 * scale) as usize;
        let h1 = (h0 as f64 * scale) as usize;
        cur = pil_bicubic_resize(&cur, w, h, w1, h1);
        w = w1;
        h = h1;
    }
    let (tw, th) = target_size(w0, h0);
    if tw != w || th != h {
        cur = pil_bicubic_resize(&cur, w, h, tw, th);
    }
    if tw / PATCH >= 512 || th / PATCH >= 512 {
        return Err(format!("patch grid too large: {tw}x{th}"));
    }
    let gh = th / PATCH;
    let gw = tw / PATCH;
    let n = gh * gw;
    let plane = th * tw;
    // normalize to CHW
    let mut chw = vec![0f32; 3 * plane];
    for y in 0..th {
        for x in 0..tw {
            let p = &cur[(y * tw + x) * 3..(y * tw + x) * 3 + 3];
            for c in 0..3 {
                chw[c * plane + y * tw + x] = p[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    // patchify
    let mut pv = vec![0f32; n * 588];
    for row in 0..gh {
        for col in 0..gw {
            let t = row * gw + col;
            for c in 0..3 {
                for i in 0..PATCH {
                    let y = row * PATCH + i;
                    for j in 0..PATCH {
                        let x = col * PATCH + j;
                        pv[t * 588 + c * 196 + i * 14 + j] = chw[c * plane + y * tw + x];
                    }
                }
            }
        }
    }
    Ok(Preprocessed {
        gh,
        gw,
        target_w: tw,
        target_h: th,
        pixel_values: pv,
    })
}

/// Bicubic resize of the learned 64x64 positional embedding to (gh, gw),
/// a = -0.75, half-pixel centers, clamped 4-tap (vit_posemb.cpp).
pub fn bicubic_pos_emb(
    src: &[f32],
    base_h: usize,
    base_w: usize,
    c: usize,
    gh: usize,
    gw: usize,
) -> Vec<f32> {
    let a = -0.75f32;
    let sh = base_h as f32 / gh as f32;
    let sw = base_w as f32 / gw as f32;
    let mut out = vec![0f32; gh * gw * c];
    let at = |y: i32, x: i32, ch: usize| -> f32 {
        let y = y.clamp(0, base_h as i32 - 1) as usize;
        let x = x.clamp(0, base_w as i32 - 1) as usize;
        src[(y * base_w + x) * c + ch]
    };
    for oy in 0..gh {
        let fy = (oy as f32 + 0.5) * sh - 0.5;
        let iy = fy.floor();
        let ty = fy - iy;
        let wy = [
            cubic_w(1.0 + ty, a),
            cubic_w(ty, a),
            cubic_w(1.0 - ty, a),
            cubic_w(2.0 - ty, a),
        ];
        for ox in 0..gw {
            let fx = (ox as f32 + 0.5) * sw - 0.5;
            let ix = fx.floor();
            let tx = fx - ix;
            let wx = [
                cubic_w(1.0 + tx, a),
                cubic_w(tx, a),
                cubic_w(1.0 - tx, a),
                cubic_w(2.0 - tx, a),
            ];
            for ch in 0..c {
                let mut acc = 0f32;
                for m in 0..4 {
                    let mut row = 0f32;
                    for n in 0..4 {
                        row += wx[n] * at(iy as i32 - 1 + m as i32, ix as i32 - 1 + n as i32, ch);
                    }
                    acc += wy[m] * row;
                }
                out[(oy * gw + ox) * c + ch] = acc;
            }
        }
    }
    out
}

/// 2-D RoPE tables for the ViT, laid out pair-major for lg_rope_2d:
///   cos[k*ntok + t], where pair k of token t = h*gw + w uses
///   invf = theta^(-4*(k/2)/head_dim) and coord = w for even k, h for odd k.
pub fn rope_2d_tables(gh: usize, gw: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let npairs = head_dim / 2;
    let ntok = gh * gw;
    let mut cos = vec![0f32; npairs * ntok];
    let mut sin = vec![0f32; npairs * ntok];
    for h in 0..gh {
        for w in 0..gw {
            let tok = h * gw + w;
            for k in 0..npairs {
                let i = (k / 2) as f32;
                let invf = theta.powf(-4.0 * i / head_dim as f32);
                let coord = if k % 2 == 0 { w as f32 } else { h as f32 };
                let ang = coord * invf;
                cos[k * ntok + tok] = ang.cos();
                sin[k * ntok + tok] = ang.sin();
            }
        }
    }
    (cos, sin)
}
