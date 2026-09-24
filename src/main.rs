//! locate-anything: standalone LocateAnything-3B (MoonViT + Qwen2) inference
//! engine reading the `.laqt` container produced by tools/prequant.py.
//!
//! Subcommands:
//!   detect [--model c.laqt] <image.png> <query>   locate objects described by <query>;
//!                                                 boxes as JSON on stdout
//!   inspect [--model c.laqt]                      print the container directory
//!   gpuinfo                                       initialise CUDA, report the device
//!
//! The model defaults to `locate-anything-allq8_0.laqt`, searched next to the
//! executable, then in a `models/` subdirectory, then up the tree; --model
//! overrides it.

mod container;
mod cpu;
mod cuda;
mod graph;
mod image;
mod model;
mod tokenizer;

use container::Container;
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() {
    eprintln!(
        "locate-anything {}\n\
         \n\
         usage:\n\
         \x20 locate-anything detect [--model c.laqt] <image.png> <query>\n\
         \x20 locate-anything inspect [--model c.laqt]\n\
         \x20 locate-anything gpuinfo\n\
         \n\
         --model defaults to locate-anything-allq8_0.laqt, searched next to the\n\
         \x20 executable, then in a models/ subdirectory, then up the tree, so a\n\
         \x20 download that keeps the two together just works.\n\
         \n\
         detect writes the boxes to stdout as\n\
         \x20 {{\"detections\":[{{\"label\":\"cat\",\"box\":[3.14,50.62,221.76,443.52]}}, ...]}}\n\
         and every diagnostic to stderr, so stdout can be piped straight into a\n\
         parser. inspect and gpuinfo report on stdout.\n\
         \n\
         env:\n\
         \x20 LA_CPU=1      force the CPU backend (the automatic fallback when\n\
         \x20               the CUDA driver cannot be initialised)\n\
         \x20 LA_MAX_NEW    decode token budget (default 1024)",
        env!("CARGO_PKG_VERSION")
    );
}

const DEFAULT_MODEL: &str = "locate-anything-allq8_0.laqt";

/// Where the model container lives when `--model` is not given.
///
/// A user who drops the model next to the binary gets the simplest invocation,
/// so the search starts next to the executable. Then a `models/` subdirectory
/// (the source-checkout layout), then up the tree from the executable, and
/// finally the current directory.
fn default_model() -> PathBuf {
    const SUBDIR: &str = "models";
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [
                dir.join(DEFAULT_MODEL),
                dir.join(SUBDIR).join(DEFAULT_MODEL),
            ] {
                if cand.exists() {
                    return cand;
                }
            }
            let mut base = dir.to_path_buf();
            for _ in 0..6 {
                for cand in [
                    base.join(DEFAULT_MODEL),
                    base.join(SUBDIR).join(DEFAULT_MODEL),
                ] {
                    if cand.exists() {
                        return cand;
                    }
                }
                if !base.pop() {
                    break;
                }
            }
        }
    }
    PathBuf::from(DEFAULT_MODEL)
}

/// Split a `--model <path>` flag off the argument list; the rest is positional.
/// Only resolves a model when none is given explicitly.
fn parse_model(args: &[String]) -> Result<(PathBuf, Vec<String>), String> {
    let mut model = None;
    let mut pos = Vec::new();
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == "--model" {
            model = Some(PathBuf::from(
                it.next()
                    .ok_or_else(|| "--model needs a value".to_string())?,
            ));
        } else {
            pos.push(a.clone());
        }
    }
    match model {
        Some(m) => Ok((m, pos)),
        None => {
            // No --model and no positional path: use the default search. Report
            // it on stderr so a default find is not a surprise.
            let m = default_model();
            if m.exists() {
                eprintln!("model     : {}", m.display());
            } else {
                eprintln!(
                    "model     : {DEFAULT_MODEL} not found (searched next to the binary, \
                     models/, up the tree); see the README's 'Get the model' or pass --model"
                );
            }
            Ok((m, pos))
        }
    }
}

fn cmd_inspect(path: &PathBuf) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let c = Container::open(path)?;
    let parse_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut q8 = 0usize;
    let mut f32n = 0usize;
    let mut q8_bytes = 0usize;
    let mut f32_bytes = 0usize;
    let mut aliases = Vec::new();
    for name in &c.order {
        let t = &c.tensors[name];
        if t.alias_of.is_some() {
            aliases.push(name.clone());
            continue;
        }
        match t.dtype {
            container::Dtype::Q8_0 => {
                q8 += 1;
                q8_bytes += t.nbytes;
            }
            container::Dtype::F32 => {
                f32n += 1;
                f32_bytes += t.nbytes;
            }
        }
    }
    println!("container : {}", path.display());
    println!("version   : {}", c.version);
    println!(
        "tensors   : {} entries ({} payloads, {} aliases)",
        c.order.len(),
        q8 + f32n,
        aliases.len()
    );
    println!("  q8_0    : {q8} tensors, {:.3} GB", q8_bytes as f64 / 1e9);
    println!(
        "  f32     : {f32n} tensors, {:.3} GB",
        f32_bytes as f64 / 1e9
    );
    println!(
        "  payload : {:.3} GB (aliases: {})",
        c.payload_bytes() as f64 / 1e9,
        aliases.join(", ")
    );
    println!(
        "tokenizer : {} tokens, {} merges",
        c.tokens.len(),
        c.merges.len()
    );
    println!("config    : {} keys", c.config.len());
    for (k, v) in &c.config {
        println!("    {k:24} {}", compact(v));
    }
    println!("parse     : {parse_ms:.1} ms");

    // Spot-check a few payloads by reading the first bytes through the mapping.
    for name in [
        "lm.tok_embd.weight",
        "vit.blk.0.fc0.weight",
        "vit.blk.0.fc1.weight",
        "proj.1.weight",
    ] {
        if let Some(t) = c.info(name) {
            let raw = c.raw(name)?;
            let head: Vec<String> = raw.iter().take(8).map(|b| format!("{b:02x}")).collect();
            println!(
                "  {name:22} {:?} {:?} {:>12} B  head {}",
                t.dtype,
                t.shape,
                t.nbytes,
                head.join(" ")
            );
        }
    }
    Ok(())
}

/// Escape a label for a JSON string literal (same rule as the reference CLI's
/// json_escape): quotes, backslashes and control characters. Labels are decoded
/// tokens, so they can hold arbitrary bytes.
fn json_escape(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' | '\\' => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
}

fn compact(v: &container::Json) -> String {
    use container::Json;
    match v {
        Json::Null => "null".into(),
        Json::Bool(b) => b.to_string(),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Json::Str(s) => format!("\"{s}\""),
        Json::Array(a) => format!("[{} items]", a.len()),
        Json::Object(o) => format!("{{{} keys}}", o.len()),
    }
}

/// Decode a PNG to RGB8 (no alpha, no palette handling beyond the png crate's
/// own expansion).
fn decode_png(path: &str) -> Result<(Vec<u8>, usize, usize), String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let (w, h) = (info.width as usize, info.height as usize);
    let bytes = &buf[..info.buffer_size()];
    let rgb = match info.color_type {
        png::ColorType::Rgb => bytes.to_vec(),
        png::ColorType::Rgba => {
            let mut v = Vec::with_capacity(w * h * 3);
            for p in bytes.chunks_exact(4) {
                v.extend_from_slice(&p[..3]);
            }
            v
        }
        png::ColorType::Grayscale => {
            let mut v = Vec::with_capacity(w * h * 3);
            for &g in bytes {
                v.extend_from_slice(&[g, g, g]);
            }
            v
        }
        other => return Err(format!("unsupported PNG color type {other:?}")),
    };
    Ok((rgb, w, h))
}

// ---------------------------------------------------------------------------
// detect <container> <image.png> <query>
// Full pipeline: ViT -> connector -> prompt -> splice -> greedy LM decode ->
// box parsing (mirrors the reference engine's --mode slow AR path).
// ---------------------------------------------------------------------------
fn cmd_detect(path: &PathBuf, image: &str, query: &str) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let c = Container::open(path)?;
    let num = |key: &str, d: f64| -> f64 {
        match c.config.iter().find(|(n, _)| n == key) {
            Some((_, container::Json::Num(n))) => *n,
            _ => d,
        }
    };
    let tokid = |key: &str, d: i32| -> i32 {
        match c.config.iter().find(|(n, _)| n == key) {
            Some((_, container::Json::Num(n))) => *n as i32,
            _ => d,
        }
    };
    let (rgb, w0, h0) = decode_png(image)?;
    let pre = image::preprocess(&rgb, w0, h0)?;
    eprintln!(
        "image     : {} ({}x{}) -> target {}x{} grid {}x{} ({} patches, {} merged)",
        image,
        w0,
        h0,
        pre.target_w,
        pre.target_h,
        pre.gh,
        pre.gw,
        pre.gh * pre.gw,
        pre.gh * pre.gw / 4
    );

    let v = graph::Vit {
        n_layers: num("vit.n_layers", 27.0) as usize,
        hidden: num("vit.hidden_size", 1152.0) as usize,
        head_dim: num("vit.head_dim", 72.0) as usize,
        n_heads: num("vit.n_heads", 16.0) as usize,
        inter: num("vit.intermediate_size", 4304.0) as usize,
        eps: num("vit.layer_norm_eps", 1e-5) as f32,
        gh: pre.gh,
        gw: pre.gw,
        rope_theta: num("vit.rope_theta", 1e4) as f32,
    };
    let lm = graph::Lm {
        n_layers: num("lm.n_layers", 36.0) as usize,
        hidden: num("lm.hidden_size", 2048.0) as usize,
        head_dim: num("lm.head_dim", 128.0) as usize,
        n_heads: num("lm.n_heads", 16.0) as usize,
        n_kv_heads: num("lm.n_kv_heads", 2.0) as usize,
        inter: num("lm.intermediate_size", 11008.0) as usize,
        vocab: num("lm.vocab_size", 152681.0) as usize,
        eps: num("lm.rms_norm_eps", 1e-6) as f32,
        rope_theta: num("lm.rope_theta", 1e6) as f32,
    };
    let img_tok = tokid("token.image", 151665);
    let coord_start = tokid("token.coord_start", 151677);
    let coord_end = tokid("token.coord_end", 152677);
    let box_start = tokid("token.box_start", 151668);
    let box_end = tokid("token.box_end", 151669);
    let ref_start = tokid("token.ref_start", 151672);
    let ref_end = tokid("token.ref_end", 151673);
    let eos = tokid("token.eos", 151645);

    // CUDA must be initialised before any allocation.
    let k = graph::K::new()?;
    let w = model::DeviceWeights::load(&c)?;

    // ---- vision path ----
    let t_vit = std::time::Instant::now();
    let merged = v.forward(&k, &w, &pre.pixel_values)?;
    let m = v.nmerged();
    let feats = graph::connector(&k, &w, &merged, m, 4 * v.hidden, 2048, 1e-5)?;
    k.sync()?;
    eprintln!(
        "vit       : {:.1} ms, features [2048, {}]",
        t_vit.elapsed().as_secs_f64() * 1e3,
        m
    );

    // ---- tokenizer + prompt ----
    let tk = tokenizer::Tokenizer::load(&c.tokens, &c.token_types, &c.merges)
        .ok_or_else(|| "tokenizer load failed".to_string())?;
    let ids = build_prompt(&tk, pre.gh, pre.gw, query);
    eprintln!("prompt    : {} tokens", ids.len());

    // ---- host splice: gather tok_embd rows, overwrite IMG_CONTEXT slots ----
    let te = w.get("lm.tok_embd.weight")?;
    let mut raw = vec![0u8; te.nbytes];
    {
        k.read_bytes(raw.as_mut_ptr(), te.addr, te.nbytes)?;
    }
    let h = lm.hidden;
    // Host row dequantization of the q8_0 embedding table. On the GPU the
    // payload is in the ALIGNED layout (36-byte blocks: f16 scale, 2 pad bytes,
    // payload at offset 4); on the CPU it is the container's native ggml layout
    // (34-byte blocks, payload at offset 2). Using the wrong stride here
    // silently reads garbage.
    let (blk, qoff) = if cuda::cpu_mode() {
        (model::Q8_0_BLOCK_IN, 2usize)
    } else {
        (model::Q8_0_BLOCK_DEV, 4usize)
    };
    let deq_row = |id: usize| -> Vec<f32> {
        let blocks = h / 32;
        let mut row = vec![0f32; h];
        for b in 0..blocks {
            let off = (id * blocks + b) * blk;
            let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
            for j in 0..32 {
                let q = raw[off + qoff + j] as i8;
                row[b * 32 + j] = d * q as f32;
            }
        }
        row
    };
    let seq = ids.len();
    let mut spliced = vec![0f32; h * seq];
    for (t, &id) in ids.iter().enumerate() {
        let row = deq_row(id as usize);
        spliced[t * h..(t + 1) * h].copy_from_slice(&row);
    }
    let mut vi = 0usize;
    for (t, &id) in ids.iter().enumerate() {
        if id == img_tok {
            let src = &feats[vi * h..(vi + 1) * h];
            spliced[t * h..(t + 1) * h].copy_from_slice(src);
            vi += 1;
        }
    }
    eprintln!("splice    : {} image rows overwritten (m = {})", vi, m);
    if vi != m {
        return Err(format!("image token count {vi} != merged tokens {m}"));
    }

    // ---- greedy decode: prefill the whole prompt, then one step per token ----
    let logits = cuda::Buffer::alloc(lm.vocab * 4)?;
    let mut gen: Vec<i32> = Vec::new();
    let max_new = std::env::var("LA_MAX_NEW")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1024);
    let argmax_of = |out: &[f32]| -> usize {
        let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
        for (i, val) in out.iter().enumerate() {
            if *val > bv {
                bv = *val;
                bi = i;
            }
        }
        bi
    };
    let pull_logits = |logits: &cuda::Buffer| -> Result<Vec<f32>, String> {
        let mut out = vec![0f32; lm.vocab];
        logits.download(unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, lm.vocab * 4)
        })?;
        Ok(out)
    };
    let dt = std::time::Instant::now();
    // Prefill the whole prompt into the KV cache, then one kernel-light step per token.
    let mut kv = graph::KvCache::new(&lm, seq + max_new + 8)?;
    let dctx = graph::DecodeCtx::new(&lm)?;
    lm.prefill_into_cache(&k, &w, &spliced, seq, &mut kv, &logits)?;
    k.sync()?;
    eprintln!(
        "prefill   : {:.1} ms (filled KV cache to {})",
        dt.elapsed().as_secs_f64() * 1e3,
        kv.len
    );
    let mut idx_host = [0i32; 1];
    for step in 0..max_new {
        // The prefill wrote the full logits; later steps argmax on device.
        let bi = if step == 0 {
            let out = pull_logits(&logits)?;
            argmax_of(&out)
        } else {
            dctx.idx.download(unsafe {
                std::slice::from_raw_parts_mut(idx_host.as_mut_ptr() as *mut u8, 4)
            })?;
            idx_host[0] as usize
        };
        gen.push(bi as i32);
        if bi as i32 == eos {
            eprintln!("decode    : EOS at step {step}");
            break;
        }
        if step + 1 == max_new {
            break;
        }
        let t_row = std::time::Instant::now();
        let row = deq_row(bi);
        let d_row = t_row.elapsed().as_secs_f64() * 1e3;
        let pos = kv.len;
        let t_step = std::time::Instant::now();
        lm.decode_step(&k, &w, &row, bi, pos, &mut kv, &None, &dctx)?;
        k.sync()?;
        let d_step = t_step.elapsed().as_secs_f64() * 1e3;
        if step < 8 {
            eprintln!(
                "  step {step:4}: row {:.2} ms, decode+sync {:.2} ms, total {:.2} ms",
                d_row,
                d_step,
                d_row + d_step
            );
        }
        if step % 16 == 0 {
            eprintln!(
                "  step {step:4} token {bi} ({:.1} s, kv={})",
                dt.elapsed().as_secs_f64(),
                kv.len
            );
        }
    }
    eprintln!(
        "decode    : {} tokens in {:.1} s",
        gen.len(),
        dt.elapsed().as_secs_f64()
    );

    // ---- parse reference-style boxes ----
    let mut out: Vec<(String, f32, f32, f32, f32)> = Vec::new();
    let mut label = String::new();
    let mut i = 0usize;
    while i < gen.len() {
        let t = gen[i];
        if t == ref_start {
            let mut j = i + 1;
            let mut lab: Vec<i32> = Vec::new();
            while j < gen.len() && gen[j] != ref_end {
                lab.push(gen[j]);
                j += 1;
            }
            label = tk.decode(&lab);
            i = j + 1;
            continue;
        }
        if t == box_start {
            let mut j = i + 1;
            let mut coords: Vec<i32> = Vec::new();
            while j < gen.len() && gen[j] != box_end {
                if gen[j] >= coord_start && gen[j] <= coord_end {
                    coords.push(gen[j] - coord_start);
                }
                j += 1;
            }
            if coords.len() == 4 {
                let (cw, ch) = (pre.target_w as f32, pre.target_h as f32);
                out.push((
                    label.clone(),
                    coords[0] as f32 / 1000.0 * cw,
                    coords[1] as f32 / 1000.0 * ch,
                    coords[2] as f32 / 1000.0 * cw,
                    coords[3] as f32 / 1000.0 * ch,
                ));
            }
            i = j + 1;
            continue;
        }
        i += 1;
    }
    eprintln!("detections: {}", out.len());
    for (lab, x1, y1, x2, y2) in &out {
        eprintln!("  {lab:10} [{x1:.2}, {y1:.2}, {x2:.2}, {y2:.2}]");
    }
    // The result goes to stdout, in exactly the shape the C++ reference CLI
    // emits, so the two can be diffed byte for byte:
    //   {"detections":[{"label":"cat","box":[3.14,50.62,221.76,443.52]}, ...]}
    // Every diagnostic above went to stderr, so stdout is pure JSON.
    let mut s = String::from("{\"detections\":[");
    for (i, (lab, x1, y1, x2, y2)) in out.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"label\":\"");
        json_escape(&mut s, lab);
        s.push_str(&format!("\",\"box\":[{x1:.2},{y1:.2},{x2:.2},{y2:.2}]}}"));
    }
    s.push_str("]}");
    eprintln!("generated : {gen:?}");
    eprintln!("total     : {:.2} s", t0.elapsed().as_secs_f64());
    // print!, not println!: the reference CLI emits the object with no trailing
    // newline, and the two outputs are diffed byte for byte.
    use std::io::Write;
    let mut stdout = std::io::stdout();
    stdout.write_all(s.as_bytes()).map_err(|e| e.to_string())?;
    stdout.flush().map_err(|e| e.to_string())?;
    Ok(())
}

fn cmd_gpuinfo() -> Result<(), String> {
    // The device query used to be open-coded here; the shared toolkit owns it
    // now (lightgpu::vm::device), including the primary-context bring-up and the
    // blocking-sync scheduling flag.
    let d = lightgpu::vm::device()?;
    println!("libcuda   : {}", d.driver.lib_path);
    println!(
        "device    : {} (sm_{}{}, {} SMs, {} KiB smem/block)",
        d.name,
        d.cc_major,
        d.cc_minor,
        d.sm_count,
        d.smem_per_block / 1024
    );
    let free = lightgpu::vm::free_vram()?;
    println!("vram      : {:.2} GiB free", free as f64 / (1u64 << 30) as f64);

    // Load the embedded fatbin and resolve one known entry point, proving module
    // load works on this driver before any kernel exists.
    #[cfg(feature = "cuda")]
    {
        let m = lightgpu::vm::Module::load(cuda::embed_fatbin())?;
        m.func("lg_noop")?;
        println!(
            "fatbin    : {} bytes, loaded module, resolved lg_noop",
            cuda::embed_fatbin().len()
        );
    }
    #[cfg(not(feature = "cuda"))]
    println!("fatbin    : none (built without the `cuda` feature)");
    Ok(())
}

/// IEEE half -> f32 (matches the GPU kernel's dequantization).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal: normalize
            let mut e = 127u32 - 15 + 1;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((f & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub fn build_prompt(tok: &tokenizer::Tokenizer, gh: usize, gw: usize, query: &str) -> Vec<i32> {
    let n = (gh / 2) * (gw / 2);
    let mut s = String::new();
    s.push_str("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
    s.push_str("<|im_start|>user\n");
    s.push_str("<image 1><img>");
    for _ in 0..n {
        s.push_str("<IMG_CONTEXT>");
    }
    s.push_str("</img>");
    s.push_str(query);
    s.push_str("<|im_end|>\n<|im_start|>assistant\n");
    tok.encode(&s)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(|s| s.as_str()) {
        Some("detect") => match parse_model(&args[1..]) {
            Ok((model, pos)) if pos.len() == 2 => cmd_detect(&model, &pos[0], &pos[1]),
            Ok(_) => Err("detect needs an image and a query".to_string()),
            Err(e) => Err(e),
        },
        Some("inspect") => match parse_model(&args[1..]) {
            Ok((model, pos)) if pos.is_empty() => cmd_inspect(&model),
            Ok((_, pos)) if pos.len() == 1 => cmd_inspect(&PathBuf::from(&pos[0])),
            Ok(_) => Err("inspect takes no positional arguments".to_string()),
            Err(e) => Err(e),
        },
        Some("gpuinfo") if args.len() == 1 => cmd_gpuinfo(),
        _ => {
            usage();
            return ExitCode::from(2);
        }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
