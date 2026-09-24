//! Reader for the `.laqt` container produced by tools/prequant.py.
//!
//! Layout (little endian):
//!   magic      : 8 bytes, b"LAQTGGUF"
//!   version    : u32
//!   header_len : u64
//!   header     : header_len bytes of UTF-8 JSON
//!   payloads   : 4096-byte aligned, tensor payload i at `header_end + offset`
//!
//! The file is mmapped read-only and tensors are exposed as byte slices into
//! that mapping, so a payload can go straight to VRAM with `cuMemcpyHtoD` with
//! no intermediate copy.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"LAQTGGUF";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    Q8_0,
}

impl Dtype {
    pub fn parse(s: &str) -> Result<Dtype, String> {
        match s {
            "f32" => Ok(Dtype::F32),
            "q8_0" => Ok(Dtype::Q8_0),
            other => Err(format!("unknown dtype `{other}`")),
        }
    }
    /// Bytes occupied by `n` elements of this dtype.
    pub fn nbytes(&self, n: usize) -> usize {
        match self {
            Dtype::F32 => n * 4,
            Dtype::Q8_0 => (n / 32) * 34,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    /// Name of the tensor this one aliases (tied weights), if any.
    pub alias_of: Option<String>,
}

/// A mmapped `.laqt` container.
pub struct Container {
    pub version: u32,
    map: *const u8,
    map_len: usize,
    pub config: Vec<(String, Json)>,
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub tensors: HashMap<String, TensorInfo>,
    pub order: Vec<String>,
}

// The mapping is read-only and shared; the raw pointer is immutable after load.
unsafe impl Send for Container {}
unsafe impl Sync for Container {}

impl Container {
    pub fn open(path: &Path) -> Result<Container, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
        if len < 20 {
            return Err(format!("{}: too small to be a .laqt", path.display()));
        }
        let map: *mut c_void;
        unsafe {
            let p = libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            );
            if p == libc::MAP_FAILED {
                return Err(format!(
                    "mmap {}: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                ));
            }
            map = p;
            // Ask the kernel to read ahead sequentially; the loader walks the
            // payloads in order, so this turns the load into one big stream.
            libc::madvise(p, len, libc::MADV_SEQUENTIAL);
        }
        let base = map as *const u8;
        let hdr = unsafe { std::slice::from_raw_parts(base, len) };
        if &hdr[0..8] != MAGIC {
            unsafe { libc::munmap(map, len) };
            return Err(format!(
                "{}: bad magic (not a .laqt container)",
                path.display()
            ));
        }
        let version = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
        let header_len = u64::from_le_bytes(hdr[12..20].try_into().unwrap()) as usize;
        if 20 + header_len > len {
            unsafe { libc::munmap(map, len) };
            return Err(format!("{}: truncated header", path.display()));
        }
        let json = std::str::from_utf8(&hdr[20..20 + header_len])
            .map_err(|e| format!("header is not UTF-8: {e}"))?;
        let root = match parse_json(json) {
            Ok(v) => v,
            Err(e) => {
                unsafe { libc::munmap(map, len) };
                return Err(format!("{}: header JSON: {e}", path.display()));
            }
        };

        let mut c = Container {
            version,
            map: base,
            map_len: len,
            // Tensor offsets in the header are ABSOLUTE file offsets (the first
            // payload sits at align_up(20 + header_len, 4096)), so nothing is
            // added here.
            config: Vec::new(),
            tokens: Vec::new(),
            token_types: Vec::new(),
            merges: Vec::new(),
            tensors: HashMap::new(),
            order: Vec::new(),
        };
        c.parse_header(&root)?;
        Ok(c)
    }

    /// Split a `KEY=VALUE` style config section; values keep their JSON form.
    fn parse_header(&mut self, root: &Json) -> Result<(), String> {
        let obj = root.as_object().ok_or("header: not an object")?;
        if let Some(Json::Object(cfg)) = obj.get("config") {
            let mut keys: Vec<_> = cfg.keys().cloned().collect();
            keys.sort();
            for k in keys {
                self.config.push((k.clone(), cfg[&k].clone()));
            }
        }
        if let Some(Json::Array(a)) = obj.get("tokens") {
            self.tokens = a
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect();
        }
        if let Some(Json::Array(a)) = obj.get("token_types") {
            self.token_types = a.iter().map(|v| v.as_i64().unwrap_or(0) as i32).collect();
        }
        if let Some(Json::Array(a)) = obj.get("merges") {
            self.merges = a
                .iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect();
        }
        let arr = obj
            .get("tensors")
            .and_then(|v| v.as_array())
            .ok_or("header: no tensors")?;
        for t in arr {
            let o = t.as_object().ok_or("tensor entry: not an object")?;
            let name = o
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or("tensor: no name")?
                .to_string();
            let dtype = Dtype::parse(
                o.get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or("tensor: no dtype")?,
            )?;
            let shape: Vec<usize> = o
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or("tensor: no shape")?
                .iter()
                .map(|v| v.as_i64().unwrap_or(0) as usize)
                .collect();
            let offset = o
                .get("offset")
                .and_then(|v| v.as_i64())
                .ok_or("tensor: no offset")? as usize;
            let nbytes = o
                .get("nbytes")
                .and_then(|v| v.as_i64())
                .ok_or("tensor: no nbytes")? as usize;
            let alias_of = o
                .get("alias_of")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let expect = dtype.nbytes(shape.iter().product::<usize>());
            if expect != nbytes {
                return Err(format!(
                    "{name}: nbytes {nbytes} != expected {expect} for {:?}",
                    shape
                ));
            }
            self.order.push(name.clone());
            self.tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dtype,
                    shape,
                    offset,
                    nbytes,
                    alias_of,
                },
            );
        }
        Ok(())
    }

    /// Raw payload bytes for a tensor, borrowed from the mapping.
    pub fn raw(&self, name: &str) -> Result<&[u8], String> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}`"))?;
        // Aliases share the target's payload.
        let t = match &t.alias_of {
            Some(target) => self
                .tensors
                .get(target)
                .ok_or_else(|| format!("{name}: bad alias `{target}`"))?,
            None => t,
        };
        let start = t.offset;
        if start + t.nbytes > self.map_len {
            return Err(format!("{name}: payload out of file bounds"));
        }
        Ok(unsafe { std::slice::from_raw_parts(self.map.add(start), t.nbytes) })
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn payload_bytes(&self) -> usize {
        self.tensors
            .values()
            .filter(|t| t.alias_of.is_none())
            .map(|t| t.nbytes)
            .sum()
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.map as *mut c_void, self.map_len) };
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON (enough for the container header: objects, arrays, strings,
// numbers, bools, null). No escapes beyond the standard set.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Array(Vec<Json>),
    Object(HashMap<String, Json>),
}

impl Json {
    pub fn as_object(&self) -> Option<&HashMap<String, Json>> {
        match self {
            Json::Object(m) => Some(m),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Num(n) => Some(*n as i64),
            _ => None,
        }
    }
}

pub fn parse_json(s: &str) -> Result<Json, String> {
    let b = s.as_bytes();
    let mut p = 0usize;
    let v = parse_value(b, &mut p)?;
    Ok(v)
}

fn skip_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && matches!(b[*p], b' ' | b'\t' | b'\n' | b'\r') {
        *p += 1;
    }
}

fn parse_value(b: &[u8], p: &mut usize) -> Result<Json, String> {
    skip_ws(b, p);
    if *p >= b.len() {
        return Err("unexpected end of input".into());
    }
    match b[*p] {
        b'{' => parse_object(b, p),
        b'[' => parse_array(b, p),
        b'"' => Ok(Json::Str(parse_string(b, p)?)),
        b't' | b'f' => {
            let t = &b[*p..(*p + 4).min(b.len())];
            if t.starts_with(b"true") {
                *p += 4;
                Ok(Json::Bool(true))
            } else if t.starts_with(b"fals") && *p + 5 <= b.len() {
                *p += 5;
                Ok(Json::Bool(false))
            } else {
                Err(format!("bad literal at {p}"))
            }
        }
        b'n' => {
            *p += 4;
            Ok(Json::Null)
        }
        _ => parse_number(b, p),
    }
}

fn parse_number(b: &[u8], p: &mut usize) -> Result<Json, String> {
    let start = *p;
    if *p < b.len() && (b[*p] == b'-' || b[*p] == b'+') {
        *p += 1;
    }
    while *p < b.len()
        && (b[*p].is_ascii_digit() || matches!(b[*p], b'.' | b'e' | b'E' | b'-' | b'+'))
    {
        *p += 1;
    }
    let s = std::str::from_utf8(&b[start..*p]).map_err(|e| e.to_string())?;
    s.parse::<f64>()
        .map(Json::Num)
        .map_err(|e| format!("bad number `{s}`: {e}"))
}

fn parse_string(b: &[u8], p: &mut usize) -> Result<String, String> {
    if b[*p] != b'"' {
        return Err(format!("expected string at {p}"));
    }
    *p += 1;
    let mut out = String::new();
    loop {
        if *p >= b.len() {
            return Err("unterminated string".into());
        }
        match b[*p] {
            b'"' => {
                *p += 1;
                return Ok(out);
            }
            b'\\' => {
                *p += 1;
                if *p >= b.len() {
                    return Err("unterminated escape".into());
                }
                let c = b[*p];
                *p += 1;
                match c {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{8}'),
                    b'f' => out.push('\u{c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if *p + 4 > b.len() {
                            return Err("truncated \\u escape".into());
                        }
                        let hex = std::str::from_utf8(&b[*p..*p + 4]).map_err(|e| e.to_string())?;
                        let cp =
                            u32::from_str_radix(hex, 16).map_err(|e| format!("bad \\u: {e}"))?;
                        *p += 4;
                        // Surrogate pairs: the tokenizer really does contain them.
                        if (0xD800..0xDC00).contains(&cp)
                            && *p + 6 <= b.len()
                            && b[*p] == b'\\'
                            && b[*p + 1] == b'u'
                        {
                            let hex2 = std::str::from_utf8(&b[*p + 2..*p + 6])
                                .map_err(|e| e.to_string())?;
                            let lo = u32::from_str_radix(hex2, 16)
                                .map_err(|e| format!("bad \\u: {e}"))?;
                            *p += 6;
                            let comb = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            out.push(char::from_u32(comb).unwrap_or('\u{fffd}'));
                        } else {
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                    }
                    other => return Err(format!("bad escape \\{}", other as char)),
                }
            }
            c if c < 0x80 => {
                out.push(c as char);
                *p += 1;
            }
            _ => {
                // Copy a whole UTF-8 sequence through.
                let len = utf8_len(b[*p]);
                if *p + len > b.len() {
                    return Err("truncated UTF-8".into());
                }
                let s = std::str::from_utf8(&b[*p..*p + len]).map_err(|e| e.to_string())?;
                out.push_str(s);
                *p += len;
            }
        }
    }
}

fn utf8_len(b: u8) -> usize {
    if b >= 0xF0 {
        4
    } else if b >= 0xE0 {
        3
    } else if b >= 0xC0 {
        2
    } else {
        1
    }
}

fn parse_array(b: &[u8], p: &mut usize) -> Result<Json, String> {
    *p += 1; // '['
    let mut out = Vec::new();
    skip_ws(b, p);
    if *p < b.len() && b[*p] == b']' {
        *p += 1;
        return Ok(Json::Array(out));
    }
    loop {
        out.push(parse_value(b, p)?);
        skip_ws(b, p);
        if *p >= b.len() {
            return Err("unterminated array".into());
        }
        match b[*p] {
            b',' => {
                *p += 1;
            }
            b']' => {
                *p += 1;
                return Ok(Json::Array(out));
            }
            other => return Err(format!("unexpected `{}` in array", other as char)),
        }
    }
}

fn parse_object(b: &[u8], p: &mut usize) -> Result<Json, String> {
    *p += 1; // '{'
    let mut out = HashMap::new();
    skip_ws(b, p);
    if *p < b.len() && b[*p] == b'}' {
        *p += 1;
        return Ok(Json::Object(out));
    }
    loop {
        skip_ws(b, p);
        let k = parse_string(b, p)?;
        skip_ws(b, p);
        if *p >= b.len() || b[*p] != b':' {
            return Err(format!("expected `:` after key `{k}`"));
        }
        *p += 1;
        let v = parse_value(b, p)?;
        out.insert(k, v);
        skip_ws(b, p);
        if *p >= b.len() {
            return Err("unterminated object".into());
        }
        match b[*p] {
            b',' => {
                *p += 1;
            }
            b'}' => {
                *p += 1;
                return Ok(Json::Object(out));
            }
            other => return Err(format!("unexpected `{}` in object", other as char)),
        }
    }
}
