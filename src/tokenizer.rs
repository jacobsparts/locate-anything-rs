// Qwen2 byte-level BPE tokenizer, ported from the reference src/tokenizer.cpp.
// Data comes from the .laqt container header (tokens / token_types / merges).
use std::collections::{HashMap, HashSet};

fn cpt_to_utf8(cp: u32, out: &mut String) {
    // Encode the CODEPOINT as UTF-8 (a Rust String is UTF-8). The earlier byte-wise
    // version pushed raw bytes as chars, which re-encoded >=0x80 bytes as 2-byte
    // sequences and broke every GPT-2-mapped piece.
    match char::from_u32(cp) {
        Some(ch) => out.push(ch),
        None => out.push(char::REPLACEMENT_CHARACTER),
    }
}

fn utf8_to_cpts(s: &str) -> Vec<u32> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let n = b.len();
    while i < n {
        let c = b[i];
        let (mut cp, len): (u32, usize) = if c < 0x80 {
            (c as u32, 1)
        } else if (c >> 5) == 0x6 {
            ((c & 0x1F) as u32, 2)
        } else if (c >> 4) == 0xE {
            ((c & 0x0F) as u32, 3)
        } else if (c >> 3) == 0x1E {
            ((c & 0x07) as u32, 4)
        } else {
            (c as u32, 1)
        };
        if i + len > n {
            out.push(c as u32);
            i += 1;
            continue;
        }
        let mut ok = true;
        for k in 1..len {
            let cc = b[i + k];
            if (cc & 0xC0) != 0x80 {
                ok = false;
                break;
            }
            cp = (cp << 6) | ((cc & 0x3F) as u32);
        }
        if !ok {
            out.push(c as u32);
            i += 1;
            continue;
        }
        out.push(cp);
        i += len;
    }
    out
}

fn is_ws(cp: u32) -> bool {
    matches!(
        cp,
        0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000
            ..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000
    )
}
fn is_num(cp: u32) -> bool {
    (b'0' as u32..=b'9' as u32).contains(&cp)
}
fn is_letter(cp: u32) -> bool {
    if cp > 0x10FFFF {
        return false;
    }
    if (cp >= b'a' as u32 && cp <= b'z' as u32) || (cp >= b'A' as u32 && cp <= b'Z' as u32) {
        return true;
    }
    if cp >= 0x80 && !is_ws(cp) && !is_num(cp) {
        return true;
    }
    false
}
fn to_lower(cp: u32) -> u32 {
    if cp >= b'A' as u32 && cp <= b'Z' as u32 {
        cp + 32
    } else {
        cp
    }
}

const OOR: u32 = 0xFFFF_FFFF;

fn qwen2_split(text: &str) -> Vec<String> {
    let cpts = utf8_to_cpts(text);
    let n = cpts.len();
    let mut words: Vec<String> = Vec::new();
    let get = |p: usize| -> u32 {
        if p < n {
            cpts[p]
        } else {
            OOR
        }
    };
    let mut prev = 0usize;
    let add = |end: usize, prev: &mut usize, words: &mut Vec<String>| {
        if end > *prev {
            let mut w = String::new();
            for p in *prev..end {
                cpt_to_utf8(cpts[p], &mut w);
            }
            words.push(w);
        }
        *prev = end;
    };
    let mut pos = 0usize;
    while pos < n {
        let cp = get(pos);
        if cp == '\'' as u32 && pos + 1 < n {
            let c1 = to_lower(get(pos + 1));
            if c1 == 's' as u32 || c1 == 't' as u32 || c1 == 'm' as u32 || c1 == 'd' as u32 {
                add(pos + 2, &mut prev, &mut words);
                pos = prev;
                continue;
            }
            if pos + 2 < n {
                let c2 = to_lower(get(pos + 2));
                if (c1 == 'r' as u32 && c2 == 'e' as u32)
                    || (c1 == 'v' as u32 && c2 == 'e' as u32)
                    || (c1 == 'l' as u32 && c2 == 'l' as u32)
                {
                    add(pos + 3, &mut prev, &mut words);
                    pos = prev;
                    continue;
                }
            }
        }
        if !(cp == '\r' as u32 || cp == '\n' as u32 || is_num(cp)) {
            if is_letter(cp) || is_letter(get(pos + 1)) {
                let mut p = pos + 1;
                while is_letter(get(p)) {
                    p += 1;
                }
                add(p, &mut prev, &mut words);
                pos = prev;
                continue;
            }
        }
        if is_num(cp) {
            add(pos + 1, &mut prev, &mut words);
            pos = prev;
            continue;
        }
        {
            let fcp = if cp == ' ' as u32 { get(pos + 1) } else { cp };
            let fcp_ok = fcp != OOR && !(is_ws(fcp) || is_letter(fcp) || is_num(fcp));
            if fcp_ok && cp != OOR {
                let mut p = pos + if cp == ' ' as u32 { 1 } else { 0 };
                loop {
                    let q = get(p);
                    if q == OOR || is_ws(q) || is_letter(q) || is_num(q) {
                        break;
                    }
                    p += 1;
                }
                let mut q = get(p);
                while q == '\r' as u32 || q == '\n' as u32 {
                    p += 1;
                    q = get(p);
                }
                add(p, &mut prev, &mut words);
                pos = prev;
                continue;
            }
        }
        let mut nws = 0usize;
        let mut last_rn = 0usize;
        while is_ws(get(pos + nws)) {
            let q = get(pos + nws);
            if q == '\r' as u32 || q == '\n' as u32 {
                last_rn = pos + nws + 1;
            }
            nws += 1;
        }
        if last_rn > 0 {
            add(last_rn, &mut prev, &mut words);
            pos = prev;
            continue;
        }
        if nws > 1 && get(pos + nws) != OOR {
            add(pos + nws - 1, &mut prev, &mut words);
            pos = prev;
            continue;
        }
        if nws > 0 {
            add(pos + nws, &mut prev, &mut words);
            pos = prev;
            continue;
        }
        add(pos + 1, &mut prev, &mut words);
        pos = prev;
    }
    words
}

pub struct Tokenizer {
    pub id_to_piece: Vec<String>,
    piece_to_id: HashMap<String, i32>,
    merge_rank: HashMap<String, i32>,
    byte_to_str: Vec<String>,
    cpt_to_byte: HashMap<u32, u8>,
    special_set: HashSet<String>,
    special_lens: Vec<usize>,
}

impl Tokenizer {
    pub fn load(tokens: &[String], token_types: &[i32], merges: &[String]) -> Option<Self> {
        if tokens.is_empty() {
            return None;
        }
        let mut piece_to_id = HashMap::with_capacity(tokens.len() * 2);
        for (i, p) in tokens.iter().enumerate() {
            piece_to_id.insert(p.clone(), i as i32);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len() * 2);
        for (r, m) in merges.iter().enumerate() {
            merge_rank.insert(m.clone(), r as i32);
        }
        let mut byte_to_str = vec![String::new(); 256];
        let mut cpt_to_byte: HashMap<u32, u8> = HashMap::new();
        let mut used = [false; 256];
        let assign = |b: usize,
                      cp: u32,
                      byte_to_str: &mut Vec<String>,
                      cpt_to_byte: &mut HashMap<u32, u8>,
                      used: &mut [bool; 256]| {
            used[b] = true;
            let mut u = String::new();
            cpt_to_utf8(cp, &mut u);
            byte_to_str[b] = u;
            cpt_to_byte.insert(cp, b as u8);
        };
        for b in 0x21..=0x7E {
            assign(b, b as u32, &mut byte_to_str, &mut cpt_to_byte, &mut used);
        }
        for b in 0xA1..=0xAC {
            assign(b, b as u32, &mut byte_to_str, &mut cpt_to_byte, &mut used);
        }
        for b in 0xAE..=0xFF {
            assign(b, b as u32, &mut byte_to_str, &mut cpt_to_byte, &mut used);
        }
        let mut nn = 0u32;
        for b in 0..256 {
            if !used[b] {
                assign(b, 256 + nn, &mut byte_to_str, &mut cpt_to_byte, &mut used);
                nn += 1;
            }
        }
        let mut special_set = HashSet::new();
        let mut lens: HashSet<usize> = HashSet::new();
        for (i, p) in tokens.iter().enumerate() {
            if i < token_types.len() && token_types[i] == 4 {
                if !p.is_empty() {
                    special_set.insert(p.clone());
                    lens.insert(p.len());
                }
            }
        }
        let mut special_lens: Vec<usize> = lens.into_iter().collect();
        special_lens.sort_unstable_by(|a, b| b.cmp(a));
        Some(Tokenizer {
            id_to_piece: tokens.to_vec(),
            piece_to_id,
            merge_rank,
            byte_to_str,
            cpt_to_byte,
            special_set,
            special_lens,
        })
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<i32>) {
        let mut syms: Vec<String> = Vec::new();
        for cp in utf8_to_cpts(word) {
            let mut s = String::new();
            cpt_to_utf8(cp, &mut s);
            syms.push(s);
        }
        while syms.len() > 1 {
            let mut best_rank = i32::MAX;
            let mut best_i = 0usize;
            let mut found = false;
            for i in 0..syms.len() - 1 {
                let key = format!("{} {}", syms[i], syms[i + 1]);
                if let Some(&r) = self.merge_rank.get(&key) {
                    if r < best_rank {
                        best_rank = r;
                        best_i = i;
                        found = true;
                    }
                }
            }
            if !found {
                break;
            }
            let merged = format!("{}{}", syms[best_i], syms[best_i + 1]);
            syms[best_i] = merged;
            syms.remove(best_i + 1);
        }
        for s in syms {
            if let Some(&id) = self.piece_to_id.get(&s) {
                out.push(id);
                continue;
            }
            for cp in utf8_to_cpts(&s) {
                let mut one = String::new();
                cpt_to_utf8(cp, &mut one);
                if let Some(&id) = self.piece_to_id.get(&one) {
                    out.push(id);
                }
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut ids: Vec<i32> = Vec::new();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut i = 0usize;
        let mut run_start = 0usize;
        // flush_run as a local closure needs &mut ids; emulate with an inline block.
        macro_rules! flush_run {
            ($end:expr) => {{
                let end = $end;
                if end > run_start {
                    let run = &text[run_start..end];
                    for word in qwen2_split(run) {
                        let mut enc = String::new();
                        for c in word.as_bytes() {
                            enc.push_str(&self.byte_to_str[*c as usize]);
                        }
                        self.bpe_word(&enc, &mut ids);
                    }
                }
            }};
        }
        while i < n {
            let mut matched = false;
            for &l in self.special_lens.iter() {
                if i + l > n {
                    continue;
                }
                // `get` yields None on a non-char-boundary; special tokens are ASCII and the
                // reference C++ walks bytes, so this matches its behavior without panicking.
                let cand = match text.get(i..i + l) {
                    Some(s) => s,
                    None => continue,
                };
                if self.special_set.contains(cand) {
                    flush_run!(i);
                    ids.push(*self.piece_to_id.get(cand).unwrap());
                    i += l;
                    run_start = i;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }
        flush_run!(n);
        ids
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let mut mapped = String::new();
        for &id in ids {
            if id >= 0 && (id as usize) < self.id_to_piece.len() {
                mapped.push_str(&self.id_to_piece[id as usize]);
            }
        }
        let mut out = String::new();
        for cp in utf8_to_cpts(&mapped) {
            if let Some(&b) = self.cpt_to_byte.get(&cp) {
                out.push(b as char);
            } else {
                cpt_to_utf8(cp, &mut out);
            }
        }
        out
    }
}
