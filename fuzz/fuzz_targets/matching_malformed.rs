#![no_main]

//! Fuzz the matching layer with structurally-malformed fingerprints.
//!
//! `WangHash` / `PanakoHash` / `HaitsmaFingerprint` have public fields and
//! derive `bytemuck::Pod`, so any bit pattern is constructible in safe
//! Rust and reachable verbatim from `from_bytes`
//! (mmap / flat-file / FFI persistence). In particular `PanakoHash`
//! documents `t_anchor < t_b < t_c`, but nothing enforces it — an
//! unguarded `t_c - t_anchor` used to panic under overflow checks and
//! silently wrap in release.
//!
//! The contract this target pins: **no input may panic or abort** the
//! matchers or the 1:N indexes. Degenerate inputs may legitimately return
//! `MatchResult::NONE` or a bogus-but-total score; they must not crash.

use libfuzzer_sys::fuzz_target;

use audiofp::classical::{
    HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangFingerprint, WangHash,
};
use audiofp::matching::{
    HaitsmaIndex, HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoIndex, PanakoMatchConfig,
    PanakoMatcher, WangIndex, WangMatchConfig, WangMatcher,
};

/// Build a Wang fingerprint from raw `(hash, t_anchor)` pairs.
fn wang(data: &[u8]) -> WangFingerprint {
    let hashes = data
        .chunks_exact(8)
        .map(|c| WangHash {
            hash: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            t_anchor: u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
        })
        .collect();
    WangFingerprint {
        hashes,
        // Also fuzz the frame rate: the matchers must soft-fail on
        // non-finite / non-positive / mismatched values.
        frames_per_sec: match data.first().copied().unwrap_or(0) % 4 {
            0 => 62.5,
            1 => 0.0,
            2 => f32::NAN,
            _ => -62.5,
        },
    }
}

/// Build a Panako fingerprint from raw `(hash, t_anchor, t_b, t_c)` quads.
fn panako(data: &[u8]) -> PanakoFingerprint {
    let hashes = data
        .chunks_exact(16)
        .map(|c| PanakoHash {
            hash: u32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            t_anchor: u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
            t_b: u32::from_le_bytes([c[8], c[9], c[10], c[11]]),
            t_c: u32::from_le_bytes([c[12], c[13], c[14], c[15]]),
        })
        .collect();
    PanakoFingerprint {
        hashes,
        frames_per_sec: 62.5,
    }
}

fn haitsma(data: &[u8]) -> HaitsmaFingerprint {
    let frames = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    HaitsmaFingerprint {
        frames,
        frames_per_sec: 78.125,
    }
}

fuzz_target!(|data: &[u8]| {
    let w = wang(data);
    let p = panako(data);
    let h = haitsma(data);

    // 1:1 matchers (both orders — the overflow was reachable on either
    // the query or the reference side).
    let wm = WangMatcher::new(WangMatchConfig::default());
    let _ = wm.match_one(&w, &w);
    let _ = wm.match_one(&w, &wang(&data[..data.len() / 2]));

    let pm = PanakoMatcher::new(PanakoMatchConfig::default());
    let _ = pm.match_one(&p, &p);
    let _ = pm.match_one(&p, &panako(&data[..data.len() / 2]));
    let _ = pm.match_one(&panako(&data[..data.len() / 2]), &p);

    let hm = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
    let _ = hm.match_one(&h, &h);

    // 1:N indexes (the Panako index had the same unguarded subtraction).
    let wi = WangIndex::build(&[w.clone()], data.first().copied().unwrap_or(100) as u32);
    let _ = wi.query(&w, &WangMatchConfig::default());
    // `insert` / `remove` must also tolerate the malformed fingerprint.
    let mut wi2 = WangIndex::build(&[], 100);
    let id = wi2.insert(&w, 100);
    assert!(wi2.remove(id));

    let pi = PanakoIndex::build(&[p.clone()], 100);
    let _ = pi.query(&p, &PanakoMatchConfig::default());

    let hi = HaitsmaIndex::build(&[h.clone()], 100);
    let _ = hi.query(&h, &HaitsmaMatchConfig::default());

    // Batch helpers share the same paths.
    let refs = [w.clone(), wang(&data[..data.len() / 2])];
    let _ = audiofp::matching::match_ranked(&wm, &w, &refs);
    let _ = audiofp::matching::match_best(&wm, &w, &refs);

    // Serialization round-trip of a malformed fingerprint must not panic
    // and must either round-trip or fail cleanly.
    let bytes = p.to_bytes();
    let _ = PanakoFingerprint::from_bytes(&bytes);
});
