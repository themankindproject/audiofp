//! Property-based tests via [`proptest`](https://docs.rs/proptest).
//!
//! These probe invariants that should hold across *any* valid input,
//! not just hand-picked inputs. They run a small number of cases by
//! default to keep `cargo test` fast; bump `PROPTEST_CASES` in your
//! environment if you want deeper coverage:
//!
//! ```bash
//! PROPTEST_CASES=2000 cargo test --test property
//! ```

use audiofp::classical::{Haitsma, Panako, StreamingHaitsma, StreamingPanako, StreamingWang, Wang};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate, StreamingFingerprinter};
use proptest::prelude::*;

const TONE_LO: f32 = 880.0;
const TONE_HI: f32 = 1320.0;

/// Deterministic xorshift32 + two-tone synthesiser. Same generator as
/// every other test in the suite, so failing inputs reproduce easily.
fn synth(seed: u32, sr: u32, n_samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_samples);
    let mut x = seed.max(1);
    for i in 0..n_samples {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        let s = 0.5 * (2.0 * std::f32::consts::PI * TONE_LO * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * TONE_HI * t).sin()
            + noise;
        out.push(s);
    }
    out
}

/// Strategy: a random sequence of chunk sizes in `1..=max_chunk` whose
/// sum is `total_samples`.
fn chunk_pattern(total_samples: usize, max_chunk: usize) -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(1usize..=max_chunk, 1..=200).prop_map(move |sizes| {
        let mut out = Vec::with_capacity(sizes.len());
        let mut remaining = total_samples;
        for s in sizes {
            if remaining == 0 {
                break;
            }
            let n = s.min(remaining);
            out.push(n);
            remaining -= n;
        }
        if remaining > 0 {
            out.push(remaining);
        }
        out
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// `StreamingWang` must produce the same hash multiset as `Wang::extract`
    /// regardless of how the input is chunked.
    #[test]
    fn streaming_wang_offline_equivalence(
        seed in 1u32..1024,
        chunks in chunk_pattern(8_000 * 4, 4_000),
    ) {
        let total: usize = chunks.iter().sum();
        let samples = synth(seed, 8_000, total);

        let mut offline = Wang::default();
        let off = offline
            .extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })
            .unwrap();

        let mut stream = StreamingWang::default();
        let mut online = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().into_iter().map(|(_, h)| h));

        let mut a = off.hashes;
        let mut b = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        prop_assert_eq!(a, b);
    }

    /// Same property for `StreamingPanako`.
    #[test]
    fn streaming_panako_offline_equivalence(
        seed in 1u32..1024,
        chunks in chunk_pattern(8_000 * 4, 4_000),
    ) {
        let total: usize = chunks.iter().sum();
        let samples = synth(seed, 8_000, total);

        let mut offline = Panako::default();
        let off = offline
            .extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })
            .unwrap();

        let mut stream = StreamingPanako::default();
        let mut online = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().into_iter().map(|(_, h)| h));

        let mut a = off.hashes;
        let mut b = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
        prop_assert_eq!(a, b);
    }

    /// Same property for `StreamingHaitsma`. Haitsma's per-frame hash
    /// is position-aligned with offline, so we can compare frame-by-frame.
    #[test]
    fn streaming_haitsma_offline_equivalence(
        seed in 1u32..1024,
        chunks in chunk_pattern(5_000 * 4, 2_500),
    ) {
        let total: usize = chunks.iter().sum();
        let samples = synth(seed, 5_000, total);

        let mut offline = Haitsma::default();
        let off = offline
            .extract(AudioBuffer { samples: &samples, rate: SampleRate::new(5_000).unwrap() })
            .unwrap();

        let mut stream = StreamingHaitsma::default();
        let mut online: Vec<u32> = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().into_iter().map(|(_, h)| h));

        prop_assert_eq!(off.frames, online);
    }

    /// `Fingerprinter::extract` is deterministic — running it twice on
    /// the same input produces identical output.
    #[test]
    fn wang_extract_is_deterministic(seed in 1u32..1024) {
        let samples = synth(seed, 8_000, 8_000 * 3);

        let mut w1 = Wang::default();
        let f1 = w1
            .extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })
            .unwrap();
        let mut w2 = Wang::default();
        let f2 = w2
            .extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })
            .unwrap();
        prop_assert_eq!(f1.hashes, f2.hashes);
    }
}
