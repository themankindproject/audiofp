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

mod common;

use audiofp::classical::{Haitsma, Panako, StreamingHaitsma, StreamingPanako, StreamingWang, Wang};
use audiofp::{AfpError, Fingerprinter, SampleRate, StreamingFingerprinter};
use proptest::prelude::*;

use common::audio_gen::two_tone as synth;

/// Strategy: a random sequence of chunk sizes in `1..=max_chunk` whose
/// sum is `total_samples`.
fn chunk_pattern(total_samples: usize, max_chunk: usize) -> impl Strategy<Value = Vec<usize>> {
    // Bound chunk count so proptest cannot generate thousands of 1-sample
    // pushes (slow, and not more informative than a modest partition sweep).
    let max_chunks = total_samples.div_ceil(max_chunk).clamp(1, 64);
    proptest::collection::vec(1usize..=max_chunk, 1..=max_chunks).prop_map(move |sizes| {
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
            .extract(&samples, SampleRate::HZ_8000 )
            .unwrap();

        let mut stream = StreamingWang::default();
        let mut online = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).unwrap().into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().unwrap().into_iter().map(|(_, h)| h));

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
            .extract(&samples, SampleRate::HZ_8000 )
            .unwrap();

        let mut stream = StreamingPanako::default();
        let mut online = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).unwrap().into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().unwrap().into_iter().map(|(_, h)| h));

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
            .extract(&samples, SampleRate::HZ_5000 )
            .unwrap();

        let mut stream = StreamingHaitsma::default();
        let mut online: Vec<u32> = Vec::new();
        let mut cursor = 0;
        for &n in &chunks {
            let end = cursor + n;
            online.extend(stream.push(&samples[cursor..end]).unwrap().into_iter().map(|(_, h)| h));
            cursor = end;
        }
        online.extend(stream.flush().unwrap().into_iter().map(|(_, h)| h));

        prop_assert_eq!(off.frames, online);
    }

    /// `Fingerprinter::extract` is deterministic — running it twice on
    /// the same input produces identical output.
    #[test]
    fn wang_extract_is_deterministic(seed in 1u32..1024) {
        let samples = synth(seed, 8_000, 8_000 * 3);

        let mut w1 = Wang::default();
        let f1 = w1
            .extract(&samples, SampleRate::HZ_8000 )
            .unwrap();
        let mut w2 = Wang::default();
        let f2 = w2
            .extract(&samples, SampleRate::HZ_8000 )
            .unwrap();
        prop_assert_eq!(f1.hashes, f2.hashes);
    }
}

// ---------------------------------------------------------------------------
// NaN / inf robustness. The PCM policy (`src/pcm.rs`): offline `extract`
// rejects non-finite input with `NonFiniteSample { index }` naming the first
// bad sample; streaming `push` sanitises it to 0.0 and keeps going. (These
// tests used to discard the result, so they passed whatever happened — and
// since `extract` rejects up front, they never reached the DSP chain.)
// ---------------------------------------------------------------------------

/// Inject `n_spikes` NaN or infinity values at random positions in a
/// clean synthetic signal of `n_samples` total samples.
fn inject_spikes(clean: &mut [f32], n_spikes: usize, seed: u32) {
    let mut rng: u64 = seed as u64 | 1;
    for _ in 0..n_spikes {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng as usize) % clean.len();
        // Alternate between NaN and infinity.
        clean[idx] = if (rng >> 62) & 1 == 0 {
            f32::NAN
        } else {
            f32::INFINITY
        };
    }
}

/// Offline extraction must reject `samples` naming its first non-finite
/// sample, and a stream must accept the same samples.
fn assert_non_finite_contract(samples: &[f32], rate: SampleRate) -> Result<(), TestCaseError> {
    let first = samples
        .iter()
        .position(|s| !s.is_finite())
        .expect("a non-finite sample");
    let rejects = |r: Result<(), AfpError>| matches!(r, Err(AfpError::NonFiniteSample { index }) if index == first);
    if rate == SampleRate::HZ_8000 {
        prop_assert!(rejects(Wang::default().extract(samples, rate).map(drop)));
        prop_assert!(rejects(Panako::default().extract(samples, rate).map(drop)));
        prop_assert!(StreamingWang::default().push(samples).is_ok());
        prop_assert!(StreamingPanako::default().push(samples).is_ok());
    } else {
        prop_assert!(rejects(Haitsma::default().extract(samples, rate).map(drop)));
        prop_assert!(StreamingHaitsma::default().push(samples).is_ok());
    }
    Ok(())
}

#[test]
fn all_nan_and_all_inf_audio_is_rejected_offline_and_sanitised_streaming() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_non_finite_contract(&vec![value; 8_000 * 3], SampleRate::HZ_8000).unwrap();
        assert_non_finite_contract(&vec![value; 5_000 * 3], SampleRate::HZ_5000).unwrap();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// A few NaN/inf values sprinkled into otherwise normal audio.
    #[test]
    fn landmark_extractors_reject_nan_inf_spikes(seed in 1u32..1024, n_spikes in 1usize..=20) {
        let mut samples = synth(seed, 8_000, 8_000 * 3);
        inject_spikes(&mut samples, n_spikes, seed);
        assert_non_finite_contract(&samples, SampleRate::HZ_8000)?;
    }

    #[test]
    fn haitsma_rejects_nan_inf_spikes(seed in 1u32..1024, n_spikes in 1usize..=20) {
        let mut samples = synth(seed, 5_000, 5_000 * 3);
        inject_spikes(&mut samples, n_spikes, seed);
        assert_non_finite_contract(&samples, SampleRate::HZ_5000)?;
    }
}

/// Panako is designed to be tempo-robust within ±5 %. This test
/// time-stretches the same signal by factors in [0.96, 1.04] and
/// asserts non-zero hash overlap with the original.
#[test]
fn panako_tempo_robustness() {
    let sr = 8_000u32;
    let dur = 6; // seconds — long enough for many hashes
    let n = sr as usize * dur;
    let original = synth(42, sr, n);

    let mut panako_orig = Panako::default();
    let fp_orig = panako_orig.extract(&original, SampleRate::HZ_8000).unwrap();

    // Time-stretch by linear interpolation (crude but sufficient for
    // this invariant test — it doesn't introduce spectral artifacts
    // that'd confuse peak detection at small stretch factors).
    for &factor in &[0.96_f32, 0.98, 1.02, 1.04] {
        let stretched_len = (n as f32 * factor) as usize;
        let mut stretched = Vec::with_capacity(stretched_len);
        for i in 0..stretched_len {
            let pos = i as f64 / factor as f64;
            let idx = pos.floor() as usize;
            let frac = (pos - pos.floor()) as f32;
            let a = original[idx.min(n - 1)];
            let b = original[(idx + 1).min(n - 1)];
            stretched.push(a * (1.0 - frac) + b * frac);
        }

        let mut panako_s = Panako::default();
        let fp_stretched = panako_s.extract(&stretched, SampleRate::HZ_8000).unwrap();

        // Collect hash values (ignoring timestamps, which shift).
        let orig_hashes: std::collections::HashSet<u32> =
            fp_orig.hashes.iter().map(|h| h.hash).collect();
        let stretched_hashes: std::collections::HashSet<u32> =
            fp_stretched.hashes.iter().map(|h| h.hash).collect();

        let overlap = orig_hashes.intersection(&stretched_hashes).count();
        assert!(
            overlap > 0,
            "Panako tempo invariance broken at factor={factor}: \
             0 hash overlap (orig={}, stretched={})",
            orig_hashes.len(),
            stretched_hashes.len(),
        );
    }
}
