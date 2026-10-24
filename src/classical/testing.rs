//! Test helpers shared by the classical fingerprinter unit tests.

use alloc::vec::Vec;
use core::f32::consts::PI;

/// Two-tone (880 Hz + 1320 Hz at 8 kHz) with low-amplitude xorshift32
/// noise: stable across runs, rich enough to produce many peaks.
pub(crate) fn synthetic_audio(seed: u32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut x: u32 = seed.max(1);
    for n in 0..len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
        let t = n as f32 / 8_000.0;
        let s = 0.5 * libm::sinf(2.0 * PI * 880.0 * t)
            + 0.3 * libm::sinf(2.0 * PI * 1320.0 * t)
            + noise;
        out.push(s);
    }
    out
}

/// Haitsma's 5 kHz variant: 600 Hz + 1200 Hz tones (inside the 300-2000 Hz
/// band range) with the same noise generator.
pub(crate) fn synthetic_audio_5k(seed: u32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut x: u32 = seed.max(1);
    for n in 0..len {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
        let t = n as f32 / 5_000.0;
        let s = 0.5 * libm::sinf(2.0 * PI * 600.0 * t)
            + 0.3 * libm::sinf(2.0 * PI * 1200.0 * t)
            + noise;
        out.push(s);
    }
    out
}

/// xorshift32 → split into deterministic pseudo-random chunk sizes.
pub(crate) fn chunk_sizes(seed: u32, total: usize, max_chunk: usize) -> Vec<usize> {
    let mut x = seed.max(1);
    let mut out = Vec::new();
    let mut remaining = total;
    while remaining > 0 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let n = ((x as usize) % max_chunk).max(1).min(remaining);
        out.push(n);
        remaining -= n;
    }
    out
}

/// Tests that are textually identical for the Wang and Panako landmark
/// extractors once the types are substituted. Invoke inside a `tests`
/// module that has `super::*`, `SampleRate`, `alloc::vec` and
/// `synthetic_audio` in scope.
macro_rules! landmark_shared_tests {
    ($fp:ident, $stream:ident, $cfg:ident) => {
        #[test]
        fn synthetic_signal_is_deterministic() {
            let samples = synthetic_audio(0xBEEF, 8_000 * 3);
            let mut a = $fp::default();
            let mut b = $fp::default();
            let fa = a.extract(&samples, SampleRate::HZ_8000).unwrap();
            let fb = b.extract(&samples, SampleRate::HZ_8000).unwrap();
            assert_eq!(fa.hashes, fb.hashes);
        }

        #[test]
        fn extreme_config_is_clamped_within_safe_bounds() {
            let cfg = $cfg {
                fan_out: u16::MAX,
                target_zone_t: u16::MAX,
                peaks_per_sec: u16::MAX,
                ..$cfg::default()
            };
            let fp = $fp::new(cfg);
            assert_eq!(fp.config().fan_out, 64);
            assert_eq!(fp.config().target_zone_t, 512);
            assert_eq!(fp.config().peaks_per_sec, 500);
        }

        #[test]
        fn clamped_config_still_produces_valid_hashes() {
            let cfg = $cfg {
                fan_out: u16::MAX,
                target_zone_t: u16::MAX,
                peaks_per_sec: u16::MAX,
                ..$cfg::default()
            };
            let mut fp = $fp::new(cfg);
            let samples = synthetic_audio(0xCAFE, 8_000 * 3);

            let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
            assert!(!fpr.hashes.is_empty());
        }

        #[test]
        fn streaming_extreme_config_is_clamped_within_safe_bounds() {
            let cfg = $cfg {
                fan_out: u16::MAX,
                target_zone_t: u16::MAX,
                peaks_per_sec: u16::MAX,
                ..$cfg::default()
            };
            let s = $stream::new(cfg);
            assert_eq!(s.config().fan_out, 64);
            assert_eq!(s.config().target_zone_t, 512);
            assert_eq!(s.config().peaks_per_sec, 500);
        }

        #[test]
        fn none_disables_max_input_check() {
            let cfg = $cfg {
                max_input_samples: None,
                ..$cfg::default()
            };
            let mut fp = $fp::new(cfg);
            // 16_000 samples (2 s) is above default limit but None passes.
            let samples = vec![0.0_f32; 16_000];

            fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        }

        #[test]
        fn max_hashes_enforced_rejects_too_many() {
            let cfg = $cfg {
                max_hashes: Some(10),
                ..$cfg::default()
            };
            let mut fp = $fp::new(cfg);
            let samples = synthetic_audio(0xCAFE, 8_000 * 5);

            let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
            assert!(matches!(err, AfpError::InputTooLarge { .. }));
        }

        #[test]
        fn max_pending_anchors_evicts_oldest() {
            let cfg = $cfg {
                max_pending_anchors: Some(100),
                ..$cfg::default()
            };
            let mut s = $stream::new(cfg);
            let samples = synthetic_audio(0xCAFE, 8_000 * 20);
            let mut hashes = s.push(&samples).unwrap();
            hashes.extend(s.flush().unwrap());
            assert!(s.config().max_pending_anchors.is_some());
            assert!(!hashes.is_empty(), "should produce hashes with cap=100");
        }

        #[test]
        fn max_push_samples_truncates_hostile_chunk() {
            let cfg = $cfg {
                max_push_samples: Some(512),
                ..$cfg::default()
            };
            let mut s = $stream::new(cfg);
            // One huge push must not panic; only the first 512 samples are kept.
            let samples = synthetic_audio(0xBEEF, 8_000 * 5);
            let _ = s.push(&samples).unwrap();
            let _ = s.flush().unwrap();
            assert_eq!(s.config().max_push_samples, Some(512));
        }

        #[test]
        fn extract_with_progress_is_called_and_monotonic() {
            let mut fp = $fp::default();
            let samples = synthetic_audio(0xCAFE, 8_000 * 5);

            let mut values: Vec<f32> = Vec::new();
            let result =
                fp.extract_with_progress(&samples, SampleRate::HZ_8000, |v| values.push(v));
            assert!(result.is_ok());
            // Must be called at least a few times.
            assert!(
                values.len() >= 3,
                "expected at least 3 progress calls, got {}",
                values.len()
            );
            // First value must be 0.0.
            assert_eq!(values[0], 0.0);
            // Last value must be 1.0.
            assert_eq!(*values.last().unwrap(), 1.0);
            // Must be monotonically non-decreasing.
            for w in values.windows(2) {
                assert!(w[1] >= w[0], "progress went backwards: {} → {}", w[0], w[1]);
            }
            // All values must be in [0, 1].
            for &v in &values {
                assert!((0.0..=1.0).contains(&v), "progress out of range: {v}");
            }
        }

        #[test]
        fn extract_with_progress_matches_extract_output() {
            let samples = synthetic_audio(0xDEAD, 8_000 * 4);

            let mut fp1 = $fp::default();
            let result1 = fp1.extract(&samples, SampleRate::HZ_8000).unwrap();

            let mut fp2 = $fp::default();
            let result2 = fp2
                .extract_with_progress(&samples, SampleRate::HZ_8000, |_| {})
                .unwrap();

            assert_eq!(result1.hashes, result2.hashes);
            assert_eq!(result1.frames_per_sec, result2.frames_per_sec);
        }

        #[test]
        fn extract_with_progress_short_audio_still_reports_0_and_1() {
            let mut fp = $fp::default();
            let samples = synthetic_audio(0xFACE, 8_000 * 2);

            let mut values: Vec<f32> = Vec::new();
            let _ = fp.extract_with_progress(&samples, SampleRate::HZ_8000, |v| values.push(v));
            assert_eq!(values[0], 0.0);
            assert_eq!(*values.last().unwrap(), 1.0);
        }
    };
}
pub(crate) use landmark_shared_tests;
