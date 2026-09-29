//! Cross-platform determinism: intermediate DSP stages must produce the
//! same bits on every target.
//!
//! The fingerprint goldens (`regression.rs`, `real_audio_golden.rs`) only
//! see the final hashes, which usually absorb a 1-ULP change in the
//! spectrum; that is how `wide`'s fused `mul_add` (NEON, FMA-enabled x86),
//! its AVX-only reduction order, and rustfft's runtime-selected SIMD FFT
//! slipped through and made hashes differ between x86_64 and aarch64 on
//! real audio. These tests hash the resampler, STFT, and mel outputs
//! directly, so any such change fails on the target that has it (macOS
//! runners are aarch64; CI also runs this file with
//! `-C target-cpu=x86-64-v3` to cover FMA/AVX2).
//!
//! The expected values are the crate's output on baseline x86_64, which
//! is unchanged from earlier releases.
//!
//! Inputs are synthesised with IEEE 754 basic operations only (no `sin`,
//! `exp`, …): those are correctly rounded everywhere, whereas `std`'s
//! transcendental functions come from the platform C library.

use audiofp::dsp::mel::{MelFilterBank, MelScale};
use audiofp::dsp::resample::SincResampler;
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
use audiofp::dsp::windows::WindowKind;

/// FNV-1a over the bit patterns of `v`.
fn fnv(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Noise plus a recursive-oscillator tone under a sawtooth envelope.
fn signal(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    let (mut y1, mut y2) = (0.25_f32, 0.0_f32);
    (0..n)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let noise = (state as i32 as f32) * (0.1 / 2_147_483_648.0);
            // y[n] = 2cos(w)·y[n-1] − y[n-2]; 2cos(w) = 1.9 → w ≈ 0.318 rad.
            let tone = 1.9 * y1 - y2;
            y2 = y1;
            y1 = tone;
            let env = (i % 4410) as f32 / 4410.0;
            noise + tone * env
        })
        .collect()
}

/// Compare every `(label, actual)` against `expected`, reporting all
/// mismatches at once so one run shows which stages diverged.
fn check(actual: &[(String, u64)], expected: &[(&str, u64)]) {
    assert_eq!(actual.len(), expected.len(), "stage list changed");
    let mismatches: Vec<String> = actual
        .iter()
        .zip(expected)
        .filter(|((_, a), (_, e))| a != e)
        .map(|((label, a), (_, e))| format!("{label}: got {a:#018x}, want {e:#018x}"))
        .collect();
    for ((label, _), (want_label, _)) in actual.iter().zip(expected) {
        assert_eq!(label, want_label, "stage order changed");
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} stages differ from the baseline x86_64 output:\n  {}",
        mismatches.len(),
        actual.len(),
        mismatches.join("\n  ")
    );
}

#[test]
fn resampler_output_is_bit_identical_across_targets() {
    let mut actual = Vec::new();
    for (from, to) in [
        (44_100, 8_000),
        (44_100, 5_000),
        (48_000, 16_000),
        (22_050, 44_100),
    ] {
        let input = signal(from as usize * 2);
        let out = SincResampler::new(from, to).process(&input);
        actual.push((format!("resample {from}->{to}"), fnv(&out)));
    }
    check(
        &actual,
        &[
            ("resample 44100->8000", 0x0cd1_03d9_dd32_5bb6),
            ("resample 44100->5000", 0xdefb_2c70_036d_a64a),
            ("resample 48000->16000", 0x7424_7f2e_cd55_3e51),
            ("resample 22050->44100", 0x567d_670d_e4ea_ca31),
        ],
    );
}

#[test]
fn stft_output_is_bit_identical_across_targets() {
    let input = signal(32_000);
    let mut actual = Vec::new();
    for n_fft in [64, 256, 1024, 2048, 4096] {
        for center in [false, true] {
            let mut stft = ShortTimeFFT::new(StftConfig {
                n_fft,
                hop: n_fft / 8,
                window: WindowKind::Hann,
                center,
            });
            let (power, _, _) = stft.power_flat(&input);
            actual.push((format!("power n_fft={n_fft} center={center}"), fnv(&power)));
        }
    }
    let mut stft = ShortTimeFFT::new(StftConfig::new(1024));
    let (mag, _, _) = stft.magnitude_flat(&input);
    actual.push(("magnitude n_fft=1024".into(), fnv(&mag)));
    check(
        &actual,
        &[
            ("power n_fft=64 center=false", 0x8143_fe75_f11b_c0d1),
            ("power n_fft=64 center=true", 0x8740_0359_e7bd_18e2),
            ("power n_fft=256 center=false", 0x8533_a375_555f_24d9),
            ("power n_fft=256 center=true", 0xb8be_9971_a0c8_4e8b),
            ("power n_fft=1024 center=false", 0xdb49_e82b_be30_fc6e),
            ("power n_fft=1024 center=true", 0xa05e_eedc_b69f_35c5),
            ("power n_fft=2048 center=false", 0xb955_db82_3635_a97a),
            ("power n_fft=2048 center=true", 0x2374_fc4c_d780_d673),
            ("power n_fft=4096 center=false", 0x77fe_bb8a_bfd8_2364),
            ("power n_fft=4096 center=true", 0x3584_396d_c9e4_d68b),
            ("magnitude n_fft=1024", 0x4b06_b8e3_6898_753e),
        ],
    );
}

#[test]
fn log_mel_is_bit_identical_across_targets() {
    let input = signal(16_000);
    let mut stft = ShortTimeFFT::new(StftConfig::new(512));
    let (power, n_frames, n_bins) = stft.power_flat(&input);
    let (mag, _, _) = stft.magnitude_flat(&input);
    let mut actual = Vec::new();
    for scale in [MelScale::Slaney, MelScale::Htk] {
        let bank = MelFilterBank::new(64, 512, 16_000, 0.0, 8_000.0, scale);
        let mut from_power = vec![0.0_f32; n_frames * 64];
        let mut from_mag = vec![0.0_f32; n_frames * 64];
        for f in 0..n_frames {
            let row = f * n_bins..(f + 1) * n_bins;
            let out = f * 64..(f + 1) * 64;
            bank.log_mel_from_power(&power[row.clone()], &mut from_power[out.clone()]);
            bank.log_mel(&mag[row], &mut from_mag[out]);
        }
        actual.push((format!("log_mel_from_power {scale:?}"), fnv(&from_power)));
        actual.push((format!("log_mel {scale:?}"), fnv(&from_mag)));
    }
    check(
        &actual,
        &[
            ("log_mel_from_power Slaney", 0x7fee_e90c_55dd_8944),
            ("log_mel Slaney", 0x234c_e487_f850_33cd),
            ("log_mel_from_power Htk", 0xeee5_e3b6_be7a_71a2),
            ("log_mel Htk", 0x745f_1688_e969_2222),
        ],
    );
}
