#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::dsp::resample::{SincQuality, SincResampler};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
    from_sr: u32,
    to_sr: u32,
    half_taps: u8,
    kaiser_beta: f32,
    polyphase_steps: u16,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let from_sr = input.from_sr.max(1_000).min(192_000);
    let to_sr = input.to_sr.max(1_000).min(192_000);
    let half_taps = input.half_taps.max(4) as usize;
    let polyphase_steps = input.polyphase_steps.max(16).min(1024);

    let quality = SincQuality {
        half_taps,
        kaiser_beta: input.kaiser_beta.abs().max(1.0).min(20.0),
        polyphase_steps,
    };

    let resampler = SincResampler::with_quality(from_sr, to_sr, quality);
    let _output = resampler.process(&input.samples);
});
