#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Haitsma, HaitsmaConfig};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let min_len = 5_000 * 2;
    if input.samples.len() < min_len {
        return;
    }

    let samples = &input.samples[..min_len];
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::HZ_5000,
    };

    let mut fp = Haitsma::new(HaitsmaConfig::default());
    let Ok(fpr) = fp.extract(buf) else {
        return;
    };

    for &frame in &fpr.frames {
        let bytes: [u8; 4] = frame.to_le_bytes();
        let roundtripped = u32::from_le_bytes(bytes);
        assert_eq!(frame, roundtripped);
    }
});
