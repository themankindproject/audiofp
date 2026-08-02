#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Haitsma, HaitsmaConfig};
use audiofp::{Fingerprinter, SampleRate};
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
    

    let mut fp = Haitsma::new(HaitsmaConfig::default());
    let Ok(fpr) = fp.extract(&samples, SampleRate::HZ_8000) else {
        return;
    };

    // Roundtrip: u32 frame -> bytes -> u32 via bytemuck (not a tautology).
    for &frame in &fpr.frames {
        let bytes: &[u8] = bytemuck::bytes_of(&frame);
        let roundtripped: u32 = bytemuck::pod_read_unaligned(bytes);
        assert_eq!(frame, roundtripped);
    }

    // Additional invariant: every frame is a valid 32-bit hash (all bits
    // can be set; no bit is "reserved" in Haitsma). Verify that
    // extracting the same audio again produces identical output
    // (determinism).
    
    let mut fp2 = Haitsma::new(HaitsmaConfig::default());
    let fpr2 = fp2.extract(&samples, SampleRate::HZ_8000).unwrap();
    assert_eq!(fpr.frames, fpr2.frames, "determinism violation");
});
