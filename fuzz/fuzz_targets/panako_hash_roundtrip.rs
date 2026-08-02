#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Panako, PanakoConfig};
use audiofp::{Fingerprinter, SampleRate};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
    fan_out: u16,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let min_len = 8_000 * 2;
    if input.samples.len() < min_len {
        return;
    }

    let cfg = {
        let mut c = PanakoConfig::default();
        c.fan_out = input.fan_out.max(1).min(10);
        c
    };

    let samples = &input.samples[..min_len];

    let mut fp = Panako::new(cfg);
    let Ok(fpr) = fp.extract(&samples, SampleRate::HZ_8000) else {
        return;
    };

    for h in &fpr.hashes {
        let bytes: [u8; 28] = bytemuck::pod_read_unaligned(bytemuck::bytes_of(h));
        let roundtripped: audiofp::classical::PanakoHash = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(*h, roundtripped);
    }
});
