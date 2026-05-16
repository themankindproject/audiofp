#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Panako, PanakoConfig};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
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

    let cfg = PanakoConfig {
        fan_out: input.fan_out.max(1).min(10),
        ..Default::default()
    };

    let samples = &input.samples[..min_len];
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::HZ_8000,
    };

    let mut fp = Panako::new(cfg);
    let Ok(fpr) = fp.extract(buf) else {
        return;
    };

    for h in &fpr.hashes {
        let bytes: [u8; 16] = bytemuck::pod_read_unaligned(bytemuck::bytes_of(h));
        let roundtripped: audiofp::classical::PanakoHash = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(*h, roundtripped);
    }
});
