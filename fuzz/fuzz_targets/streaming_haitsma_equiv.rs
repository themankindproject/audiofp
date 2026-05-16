#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Haitsma, HaitsmaConfig, StreamingHaitsma};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate, StreamingFingerprinter};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
    fmin: f32,
    fmax: f32,
    chunk_size: usize,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let min_len = 5_000 * 2;
    if input.samples.len() < min_len {
        return;
    }

    let fmin = input.fmin.abs().max(1.0).min(1000.0);
    let fmax = input.fmax.abs().max(fmin + 100.0).min(2000.0);
    let cfg = HaitsmaConfig { fmin, fmax };

    let samples = &input.samples[..min_len];
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::new(5_000).unwrap(),
    };

    let mut offline = Haitsma::new(cfg.clone());
    let Ok(off) = offline.extract(buf) else {
        return;
    };

    let chunk = input.chunk_size.max(1).min(5_000);
    let mut stream = StreamingHaitsma::new(cfg);
    let mut online: Vec<u32> = Vec::new();
    for c in samples.chunks(chunk) {
        online.extend(stream.push(c).into_iter().map(|(_, h)| h));
    }
    online.extend(stream.flush().into_iter().map(|(_, h)| h));

    assert_eq!(off.frames, online);
});
