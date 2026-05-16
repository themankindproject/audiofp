#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Panako, PanakoConfig, StreamingPanako};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate, StreamingFingerprinter};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    samples: Vec<f32>,
    fan_out: u16,
    target_zone_t: u16,
    peaks_per_sec: u16,
    chunk_size: usize,
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
        target_zone_t: input.target_zone_t.max(1).min(128),
        peaks_per_sec: input.peaks_per_sec.max(1).min(60),
        ..Default::default()
    };

    let samples = &input.samples[..min_len];
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::HZ_8000,
    };

    let mut offline = Panako::new(cfg.clone());
    let Ok(off) = offline.extract(buf) else {
        return;
    };

    let chunk = input.chunk_size.max(1).min(8_000);
    let mut stream = StreamingPanako::new(cfg);
    let mut online = Vec::new();
    for c in samples.chunks(chunk) {
        online.extend(stream.push(c).into_iter().map(|(_, h)| h));
    }
    online.extend(stream.flush().into_iter().map(|(_, h)| h));

    let mut a = off.hashes;
    let mut b = online;
    a.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    b.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    assert_eq!(a, b);
});
