#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::classical::{Panako, PanakoConfig};
use audiofp::{Fingerprinter, SampleRate};
use libfuzzer_sys::fuzz_target;

mod pcm_seed {
    include!("../common/pcm_seed.rs");
}

#[derive(Arbitrary, Debug)]
struct Input {
    fan_out: u16,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let samples = pcm_seed::synth_pcm(data, 8_000, pcm_seed::WANG_MIN, pcm_seed::WANG_MIN + 4_000);

    let cfg = {
        let mut c = PanakoConfig::default();
        c.fan_out = input.fan_out.max(1).min(10);
        c
    };

    let mut fp = Panako::new(cfg);
    let Ok(fpr) = fp.extract(&samples, SampleRate::HZ_8000) else {
        return;
    };

    for h in &fpr.hashes {
        let bytes: [u8; 16] = bytemuck::pod_read_unaligned(bytemuck::bytes_of(h));
        let roundtripped: audiofp::classical::PanakoHash = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(*h, roundtripped);
    }
});
