// Run with: cargo test --test generate_real_goldens --all-features -- --ignored
#![cfg(feature = "std")]

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
use std::fs;

#[test]
#[ignore]
fn generate_real_audio_goldens() {
    // Wang on piano
    let samples = decode_to_mono_at("tests/assets/piano.ogg", 8_000).unwrap();
    let mut wang = Wang::default();
    let fp = wang
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap();
    let bytes: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    fs::write("tests/goldens/wang_v1_piano.bin", &bytes).unwrap();
    eprintln!("wang piano: {} hashes", fp.hashes.len());

    // Wang on speech
    let samples = decode_to_mono_at("tests/assets/speech.ogg", 8_000).unwrap();
    let fp = wang
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap();
    let bytes: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    fs::write("tests/goldens/wang_v1_speech.bin", &bytes).unwrap();
    eprintln!("wang speech: {} hashes", fp.hashes.len());

    // Panako on piano
    let samples = decode_to_mono_at("tests/assets/piano.ogg", 8_000).unwrap();
    let mut panako = Panako::default();
    let fp = panako
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap();
    let bytes: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    fs::write("tests/goldens/panako_v2_piano.bin", &bytes).unwrap();
    eprintln!("panako piano: {} hashes", fp.hashes.len());

    // Panako on speech
    let samples = decode_to_mono_at("tests/assets/speech.ogg", 8_000).unwrap();
    let fp = panako
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap();
    let bytes: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    fs::write("tests/goldens/panako_v2_speech.bin", &bytes).unwrap();
    eprintln!("panako speech: {} hashes", fp.hashes.len());

    // Haitsma on piano
    let samples = decode_to_mono_at("tests/assets/piano.ogg", 5_000).unwrap();
    let mut haitsma = Haitsma::default();
    let fp = haitsma
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_5000))
        .unwrap();
    let bytes: Vec<u8> = fp.frames.iter().flat_map(|f| f.to_le_bytes()).collect();
    fs::write("tests/goldens/haitsma_v1_piano.bin", &bytes).unwrap();
    eprintln!("haitsma piano: {} frames", fp.frames.len());

    // Haitsma on speech
    let samples = decode_to_mono_at("tests/assets/speech.ogg", 5_000).unwrap();
    let fp = haitsma
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_5000))
        .unwrap();
    let bytes: Vec<u8> = fp.frames.iter().flat_map(|f| f.to_le_bytes()).collect();
    fs::write("tests/goldens/haitsma_v1_speech.bin", &bytes).unwrap();
    eprintln!("haitsma speech: {} frames", fp.frames.len());
}
