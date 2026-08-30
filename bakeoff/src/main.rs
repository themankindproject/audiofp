mod cp;

fn main() {
    // galway.flac is 44.1 kHz native; decode at native rate (no resample),
    // feed at native rate — chromaprint resamples internally.
    let (samples, sr) =
        audiofp::io::decode_to_mono("../tests/assets/galway.flac").expect("decode galway.flac");
    let raw = cp::extract(&samples, sr);
    println!(
        "chromaprint {0} | galway.flac ({1} Hz) -> {2} u32 items (b64 {3} chars)",
        cp::version(),
        sr,
        raw.len(),
        cp::to_base64(&raw).len()
    );
    assert!(
        cp::encode_decode_roundtrip(&raw),
        "encode/decode roundtrip failed"
    );
    println!("encode/decode roundtrip: OK");
}
