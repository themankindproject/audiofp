//! Minimal WAV / raw-PCM I/O for the watermark bakeoff (dev-only).
//!
//! Decoding of compressed formats (flac/ogg/mp3/aac) is done up front with
//! ffmpeg into 16-bit PCM WAV; this module only needs to read and write
//! canonical PCM WAV, which keeps the harness dependency-free.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Read a mono f32 buffer from a 16- or 32-bit float PCM WAV file.
/// Multi-channel input is downmixed by averaging.
pub fn read_wav_mono_f32(path: &Path) -> std::io::Result<Vec<f32>> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut riff = [0u8; 12];
    r.read_exact(&mut riff)?;
    assert_eq!(&riff[0..4], b"RIFF", "not RIFF: {path:?}");
    assert_eq!(&riff[8..12], b"WAVE", "not WAVE: {path:?}");

    let mut channels = 1u16;
    let mut fmt_code = 0u16;
    let mut bits = 0u16;
    let mut data: Vec<u8> = Vec::new();

    loop {
        let mut hdr = [0u8; 8];
        if r.read_exact(&mut hdr).is_err() {
            break;
        }
        let id = &hdr[0..4];
        let sz = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        let mut body = vec![0u8; sz];
        r.read_exact(&mut body)?;
        if id == b"fmt " {
            fmt_code = u16::from_le_bytes([body[0], body[1]]);
            channels = u16::from_le_bytes([body[2], body[3]]);
            bits = u16::from_le_bytes([body[14], body[15]]);
        } else if id == b"data" {
            data = body;
        }
        // chunks are word-aligned
        if sz % 2 == 1 {
            let mut pad = [0u8; 1];
            let _ = r.read_exact(&mut pad);
        }
    }

    let nch = channels.max(1) as usize;
    let mut per_ch: Vec<Vec<f32>> = (0..nch).map(|_| Vec::new()).collect();

    match (fmt_code, bits) {
        (3, 32) => {
            // IEEE float
            let n = data.len() / 4;
            let frames = n / nch;
            for fr in 0..frames {
                for (c, ch) in per_ch.iter_mut().enumerate() {
                    let o = (fr * nch + c) * 4;
                    let v = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    ch.push(v);
                }
            }
        }
        (1, 16) => {
            let n = data.len() / 2;
            let frames = n / nch;
            for fr in 0..frames {
                for (c, ch) in per_ch.iter_mut().enumerate() {
                    let o = (fr * nch + c) * 2;
                    let v = i16::from_le_bytes([data[o], data[o + 1]]) as f32 / 32768.0;
                    ch.push(v);
                }
            }
        }
        (1, 32) => {
            let n = data.len() / 4;
            let frames = n / nch;
            for fr in 0..frames {
                for (c, ch) in per_ch.iter_mut().enumerate() {
                    let o = (fr * nch + c) * 4;
                    let v = i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    ch.push(v as f32 / 2147483648.0);
                }
            }
        }
        _ => panic!("unsupported WAV: fmt={fmt_code} bits={bits}"),
    }

    if nch == 1 {
        Ok(per_ch.into_iter().next().unwrap())
    } else {
        let frames = per_ch[0].len();
        let mut out = vec![0.0f32; frames];
        for ch in &per_ch {
            for (o, &v) in out.iter_mut().zip(ch.iter()) {
                *o += v;
            }
        }
        let inv = 1.0 / nch as f32;
        for v in &mut out {
            *v *= inv;
        }
        Ok(out)
    }
}

/// Write a mono f32 buffer as a 16-bit PCM WAV file.
pub fn write_wav_mono_f32(path: &Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    let data_bytes = samples.len() * 2;
    let byte_rate = sample_rate * 2;

    w.write_all(b"RIFF")?;
    w.write_all(&((36 + data_bytes) as u32).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&1u16.to_le_bytes())?; // mono
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?; // block align
    w.write_all(&16u16.to_le_bytes())?; // bits
    w.write_all(b"data")?;
    w.write_all(&(data_bytes as u32).to_le_bytes())?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}
