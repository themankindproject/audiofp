//! Bounded RIFF/RIFX and AIFF/AIFC header validation before Symphonia
//! probe.
//!
//! Symphonia 0.6.x can panic in debug/overflow-checked builds when a WAVE
//! `fmt` chunk or an AIFF `COMM` chunk reports a pathological channel
//! count (e.g. 65535 × 16-bit), because its block-align arithmetic
//! overflows `u16` inside `probe`, before `audiofp`'s own channel guard
//! runs. In release builds the same header probes cleanly and the PCM
//! decoder then allocates `1152 × channels` samples per packet buffer
//! (≈ 300 MB for a 70-byte file). This module walks the container header
//! with checked arithmetic and rejects absurd values early. It is **not** a
//! full parser sandbox — only the outer envelope and the format chunk's
//! channel / sample-width fields are inspected.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::IoError;
use crate::{AfpError, Result};

/// Same ceiling as [`decoder::MAX_DECODE_CHANNELS`]; kept local so this
/// module does not depend on private decoder constants.
const MAX_PREFLIGHT_CHANNELS: u32 = 64;

/// Which interchange-file family the header belongs to. The two share the
/// chunk walk (4-byte id, 4-byte size, payload padded to even length) and
/// differ in byte order and in where the channel / sample-width fields sit.
#[derive(Clone, Copy)]
enum Container {
    /// RIFF (`le = true`) or RIFX (`le = false`) `WAVE`; format chunk
    /// `fmt `.
    Wave { le: bool },
    /// `FORM` `AIFF` / `AIFC` (always big-endian); format chunk `COMM`.
    Aiff,
}

impl Container {
    fn le(self) -> bool {
        matches!(self, Self::Wave { le: true })
    }

    fn format_chunk_id(self) -> &'static [u8; 4] {
        match self {
            Self::Wave { .. } => b"fmt ",
            Self::Aiff => b"COMM",
        }
    }
}

/// Validate a WAVE or AIFF header on an already-opened file.
///
/// Other containers are ignored (they are Symphonia's job). The file is
/// rewound to the start before returning. Returns `Ok(())` when the header
/// is absent, well-formed, or uses channel counts within
/// [`MAX_PREFLIGHT_CHANNELS`].
///
/// The walk stops at the first format chunk: that is the only chunk it
/// validates, and junk chunks after it cost nothing. Reads go through a
/// `BufReader` so a long run of small chunks before the format chunk costs
/// buffered reads rather than two syscalls each.
pub(crate) fn preflight_header_from_file(file: &mut File, path: &Path) -> Result<()> {
    let len = file
        .metadata()
        .map_err(|e| AfpError::io_with_path(path, e))?
        .len();
    if len >= 12 {
        let mut reader = BufReader::new(&mut *file);
        walk_header(&mut reader, len).map_err(|e| match e {
            AfpError::Io(io) if io.path.is_none() => AfpError::Io(IoError::new(path, io.source)),
            other => other,
        })?;
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|e| AfpError::io_with_path(path, e))?;
    Ok(())
}

fn walk_header<R: Read + Seek>(reader: &mut BufReader<R>, len: u64) -> Result<()> {
    let io = |e: std::io::Error| AfpError::Io(IoError::without_path(e));

    let mut header = [0_u8; 12];
    reader.read_exact(&mut header).map_err(io)?;
    let container = match (&header[0..4], &header[8..12]) {
        (b"RIFF", b"WAVE") => Container::Wave { le: true },
        (b"RIFX", b"WAVE") => Container::Wave { le: false },
        (b"FORM", b"AIFF" | b"AIFC") => Container::Aiff,
        _ => return Ok(()),
    };
    let le = container.le();

    let outer_size = read_u32(&header[4..8], le);
    let container_end = 8u64.saturating_add(outer_size);
    let scan_end = len.min(container_end);

    let mut pos = 12u64;
    while pos + 8 <= scan_end {
        let mut chunk_hdr = [0_u8; 8];
        reader.read_exact(&mut chunk_hdr).map_err(io)?;
        let id = &chunk_hdr[0..4];
        let size = read_u32(&chunk_hdr[4..8], le);
        let data_off = pos + 8;

        if id == container.format_chunk_id() {
            let mut fmt = [0_u8; 16];
            let want = match container {
                Container::Wave { .. } => 16,
                // numChannels (2) + numSampleFrames (4) + sampleSize (2).
                Container::Aiff => 8,
            };
            if size >= want && data_off + want <= len {
                reader.read_exact(&mut fmt[..want as usize]).map_err(io)?;
                match container {
                    Container::Wave { .. } => validate_fmt_fields(&fmt, le)?,
                    Container::Aiff => validate_comm_fields(&fmt)?,
                }
            }
            return Ok(());
        }

        // RIFF and IFF chunk payloads are padded to an even length.
        let padded = size + (size & 1);
        match data_off.checked_add(padded) {
            Some(n) if n <= container_end.saturating_add(8) => {
                reader.seek_relative(padded as i64).map_err(io)?;
                pos = n;
            }
            _ => break,
        }
    }
    Ok(())
}

fn read_u32(bytes: &[u8], le: bool) -> u64 {
    let b = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if le {
        u32::from_le_bytes(b) as u64
    } else {
        u32::from_be_bytes(b) as u64
    }
}

fn read_u16(bytes: &[u8], le: bool) -> u16 {
    let b = [bytes[0], bytes[1]];
    if le {
        u16::from_le_bytes(b)
    } else {
        u16::from_be_bytes(b)
    }
}

fn validate_fmt_fields(fmt: &[u8], le: bool) -> Result<()> {
    let num_channels = read_u16(&fmt[2..4], le);
    let bits_per_sample = read_u16(&fmt[14..16], le);
    validate_layout("wav", num_channels, bits_per_sample)
}

/// AIFF `COMM`: `numChannels: i16`, `numSampleFrames: u32`,
/// `sampleSize: i16`, all big-endian. Read as `u16` so a negative count
/// (which Symphonia reinterprets as a huge unsigned one) is rejected too.
fn validate_comm_fields(comm: &[u8]) -> Result<()> {
    let num_channels = read_u16(&comm[0..2], false);
    let sample_size = read_u16(&comm[6..8], false);
    validate_layout("aiff", num_channels, sample_size)
}

fn validate_layout(kind: &str, num_channels: u16, bits_per_sample: u16) -> Result<()> {
    if u32::from(num_channels) > MAX_PREFLIGHT_CHANNELS {
        return Err(malformed(format!(
            "{kind} header: {num_channels} channels exceeds limit"
        )));
    }
    if bits_per_sample > 0 {
        let bytes_per_sample = u32::from(bits_per_sample).div_ceil(8);
        let align = u32::from(num_channels) * bytes_per_sample;
        if align > u32::from(u16::MAX) {
            return Err(malformed(format!(
                "{kind} header: channel × sample-width block alignment overflow"
            )));
        }
    }
    Ok(())
}

fn malformed(msg: impl Into<String>) -> AfpError {
    AfpError::Io(IoError::without_path(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_65535_channel_wav_header() {
        let bytes: [u8; 44] = [
            b'R', b'I', b'F', b'F', 36, 0, 0, 0, b'W', b'A', b'V', b'E', b'f', b'm', b't', b' ',
            16, 0, 0, 0, 1, 0, 0xFF, 0xFF, 0x40, 0x1F, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, b'd', b'a',
            b't', b'a', 0, 0, 0, 0,
        ];
        let err = validate_riff_wave_bytes(&bytes).unwrap_err();
        assert!(
            err.to_string().contains("channels"),
            "expected channel rejection, got {err}"
        );
    }

    #[test]
    fn late_fmt_is_validated_without_reading_the_junk_payload() {
        let junk_size = (64 * 1024 - 12 - 8) as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        // Placeholder RIFF size; patched after body is built.
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"junk");
        bytes.extend_from_slice(&junk_size.to_le_bytes());
        bytes.resize(bytes.len() + junk_size as usize, 0);
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 0, 0xFF, 0xFF, 0x40, 0x1F, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0]);
        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());

        let mut file = tempfile_from_bytes(&bytes);
        let err = preflight_header_from_file(&mut file, Path::new("test.wav")).unwrap_err();
        assert!(
            err.to_string().contains("channels"),
            "expected late fmt channel rejection, got {err}"
        );
        let channels = 64 * 1024 + 8 + 2;
        bytes[channels..channels + 2].copy_from_slice(&2u16.to_le_bytes());
        let mut valid = tempfile_from_bytes(&bytes);
        assert!(preflight_header_from_file(&mut valid, Path::new("valid.wav")).is_ok());
    }

    fn validate_riff_wave_bytes(buf: &[u8]) -> Result<()> {
        let mut file = tempfile_from_bytes(buf);
        preflight_header_from_file(&mut file, Path::new("test.wav"))
    }

    fn tempfile_from_bytes(bytes: &[u8]) -> File {
        let path = std::env::temp_dir().join(format!(
            "audiofp-riff-preflight-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::write(&path, bytes).unwrap();
        let file = File::open(&path).unwrap();
        // Unlink now so the fixture never outlives the test (unix keeps the
        // open handle readable; elsewhere a failed unlink is harmless).
        std::fs::remove_file(&path).ok();
        file
    }

    /// `FORM`/`AIFF` with a `COMM` chunk declaring `channels`, 16-bit,
    /// 8 kHz, followed by an empty `SSND` chunk.
    fn aiff_bytes(form: &[u8; 4], channels: u16) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"FORM");
        bytes.extend_from_slice(&0u32.to_be_bytes());
        bytes.extend_from_slice(form);
        bytes.extend_from_slice(b"COMM");
        bytes.extend_from_slice(&18u32.to_be_bytes());
        bytes.extend_from_slice(&channels.to_be_bytes());
        bytes.extend_from_slice(&0u32.to_be_bytes()); // numSampleFrames
        bytes.extend_from_slice(&16u16.to_be_bytes()); // sampleSize
        // 8000 Hz as an 80-bit IEEE extended float.
        bytes.extend_from_slice(&[0x40, 0x0B, 0xFA, 0, 0, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(b"SSND");
        bytes.extend_from_slice(&8u32.to_be_bytes());
        bytes.extend_from_slice(&[0; 8]); // offset, blockSize
        let form_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&form_size.to_be_bytes());
        bytes
    }

    /// A 70-byte AIFF claiming 65535 channels used to overflow Symphonia's
    /// `u16` block-align math inside `probe` (debug panic) or make the PCM
    /// decoder allocate ~300 MB (release). Preflight must stop it first.
    #[test]
    fn rejects_65535_channel_aiff_and_aifc_headers() {
        for form in [b"AIFF", b"AIFC"] {
            let mut file = tempfile_from_bytes(&aiff_bytes(form, u16::MAX));
            let err = preflight_header_from_file(&mut file, Path::new("x.aiff")).unwrap_err();
            assert!(
                err.to_string().contains("channels"),
                "expected channel rejection, got {err}"
            );
        }
        let mut ok = tempfile_from_bytes(&aiff_bytes(b"AIFF", 2));
        assert!(preflight_header_from_file(&mut ok, Path::new("ok.aiff")).is_ok());
    }

    /// The walk stops at the format chunk, so a flood of chunks after it
    /// is never read, and the file is left rewound for the probe.
    #[test]
    fn walk_stops_at_the_format_chunk_and_rewinds() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 0, 1, 0, 0x40, 0x1F, 0, 0, 0x80, 0x3E, 0, 0, 2, 0, 16, 0]);
        for _ in 0..10_000 {
            bytes.extend_from_slice(b"junk");
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        let riff_size = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff_size.to_le_bytes());
        let mut file = tempfile_from_bytes(&bytes);
        assert!(preflight_header_from_file(&mut file, Path::new("x.wav")).is_ok());
        let mut first = [0_u8; 4];
        file.read_exact(&mut first).unwrap();
        assert_eq!(&first, b"RIFF", "preflight must rewind the file");
    }

    #[test]
    fn ignores_short_files_and_non_wave_containers() {
        let mut short = tempfile_from_bytes(b"RIFF\x08\x00\x00\x00WAVE");
        assert!(preflight_header_from_file(&mut short, Path::new("short.wav")).is_ok());

        let mut avi = tempfile_from_bytes(b"RIFF\x1c\x00\x00\x00AVI ");
        assert!(preflight_header_from_file(&mut avi, Path::new("clip.avi")).is_ok());

        let mut flac = tempfile_from_bytes(b"fLaC\x00\x00\x00\x22");
        assert!(preflight_header_from_file(&mut flac, Path::new("x.flac")).is_ok());
    }

    #[test]
    fn rifx_big_endian_valid_header_passes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFX");
        bytes.extend_from_slice(&38u32.to_be_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_be_bytes()); // mono
        bytes.extend_from_slice(&8_000u32.to_be_bytes());
        bytes.extend_from_slice(&8_000u32.to_be_bytes());
        bytes.extend_from_slice(&1u16.to_be_bytes());
        bytes.extend_from_slice(&8u16.to_be_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&2u32.to_be_bytes());
        bytes.extend_from_slice(&[0x80, 0x00]);

        let mut file = tempfile_from_bytes(&bytes);
        assert!(preflight_header_from_file(&mut file, Path::new("be.wav")).is_ok());
    }

    #[test]
    fn preflight_attaches_path_to_malformed_fmt_errors() {
        let bytes: [u8; 44] = [
            b'R', b'I', b'F', b'F', 36, 0, 0, 0, b'W', b'A', b'V', b'E', b'f', b'm', b't', b' ',
            16, 0, 0, 0, 1, 0, 0xFF, 0xFF, 0x40, 0x1F, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, b'd', b'a',
            b't', b'a', 0, 0, 0, 0,
        ];
        let path = std::env::temp_dir().join(format!(
            "audiofp-riff-path-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::write(&path, bytes).unwrap();
        let mut file = File::open(&path).unwrap();
        let err = preflight_header_from_file(&mut file, &path).unwrap_err();
        assert!(
            err.to_string()
                .contains(path.file_name().unwrap().to_str().unwrap()),
            "malformed fmt must include the source path, got {err}"
        );
    }

    #[test]
    fn rejects_block_alignment_overflow_before_channel_cap() {
        let mut fmt = [0_u8; 16];
        fmt[0..2].copy_from_slice(&1u16.to_le_bytes());
        fmt[2..4].copy_from_slice(&64u16.to_le_bytes());
        fmt[14..16].copy_from_slice(&65_528u16.to_le_bytes());
        let err = validate_fmt_fields(&fmt, true).unwrap_err();
        assert!(
            err.to_string().contains("block alignment overflow"),
            "expected alignment overflow, got {err}"
        );
    }
}
