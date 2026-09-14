//! Bounded RIFF/RIFX header validation before Symphonia probe.
//!
//! Symphonia 0.6.x can panic in debug/overflow-checked builds when a WAVE
//! `fmt` chunk reports pathological channel counts (e.g. 65535 × 16-bit)
//! because block-align arithmetic overflows `u16` before `audiofp`'s own
//! channel guard runs. This module walks the container header with checked
//! arithmetic and rejects absurd values early. It is **not** a full parser
//! sandbox — only the initial RIFF envelope and `fmt` PCM/IEEE fields are
//! inspected.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::IoError;
use crate::{AfpError, Result};

/// Same ceiling as [`decoder::MAX_DECODE_CHANNELS`]; kept local so this
/// module does not depend on private decoder constants.
const MAX_PREFLIGHT_CHANNELS: u32 = 64;

/// Validate a RIFF/RIFX WAVE header on an already-opened file.
///
/// Non-RIFF files are ignored (other containers are Symphonia's job).
/// The file is rewound to the start before returning. Returns `Ok(())`
/// when the header is absent, well-formed, or uses channel counts within
/// [`MAX_PREFLIGHT_CHANNELS`].
pub(crate) fn preflight_wav_from_file(file: &mut File, path: &Path) -> Result<()> {
    let len = file
        .metadata()
        .map_err(|e| AfpError::io_with_path(path, e))?
        .len();
    if len < 12 {
        rewind(file, path)?;
        return Ok(());
    }

    let mut header = [0_u8; 12];
    file.read_exact(&mut header)
        .map_err(|e| AfpError::io_with_path(path, e))?;

    let (le, form) = match &header[0..4] {
        b"RIFF" => (true, &header[8..12]),
        b"RIFX" => (false, &header[8..12]),
        _ => {
            rewind(file, path)?;
            return Ok(());
        }
    };
    if form != b"WAVE" {
        rewind(file, path)?;
        return Ok(());
    }

    let riff_size = read_u32(&header[4..8], le);
    let container_end = 8u64.saturating_add(riff_size);
    let scan_end = len.min(container_end);

    let mut pos = 12u64;
    while pos + 8 <= scan_end {
        file.seek(SeekFrom::Start(pos))
            .map_err(|e| AfpError::io_with_path(path, e))?;
        let mut chunk_hdr = [0_u8; 8];
        file.read_exact(&mut chunk_hdr)
            .map_err(|e| AfpError::io_with_path(path, e))?;
        let id = &chunk_hdr[0..4];
        let size = read_u32(&chunk_hdr[4..8], le);
        let data_off = pos + 8;

        if id == b"fmt " && size >= 16 && data_off + 16 <= len {
            file.seek(SeekFrom::Start(data_off))
                .map_err(|e| AfpError::io_with_path(path, e))?;
            let mut fmt = [0_u8; 16];
            file.read_exact(&mut fmt)
                .map_err(|e| AfpError::io_with_path(path, e))?;
            validate_fmt_fields(&fmt, le).map_err(|e| match e {
                AfpError::Io(io) => AfpError::Io(IoError::new(path, io.source)),
                other => other,
            })?;
        }

        let padded = padded_chunk_payload(size)?;
        let next = data_off.checked_add(padded);
        match next {
            Some(n) if n > pos && n <= container_end.saturating_add(8) => pos = n,
            _ => break,
        }
    }

    rewind(file, path)?;
    Ok(())
}

fn rewind(file: &mut File, path: &Path) -> Result<()> {
    file.seek(SeekFrom::Start(0))
        .map_err(|e| AfpError::io_with_path(path, e))?;
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

fn padded_chunk_payload(size: u64) -> Result<u64> {
    let pad = size & 1;
    size.checked_add(pad)
        .ok_or_else(|| malformed("wav header: chunk size overflow"))
}

fn validate_fmt_fields(fmt: &[u8], le: bool) -> Result<()> {
    let num_channels = read_u16(&fmt[2..4], le);
    let bits_per_sample = read_u16(&fmt[14..16], le);

    if u32::from(num_channels) > MAX_PREFLIGHT_CHANNELS {
        return Err(malformed(format!(
            "wav header: {num_channels} channels exceeds limit"
        )));
    }

    if bits_per_sample > 0 && bits_per_sample.is_multiple_of(8) {
        let bytes_per_sample = bits_per_sample / 8;
        let expected_align = u32::from(num_channels)
            .checked_mul(u32::from(bytes_per_sample))
            .filter(|v| *v <= u16::MAX as u32);
        if expected_align.is_none() {
            return Err(malformed(
                "wav header: channel × sample-width block alignment overflow",
            ));
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
        let err = preflight_wav_from_file(&mut file, Path::new("test.wav")).unwrap_err();
        assert!(
            err.to_string().contains("channels"),
            "expected late fmt channel rejection, got {err}"
        );
        let channels = 64 * 1024 + 8 + 2;
        bytes[channels..channels + 2].copy_from_slice(&2u16.to_le_bytes());
        let mut valid = tempfile_from_bytes(&bytes);
        assert!(preflight_wav_from_file(&mut valid, Path::new("valid.wav")).is_ok());
    }

    fn validate_riff_wave_bytes(buf: &[u8]) -> Result<()> {
        let mut file = tempfile_from_bytes(buf);
        preflight_wav_from_file(&mut file, Path::new("test.wav"))
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
        File::open(&path).unwrap()
    }
}
