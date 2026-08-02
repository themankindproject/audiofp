# Codec Robustness Methodology

This document describes the test corpus, methodology, and published overlap
numbers used to validate `audiofp`'s fingerprinting algorithms against lossy
codec re-encoding.

## Corpus

| Track | Artist | Duration | License |
|-------|--------|----------|---------|
| **Galway** | Kevin MacLeod | 16 s | CC-BY 3.0 |
| **Furious Freak** | Kevin MacLeod | 16 s | CC-BY 3.0 |

Both tracks are sourced from [incompetech.com](https://incompetech.com) and
redistributed via the [Espressif ESP-ADF audio samples](https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html).

### How to Obtain

The test assets are checked into `tests/assets/` and ship with the repository.
If you need to regenerate them from scratch:

1. Download the original WAV/FLAC from <https://incompetech.com>
2. Trim to 16 seconds
3. Re-encode using FFmpeg:

```bash
# Lossless references
ffmpeg -i galway.wav -c:a flac galway.flac
ffmpeg -i galway.wav galway.wav  # copy

# Lossy codecs
ffmpeg -i galway.wav -b:a 128k galway.mp3
ffmpeg -i galway.wav -c:a libvorbis -q:a 4 galway.ogg
ffmpeg -i galway.wav -c:a aac -b:a 128k galway.m4a
ffmpeg -i galway.wav -c:a pcm_s16be galway.aiff

# Stereo variant (joint-stereo MP3)
ffmpeg -i galway.wav -ac 2 -b:a 128k galway_stereo.mp3
ffmpeg -i galway.wav -ac 2 -c:a flac galway_stereo.flac

# Sample-rate ladder (Furious Freak)
for sr in 8000 11025 16000 22050 32000 44100; do
  ffmpeg -i freak.wav -ar $sr -b:a 128k freak_${sr}hz.mp3
done
```

### License

All test audio is licensed under **Creative Commons Attribution 3.0** (CC-BY 3.0).
Attribution: "Music by Kevin MacLeod (incompetech.com), licensed under CC-BY 3.0."

Full credits are in [`tests/assets/CREDITS.md`](tests/assets/CREDITS.md).

## Methodology

### Codec Round-Trip Test

For each algorithm (Wang, Panako, Haitsma):

1. **Reference extraction** — Decode the lossless FLAC and extract fingerprints
   at the algorithm's native sample rate (8 kHz for Wang/Panako, 5 kHz for
   Haitsma).

2. **Lossy extraction** — Decode each lossy variant (MP3 128 kbps, OGG-Vorbis,
   AAC-in-M4A) and extract fingerprints with identical parameters.

3. **Overlap measurement**:
   - **Wang & Panako**: Jaccard similarity of hash sets —
     `|A ∩ B| / |A ∪ B|`
   - **Haitsma**: Per-frame bit similarity —
     `Σ (32 − popcount(frame_a ⊕ frame_b)) / (N × 32)`

4. **Pass criteria**: Each codec must meet a minimum threshold (Wang ≥ 0.25,
   Panako ≥ 0.20, Haitsma ≥ 0.75 for lossy; ≥ 0.95/0.99 for lossless).

### Two-Track Discrimination

5. **Cross-track test** — Extract fingerprints from both Galway and Furious
   Freak (same algorithm, same parameters). The Jaccard overlap between
   different songs must be < 0.05 (random collision floor).

6. **Identification scenario** — Query `galway.mp3` against a "database"
   containing `galway.flac` and `freak.flac`. The correct match must be
   ≥ 5× higher overlap than the incorrect match.

### Additional Robustness Axes

- **Stereo → mono downmix**: Stereo variant vs mono of the same track (≥ 0.25 Jaccard for MP3, ≥ 0.60 for FLAC).
- **Sample-rate ladder**: MP3 files at 8 kHz through 44.1 kHz native rate, all resampled to 8 kHz before fingerprinting. All must produce > 20 hashes.
- **AIFF lossless container**: Must match FLAC at ≥ 0.95 Jaccard.

## Published Numbers

Measured on the Galway corpus (16 s), `audiofp` v0.3.x:

| Codec | Wang (Jaccard) | Panako (Jaccard) | Haitsma (bit-sim) |
|-------|---------------|-----------------|-------------------|
| WAV/FLAC (lossless) | 1.000 | — | 1.000 |
| MP3 128 kbps | 0.40 | 0.45 | 0.93 |
| OGG-Vorbis | 0.36 | 0.42 | 0.91 |
| AAC (M4A) | 0.50 | 0.54 | 0.77 |
| AIFF (lossless) | 1.000 | — | — |
| **Cross-track (different song)** | **0.001** | — | — |

> **Pass thresholds**: Wang ≥ 0.25, Panako ≥ 0.20, Haitsma ≥ 0.75.
> In practice, 5–10 matching hashes suffice for confident identification.

## How to Reproduce

### Quick (run the existing test suite)

```bash
# Run all codec robustness tests
cargo test --test codec_roundtrip --all-features -- --nocapture
cargo test --test codec_extended --all-features -- --nocapture

# Run everything (includes 44 real-audio E2E tests)
cargo test --all-features -- --nocapture 2>&1 | grep -E "(Jaccard|bit-sim|overlap)"
```

### Helper Script

A convenience script is provided at [`scripts/codec_robustness.sh`](scripts/codec_robustness.sh):

```bash
./scripts/codec_robustness.sh
```

It verifies the corpus is present, runs the codec tests, and formats the
`eprintln!` output into a summary table.

### Full E2E Suite

```bash
# All 44 real-audio tests (codec, sample-rate, stereo, identification)
cargo test --test codec_roundtrip --test codec_extended --test real_audio_e2e --all-features -- --nocapture
```

## Test Files

| File | Test Module | What It Verifies |
|------|-------------|------------------|
| `tests/codec_roundtrip.rs` | `codec_roundtrip` | Wang/Panako/Haitsma vs FLAC reference per codec |
| `tests/codec_extended.rs` | `codec_extended` | AIFF, two-track ID, stereo, sample-rate ladder |
| `tests/real_audio_e2e.rs` | `real_audio_e2e` | Segment matching, gain invariance, determinism, time-stretch, decoder edge cases |
| `tests/robustness.rs` | `robustness` | Synthetic degradation (noise, fade, offset) without real codecs |

## Interpreting Results

- **Jaccard ≥ 0.25 (Wang)**: At least 25% of unique hashes survived lossy
  re-encoding. Given ~1700+ hashes for 16 s, this means 400+ matching hashes —
  far more than the 5–10 needed for identification.

- **Haitsma bit-sim ≥ 0.75**: On average, ≥ 24 of 32 bits per frame are
  identical after re-encoding. With ~1250 frames in 16 s, this gives an
  overwhelming statistical signal.

- **Cross-track < 0.05**: Random hash collisions produce < 0.1% overlap between
  unrelated songs — the "is this the same recording?" question is decidable with
  high confidence.
