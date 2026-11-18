# Proposed changes that are not backward compatible

Nothing in this file is implemented. It lists what the November 2026 review
of 0.4.6 found and left alone because doing it would change a public
signature, an error a caller can match on, a fingerprint, a match result, a
detector score, or a project rule. Everything that could be done without any
of that is in the `[Unreleased]` section of `CHANGELOG.md`.

Each entry says what would change for a caller, why it cannot be done
compatibly, and what it would buy. Entries are grouped by what they break
and, within a group, ordered by how much they are worth.

## 1. Fingerprints would change

The hash output of `Wang`, `Panako` and `Haitsma` is pinned to the bit by
`tests/regression.rs`, `tests/real_audio_golden.rs` and
`tests/cross_platform_determinism.rs`. Anything here needs new goldens and
means a catalog built with an older version may not match one built with a
newer one on the affected inputs.

### 1.1 A vectorised FFT

- **Now.** Every transform runs on rustfft's scalar planner. After this
  review's work the FFT core is the largest single cost of extraction, a
  little under half of the instructions of a Wang extract.
- **Change.** Use an 8-lane FFT (rustfft's SIMD planners, or a radix-4
  kernel written on `wide` so it stays free of `unsafe`).
- **Breaks.** A different butterfly order rounds differently, so spectra move
  in the last bit. Peaks that are exact ties can flip, and with them a small
  number of hashes on some inputs. rustfft's own SIMD planners also pick the
  code path from the CPU at run time, which would end the guarantee that two
  machines produce the same fingerprint; a `wide` kernel would keep it.
- **Worth.** Roughly 1.5× on extraction, the largest speed-up still available.

### 1.2 Decibels only for peak candidates

- **Now.** Every cell of the spectrogram is converted to dB, then the peak
  picker compares dB values. The conversion is about a sixth of a Wang
  extract.
- **Change.** Pick peaks on power, which orders the same way, and convert
  only the cells that survive.
- **Breaks.** `log2` is monotonic but not injective in `f32`: two different
  powers can round to the same dB value. Today those are a plateau, and a
  plateau is not a peak; on power one of them would be. The peak set changes
  on inputs with such near-ties. An exact variant (re-check near-ties in dB)
  is possible but complicated enough that it was not attempted.
- **Worth.** 10 to 15 % of extraction.

### 1.3 One dB path for the last cells of an offline extract

- **Now.** Offline `Wang` and `Panako` convert the spectrogram eight cells at
  a time and finish the last `total % 8` cells of the whole buffer (the top
  bins of the final frame) with scalar `libm::log2f`. Streaming converts every
  cell on the 8-lane path. The two can differ by one unit in the last place.
- **Change.** Pad the offline tail so it takes the 8-lane path too.
- **Breaks.** Offline hashes could change for a signal whose top bins tie
  exactly in the final frame. Real audio does not do that; a synthetic
  stationary signal can, and then offline and streaming disagree by a few
  peaks today.
- **Worth.** Removes the last known offline/streaming difference.

### 1.4 Resampler quality

- **Now.** `SincQuality::default()` is 32 input samples either side whatever
  the ratio, so when decimating the filter's transition is wide: 48 kHz to
  16 kHz attenuates just above the new Nyquist by about 23 dB, not the 80 dB
  the Kaiser window suggests. The kernel phase is the fractional position
  truncated to 1/256 of a sample (an error near −55 dB at 4 kHz).
- **Change.** Scale the tap count with the decimation ratio by default, and
  round or interpolate the phase.
- **Breaks.** Every resampled sample changes, so the fingerprint of any file
  that is not already at the extractor's rate changes. The pinned resampler
  hashes change too. `decode_to_mono_at` gets slower in proportion to the
  ratio.
- **Worth.** Less aliasing into the band the fingerprints read. Whether
  recognition improves is not measured.

### 1.5 Haitsma frames from the band alone

- **Now.** Haitsma runs a 2048-point transform per frame and reads 33 bands
  from part of the spectrum.
- **Change.** Compute only the bins the bands use (a decimated or zoom
  transform).
- **Breaks.** Different arithmetic, so band energies move in the last bits
  and frames whose band differences are near zero can flip a bit.
- **Worth.** Unmeasured.

## 2. Match results would change

### 2.1 Gate Haitsma index candidates before verifying

- **Now.** `HaitsmaIndex::query` verifies every reference that shares a
  single exact 32-bit frame with the query, at up to eight offsets.
- **Change.** Verify only references with at least two hits at one offset, or
  a minimum number of hits overall.
- **Breaks.** Recall. A true match whose bit error rate is above roughly 25 %
  can have a single exact frame in common and would be lost.
- **Worth.** The dominant cost of a query on a catalog where candidates run to
  the thousands.

### 2.2 Stop Panako's RANSAC early

- **Now.** 128 candidate lines are always tried. This review made each
  distinct line cost one pass instead of many, which removed most of the cost
  without changing any result.
- **Change.** Stop once no better line has appeared for a while.
- **Breaks.** Which of two nearly equal lines wins can change, and with it the
  reported offset and time scale in the last digits.
- **Worth.** Little now that repeated lines are counted once.

### 2.3 Discount frames that are silent on both sides in `HaitsmaMatcher`

- **Now.** Two different tracks that are both partly silent (clips padded with
  zeros to the same length, for instance) can match on their silence. Only a
  wholly degenerate region is rejected.
- **Change.** Leave frames that are silent in both fingerprints out of the bit
  error rate.
- **Breaks.** Scores and match decisions for any pair with shared silence.
- **Worth.** Removes a known false-match case.

## 3. Errors a caller can match on would change

### 3.1 A WAV or AIFF with more than 64 channels

- **Now.** `AfpError::Io` of kind `InvalidData`, from the header check. The
  same file as FLAC or Matroska gives `AfpError::UnsupportedChannels`. The
  module docs now describe this as it is.
- **Change.** Return `UnsupportedChannels` from the header check.
- **Breaks.** The variant and the `Display` text for those files.

### 3.2 Decode failures keep their kind

- **Now.** Every decode failure is `AfpError::Io` of kind `Other` with a
  message. A read fault (`PermissionDenied`, `TimedOut`, a failing disk)
  looks the same as a corrupt or unsupported file, and the source error is
  gone.
- **Change.** Pass a real I/O error through unchanged; use `InvalidData` for
  corrupt data and `Unsupported` when no format reader fits.
- **Breaks.** `err.kind()` for callers that match on `Other`.
- **Worth.** A service can tell a bad upload from its own disk fault without
  matching on message text.

### 3.3 A lenient decode in which every packet failed

- **Now.** `decode_to_mono` returns `Ok` with no samples, and the failure
  shows up later as `AudioTooShort` from `extract`.
- **Change.** Return an error when no packet decoded and at least one was
  skipped.
- **Breaks.** `Ok(empty)` becomes `Err` for those files.

### 3.4 Oversize `.afp` files

- **Now.** `InputTooLarge` names no file, although `load_all_cached` documents
  that errors identify the offending file, and its message advises raising a
  limit that is a constant.
- **Change.** Carry the path (a new field, or an `Io` wrapper) and drop the
  advice from the message.
- **Breaks.** The variant's shape or its `Display` text.

### 3.5 `StreamingNeuralEmbedder::new` and `batch_size`

- **Now.** A `batch_size` of 0 or above 4096, which `NeuralEmbedder::new`
  rejects, is accepted and never used. The docs now say so.
- **Change.** Reject it, as the offline constructor does.
- **Breaks.** A configuration that builds today would return `Config`.

### 3.6 Samples too large to transform

- **Now.** The classical watermark accepts any finite sample. Samples from
  roughly `1e36` upward overflow the transform: `embed` returns non-finite
  samples, and `detect` reports no payload and a score of 0 (LogHost
  reported `-inf` before this review).
- **Change.** Reject such input with an error.
- **Breaks.** A new error where there was none; it needs a variant or a
  message under `Config`.

### 3.7 `DecodeStats` when decoding fails

- **Now.** The counters reach the caller only on success.
- **Change.** Carry them on the error path.
- **Breaks.** The error type or the signature of `decode_to_mono_report`.

## 4. Detector scores would change

Payloads would not, on anything measured, but `ClassicalResult::score` is a
public number and both changes move it in about the sixth digit. The recovery
counts quoted in the docs would have to be measured again.

### 4.1 LogHost detection on the real-input FFT

- **Now.** Detection runs a 2048-point complex transform on real input and
  reads 300 magnitudes. QIM detection already uses the half-size real
  transform.
- **Change.** Use the real transform in LogHost detection too. Embedding must
  keep the complex one: its output is pinned.
- **Worth.** About a quarter of LogHost detection time.

### 4.2 QIM alignment search eight carriers at a time

- **Now.** Each of up to 124 trial alignments evaluates `log10f` and
  `sincosf` once per carrier, which outweighs the FFT.
- **Change.** Compute the levels and lattice phases lane-wise with the
  crate's deterministic `log2` and a fixed polynomial sine and cosine.
- **Worth.** Unmeasured; by operation count up to half of the search.

## 5. Public API to remove or tighten (the 0.5.0 batch)

- **`dsp::peaks::rolling_max_2d_pooled`**, deprecated since 0.4.6, and about
  140 lines of helpers that exist only for it.
- **The ONNX `watermark` feature and `WatermarkDetector`**, deprecated since
  0.4.5 in favour of `classical-watermark`. Two small defects in it were left unfixed
  for that reason: a cache miss holds one model copy more than the documented
  maximum while the new plan is built, and `detect_validated` clamps the
  averaged confidence and the individual scores separately, so the two can
  disagree.
- **`MelFilterBank`'s public fields** made private behind accessors.
- **`#[non_exhaustive]` on the configuration structs**, so a field can be
  added later without another breaking release.
- **`matching::score_compare`** is not a total order when a score is NaN.
  The docs now warn against using it in `sort_by`. Making it total would
  change where NaN sorts.

## 6. Needs a decision on project rules, not on the API

### 6.1 Opening a FIFO that has no writer

- **Now.** Decoding from a pipe works (fixed in this review), but opening a
  FIFO nobody is writing to blocks inside the operating system's `open`,
  before any `timeout` in `DecodeLimits` can apply.
- **Change.** Open with `O_NONBLOCK` and clear the flag afterwards.
- **Breaks.** Clearing the flag takes `fcntl`, which means `unsafe` or a new
  dependency. The crate is `#![forbid(unsafe_code)]`.

### 6.2 Removing a reference from a large Wang or Panako index

- **Now.** `remove` scans the whole table to delete one reference (`remove_many`
  shares that scan across a batch, and this review made it about four times
  faster).
- **Change.** Keep each reference's own key list, as `HaitsmaIndex` does.
- **Breaks.** Nothing in the API, but memory per reference goes up by about
  four bytes per hash. It is listed here because it trades memory every user
  pays for latency only some need.
