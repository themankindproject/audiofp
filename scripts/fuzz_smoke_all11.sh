#!/usr/bin/env bash
# Bounded smoke run for every libFuzzer target in fuzz/Cargo.toml.
#
# Usage:
#   ./scripts/fuzz_smoke_all11.sh
#   MAX_RUNS=500 MAX_TOTAL_TIME=15 ./scripts/fuzz_smoke_all11.sh
#
# Exits non-zero when any target fails or when the target list drifts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MAX_RUNS="${MAX_RUNS:-1000}"
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-30}"
MAX_LEN="${MAX_LEN:-8192}"

TARGETS=(
  streaming_wang_equiv
  streaming_panako_equiv
  streaming_haitsma_equiv
  wang_hash_roundtrip
  panako_hash_roundtrip
  haitsma_hash_roundtrip
  sinc_resampler
  decode_bytes
  decode_resample
  serial_roundtrip
  matching_malformed
)

if [[ "${#TARGETS[@]}" -ne 11 ]]; then
  echo "expected 11 fuzz targets, found ${#TARGETS[@]}" >&2
  exit 1
fi

cd "$PROJECT_ROOT/fuzz"

for target in "${TARGETS[@]}"; do
  echo "=== Fuzzing $target (${MAX_RUNS} runs, ${MAX_TOTAL_TIME}s cap) ==="
  cargo +nightly fuzz run --sanitizer=none "$target" -- \
    -max_total_time="$MAX_TOTAL_TIME" \
    -runs="$MAX_RUNS" \
    -max_len="$MAX_LEN"
done

echo "all ${#TARGETS[@]} fuzz targets completed smoke run"
