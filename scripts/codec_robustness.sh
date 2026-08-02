#!/usr/bin/env bash
# scripts/codec_robustness.sh — Run codec robustness tests and format results.
#
# Usage:
#   ./scripts/codec_robustness.sh          # default: run all codec tests
#   ./scripts/codec_robustness.sh --quick  # codec_roundtrip only
#
# Requires: cargo, Rust toolchain, tests/assets/ populated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Color helpers ─────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[info]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ── Verify corpus ────────────────────────────────────────────────────────

info "Checking test corpus in tests/assets/ ..."

REQUIRED_FILES=(
    "tests/assets/galway.flac"
    "tests/assets/galway.mp3"
    "tests/assets/galway.ogg"
    "tests/assets/galway.m4a"
    "tests/assets/galway.wav"
    "tests/assets/galway.aiff"
    "tests/assets/galway_stereo.mp3"
    "tests/assets/galway_stereo.flac"
    "tests/assets/freak.flac"
    "tests/assets/freak.mp3"
    "tests/assets/freak.ogg"
    "tests/assets/freak.m4a"
    "tests/assets/freak_8000hz.mp3"
    "tests/assets/freak_11025hz.mp3"
    "tests/assets/freak_16000hz.mp3"
    "tests/assets/freak_22050hz.mp3"
    "tests/assets/freak_32000hz.mp3"
    "tests/assets/freak_44100hz.mp3"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$PROJECT_ROOT/$f" ]]; then
        fail "Missing: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo ""
    fail "$MISSING required test asset(s) missing."
    echo "  See tests/assets/CREDITS.md and ROBUSTNESS.md for how to obtain them."
    exit 1
fi

ok "All ${#REQUIRED_FILES[@]} required corpus files present."
echo ""

# ── Determine test scope ─────────────────────────────────────────────────

TESTS="--test codec_roundtrip --test codec_extended"
LABEL="codec_roundtrip + codec_extended"

if [[ "${1:-}" == "--quick" ]]; then
    TESTS="--test codec_roundtrip"
    LABEL="codec_roundtrip (quick mode)"
fi

# ── Run tests ────────────────────────────────────────────────────────────

info "Running: cargo test $TESTS --all-features -- --nocapture"
echo ""

cd "$PROJECT_ROOT"

# Capture both stdout and stderr (eprintln! goes to stderr)
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

if cargo test $TESTS --all-features -- --nocapture 2>"$TMPFILE"; then
    echo ""
    ok "All tests passed."
else
    echo ""
    fail "Some tests failed. See output above."
fi

# ── Format results table ─────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Codec Robustness Results ($LABEL)${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Extract Jaccard and bit-sim lines from test stderr output
RESULTS=$(grep -E "(Jaccard|bit-sim)" "$TMPFILE" 2>/dev/null || true)

if [[ -z "$RESULTS" ]]; then
    echo "  (No overlap metrics found in test output.)"
    echo "  Ensure tests are run with --nocapture to see eprintln! output."
else
    # Print header
    printf "  %-40s %s\n" "Test" "Score"
    printf "  %-40s %s\n" "────────────────────────────────────────" "─────────"

    # Parse and print each result line
    echo "$RESULTS" | while IFS= read -r line; do
        # Strip leading whitespace and "thread '...' " prefix if present
        clean=$(echo "$line" | sed 's/^[[:space:]]*//')
        printf "  %s\n" "$clean"
    done
fi

echo ""
echo -e "${BOLD}───────────────────────────────────────────────────────────────${NC}"
echo ""
echo "Published thresholds (ROBUSTNESS.md):"
echo "  Wang    ≥ 0.25 Jaccard   (lossy codecs)"
echo "  Panako  ≥ 0.20 Jaccard   (lossy codecs)"
echo "  Haitsma ≥ 0.75 bit-sim   (lossy codecs)"
echo "  Cross-track    < 0.05    (different songs)"
echo ""
echo "Full methodology: ROBUSTNESS.md"
echo "Raw test files:   tests/codec_roundtrip.rs, tests/codec_extended.rs"
