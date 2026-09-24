#!/usr/bin/env bash
#
# Build the quantized LocateAnything-3B container that locate-anything-rs reads.
#
# Downloads NVIDIA's upstream checkpoint from Hugging Face and quantizes it with
# tools/prequant.py.  Usage, either from a checkout or straight off GitHub:
#
#   ./get-model.sh [OUT]                 # OUT defaults to ./locate-anything-allq8_0.laqt
#   curl -fsSL https://raw.githubusercontent.com/jacobsparts/locate-anything-rs/main/get-model.sh | bash
#   wget -qO- https://raw.githubusercontent.com/jacobsparts/locate-anything-rs/main/get-model.sh | bash
#
# The converter is three Python files that live in tools/ in the repository:
# prequant.py, and the laformat and namemap modules it imports.  When this
# script is piped into a shell there is no checkout to find them in, so they are
# fetched from GitHub at the same revision as the script.
#
# Needs curl, and Python 3 with numpy (python3 -m pip install numpy).
# The weights are non-commercial: see the upstream licence at
# https://huggingface.co/nvidia/LocateAnything-3B .

set -euo pipefail

REPO=${LA_REPO:-jacobsparts/locate-anything-rs}
REF=${LA_REF:-main}
RAW=${LA_RAW_BASE:-https://raw.githubusercontent.com/$REPO/$REF}

say() { printf '[get-model] %s\n' "$*" >&2; }
die() { printf '[get-model] error: %s\n' "$*" >&2; exit 1; }

# Where the script lives.  Piped in, $0 is the shell itself, which is not a
# file that carries the repository layout - hence the -f test rather than a
# bare dirname.
SELF=${BASH_SOURCE[0]:-}
SCRIPT_DIR=
if [[ -n $SELF && -f $SELF ]]; then
    SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$SELF")" && pwd)
fi

OUT=${1:-${LA_MODEL_OUT:-locate-anything-allq8_0.laqt}}
CACHE=${LA_MODEL_CACHE:-"${SCRIPT_DIR:-$PWD}/model-source"}
BASE=${LA_BASE:-https://huggingface.co/nvidia/LocateAnything-3B/resolve/main}

case $OUT in
    -h|--help)
        cat >&2 <<'USAGE'
Build the quantized LocateAnything-3B container that locate-anything-rs reads.

  get-model.sh [OUT]     OUT defaults to ./locate-anything-allq8_0.laqt
                         (or set LA_MODEL_OUT)

Downloads NVIDIA's upstream checkpoint into model-source/ (7.7 GB) and
quantizes it into the container (4.16 GB).  Needs curl, Python 3 and numpy.

  LA_MODEL_CACHE   where the 7.7 GB source download goes (default: model-source/)
  LA_REPO / LA_REF / LA_RAW_BASE
                   fetch the converter from somewhere else

The weights are non-commercial: see
https://huggingface.co/nvidia/LocateAnything-3B
USAGE
        exit 0
        ;;
esac

command -v curl >/dev/null || die "curl is required"
python3 -c 'import numpy' 2>/dev/null \
    || die "Python 3 and numpy are required (try: python3 -m pip install numpy)"

# The converter, and the modules it imports.  All three are fetched when the
# checkout is not there, into one directory, because prequant.py resolves its
# siblings relative to its own location.
TOOLS=(prequant.py laformat.py namemap.py)
TOOLS_DIR=$SCRIPT_DIR/tools
TOOLS_TMP=
if [[ -z $SCRIPT_DIR || ! -f $SCRIPT_DIR/tools/prequant.py ]]; then
    TOOLS_DIR=$(mktemp -d "${TMPDIR:-/tmp}/la-tools.XXXXXX")
    TOOLS_TMP=$TOOLS_DIR
fi

# Nothing half-written survives this script, whether it succeeds, fails or is
# interrupted: the fetched converter is removed, and so is any partially
# downloaded file.  A file is only ever moved into place once it is complete, so
# an interrupted run leaves no *.part (or anything else) behind for the next one
# to trip over - and the cache it fills is only ever complete files.
# shellcheck disable=SC2317  # invoked by the trap below
cleanup() {
    [[ -n $TOOLS_TMP ]] && rm -rf "$TOOLS_TMP"
    [[ -n ${CACHE:-} && -d ${CACHE:-} ]] && rm -f "$CACHE"/*.part "$OUT.part"
    return 0
}
# EXIT covers every normal and failing path; INT and TERM additionally have to
# exit, because installing a handler for them replaces bash's default abort.
on_signal() { cleanup; exit 130; }
trap cleanup EXIT
trap on_signal INT TERM

if [[ -n $TOOLS_TMP ]]; then
    for tool in "${TOOLS[@]}"; do
        curl -fsSL "$RAW/tools/$tool" -o "$TOOLS_TMP/$tool" \
            || die "could not fetch tools/$tool from $REPO@$REF"
    done
fi

mkdir -p "$CACHE"
files=(
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
    config.json
    preprocessor_config.json
    vocab.json
    added_tokens.json
    merges.txt
)
for file in "${files[@]}"; do
    if [[ -s "$CACHE/$file" ]]; then
        say "$file (already downloaded)"
        continue
    fi
    say "$file"
    curl --fail --location --retry 3 \
        --output "$CACHE/$file.part" "$BASE/$file" \
        || die "could not download $file from $BASE"
    mv "$CACHE/$file.part" "$CACHE/$file"
done

# The converter writes next to the final name and only moves it in on success,
# so a converter that dies part-way cannot leave a usable-looking container.
rm -f "$OUT.part"
python3 "$TOOLS_DIR/prequant.py" --model-dir "$CACHE" --out "$OUT.part" \
    || die "the converter failed; no container was written to $OUT"
mv "$OUT.part" "$OUT"

say "ready: $OUT"
say "the downloaded source checkpoint remains in $CACHE; remove it to reclaim 7.7 GB"
