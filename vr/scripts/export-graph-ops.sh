#!/usr/bin/env bash
# Export one exact model graph signature set and reuse it only when inputs match.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_llama_root.sh"
ROOT="$(resolve_llama_root)"
cd "$ROOT"

usage() {
    cat <<'EOF'
Usage:
  MODEL=/path/model.gguf vr/scripts/export-graph-ops.sh --dry-run
  MODEL=/path/model.gguf vr/scripts/export-graph-ops.sh

Environment:
  BUILD=build/head-control
  CONTEXT=8192
  BATCH=8192
  UBATCH=512
  OUT=/tmp/<model>-ub512-ops.txt

An existing OUT is reused only when its .meta file exactly matches the model
stat, exporter binary hash, and graph arguments.
EOF
}

ACTION=run
case "${1:-}" in
    "")
        ;;
    --dry-run)
        ACTION=dry-run
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown argument: $1" >&2
        exit 2
        ;;
esac

MODEL="${MODEL:-}"
BUILD="${BUILD:-$ROOT/build/head-control}"
CONTEXT="${CONTEXT:-8192}"
BATCH="${BATCH:-8192}"
UBATCH="${UBATCH:-512}"
[[ -n "$MODEL" ]] || {
    echo "Error: MODEL is required." >&2
    usage >&2
    exit 2
}

MODEL_TAG="$(basename "$MODEL" .gguf)"
OUT="${OUT:-/tmp/${MODEL_TAG}-c${CONTEXT}-b${BATCH}-ub${UBATCH}-ops.txt}"
EXPORTER="$(realpath -m "$BUILD")/bin/test-export-graph-ops"
META="$OUT.meta"

for value in "$CONTEXT" "$BATCH" "$UBATCH"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "Error: graph sizes must be positive integers." >&2
        exit 2
    }
done
(( UBATCH <= BATCH && BATCH <= CONTEXT )) || {
    echo "Error: require UBATCH <= BATCH <= CONTEXT." >&2
    exit 2
}

print_command() {
    printf 'output=%s\ncommand:' "$OUT"
    printf ' %q' "$EXPORTER" -m "$MODEL" -c "$CONTEXT" -b "$BATCH" \
        -ub "$UBATCH" -fa on -o "$OUT"
    printf '\n'
}

if [[ "$ACTION" == "dry-run" ]]; then
    print_command
    exit 0
fi

[[ -r "$MODEL" ]] || {
    echo "Error: unreadable model: $MODEL" >&2
    exit 2
}
[[ -x "$EXPORTER" ]] || {
    echo "Error: missing exporter: $EXPORTER" >&2
    exit 2
}

mkdir -p "$(dirname "$OUT")"
expected_meta="$META.expected.$$"
{
    stat -c 'model=%n,%s,%Y' "$MODEL"
    printf 'exporter=%s\n' "$EXPORTER"
    sha256sum "$EXPORTER"
    for library in libllama.so libggml.so libggml-hip.so; do
        [[ -r "$(dirname "$EXPORTER")/$library" ]] && sha256sum "$(dirname "$EXPORTER")/$library"
    done
    printf 'context=%s\nbatch=%s\nubatch=%s\nflash_attn=on\n' \
        "$CONTEXT" "$BATCH" "$UBATCH"
} > "$expected_meta"

if [[ -e "$OUT" || -e "$META" ]]; then
    if [[ -r "$OUT" && -r "$META" ]] && cmp -s "$expected_meta" "$META"; then
        rm -f "$expected_meta"
        echo "Reusing exact graph export: $OUT"
        exit 0
    fi
    rm -f "$expected_meta"
    echo "Error: existing graph export does not match current inputs: $OUT" >&2
    echo "Choose a new OUT or remove the stale export explicitly." >&2
    exit 2
fi

print_command
"$EXPORTER" -m "$MODEL" -c "$CONTEXT" -b "$BATCH" \
    -ub "$UBATCH" -fa on -o "$OUT"
mv "$expected_meta" "$META"
echo "Exported: $OUT"
echo "Note: signatures are de-duplicated and do not contain phase counts."
