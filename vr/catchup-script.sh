#!/usr/bin/env bash
set -euo pipefail

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)

usage() {
    cat <<'HELP'
Usage:
  catchup-script.sh prepare [SESSION_DIR]
  catchup-script.sh diff SESSION_DIR
  catchup-script.sh merge SESSION_DIR
  catchup-script.sh bench SESSION_DIR MODEL [BINARY] [--run-approved]

prepare: fetch official master, pin SHAs, create upstream and MX worktrees,
         copy the handoff, and save diff reports. Default session: /tmp/mx-catchup.*
diff:    refresh reports, including conflicts and the current upstream overlay.
merge:   start an uncommitted upstream merge in SESSION_DIR/port; never resolve it.
bench:   print a PP2048/TG128 command; run once only with --run-approved after
         explicit user approval. Default binary: original build-ilp/bin/llama-bench.
         Bound: 110 seconds plus 5 seconds for termination. No downloads.

Session directories must be new. Existing reports are refreshed; benchmark logs
are never overwritten. No build, test suite, commit, push, or patch export runs.
HELP
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

load_session() {
    SESSION=$(realpath -e "$1")
    SOURCE=$(cat "$SESSION/source-sha")
    UPSTREAM=$(cat "$SESSION/upstream-sha")
    SOURCE_REPO=$(cat "$SESSION/source-repo")
    PORT="$SESSION/port"
    git -C "$PORT" cat-file -e "$SOURCE^{commit}"
    git -C "$PORT" cat-file -e "$UPSTREAM^{commit}"
}

reports() {
    git -C "$PORT" status --short > "$SESSION/status.txt"
    git -C "$PORT" ls-files -u > "$SESSION/conflicts.txt"
    git -C "$PORT" log --oneline "$SOURCE..$UPSTREAM" > "$SESSION/upstream-log.txt"
    BASE=$(git -C "$PORT" merge-base "$SOURCE" "$UPSTREAM")
    git -C "$PORT" diff --binary "$BASE" "$SOURCE" > "$SESSION/old-overlay.patch"
    git -C "$PORT" diff --stat "$SOURCE...$UPSTREAM" > "$SESSION/upstream-stat.txt"
    git -C "$PORT" diff --binary "$SOURCE...$UPSTREAM" > "$SESSION/upstream.patch"
    git -C "$PORT" diff --stat "$UPSTREAM" > "$SESSION/overlay-stat.txt"
    git -C "$PORT" diff --binary "$UPSTREAM" > "$SESSION/overlay.patch"
    git -C "$PORT" diff --binary "$SOURCE" > "$SESSION/merge-changes.patch"
    printf 'Reports: %s\nPort: %s\nUpstream: %s\n' "$SESSION" "$PORT" "$UPSTREAM"
}

ACTION=${1:---help}
if (($#)); then shift; fi
case "$ACTION" in
    -h|--help) usage ;;
    prepare)
        (($# <= 1)) || fail 'prepare takes at most one session directory'
        test -z "$(git -C "$REPO" ls-files -u)" || fail 'resolve the current conflicts first'
        if git -C "$REPO" rev-parse -q --verify MERGE_HEAD >/dev/null; then
            fail 'finish the current merge first'
        fi
        git -C "$REPO" diff --quiet || fail 'tracked working-tree changes must be checkpointed first'
        git -C "$REPO" diff --cached --quiet || fail 'staged changes must be checkpointed first'
        SOURCE=$(git -C "$REPO" rev-parse HEAD)
        git -C "$REPO" fetch https://github.com/ggml-org/llama.cpp.git master
        UPSTREAM=$(git -C "$REPO" rev-parse FETCH_HEAD)
        git -C "$REPO" merge-base "$SOURCE" "$UPSTREAM" >/dev/null
        if (($#)); then
            test ! -e "$1" || fail 'session directory already exists'
            mkdir -- "$1"
            SESSION=$(realpath -e "$1")
        else
            SESSION=$(mktemp -d /tmp/mx-catchup.XXXXXX)
        fi
        printf '%s\n' "$SOURCE" > "$SESSION/source-sha"
        printf '%s\n' "$UPSTREAM" > "$SESSION/upstream-sha"
        printf '%s\n' "$REPO" > "$SESSION/source-repo"
        git -C "$REPO" show -s --format=fuller "$SOURCE" > "$SESSION/source.txt"
        git -C "$REPO" show -s --format=fuller "$UPSTREAM" > "$SESSION/upstream.txt"
        git -C "$REPO" worktree add --detach "$SESSION/upstream" "$UPSTREAM"
        git -C "$REPO" worktree add --detach "$SESSION/port" "$SOURCE"
        mkdir -p "$SESSION/port/vr"
        for name in how-to-catchup.md catchup-script.sh; do
            if test -f "$REPO/vr/$name"; then
                cp "$REPO/vr/$name" "$SESSION/port/vr/$name"
            fi
        done
        load_session "$SESSION"
        reports
        printf 'Next: preview the baseline with bench, then use merge when ready.\n'
        ;;
    diff|merge)
        (($# == 1)) || fail "$ACTION needs a session directory"
        load_session "$1"
        if test "$ACTION" = merge; then
            git -C "$PORT" diff --quiet || fail 'port has unstaged tracked changes'
            git -C "$PORT" diff --cached --quiet || fail 'port has staged changes'
            if git -C "$PORT" rev-parse -q --verify MERGE_HEAD >/dev/null; then
                fail 'merge already in progress; resolve conflicts, then run diff'
            fi
            merge_status=0
            git -C "$PORT" merge --no-commit --no-ff "$UPSTREAM" || merge_status=$?
            reports
            if ((merge_status)); then
                printf 'Merge stopped. Inspect conflicts.txt and resolve in %s.\n' "$PORT" >&2
                exit "$merge_status"
            fi
        else
            reports
        fi
        ;;
    bench)
        (($# >= 2 && $# <= 4)) || fail 'bench needs SESSION_DIR MODEL [BINARY] [--run-approved]'
        load_session "$1"
        MODEL=$(realpath -e "$2")
        test -f "$MODEL" || fail 'model must be a local file'
        shift 2
        BINARY="$SOURCE_REPO/build-ilp/bin/llama-bench"
        if (($#)) && test "$1" != --run-approved; then
            BINARY=$1
            shift
        fi
        APPROVED=false
        if (($#)); then
            test "$1" = --run-approved && (($# == 1)) || fail 'unknown bench argument'
            APPROVED=true
        fi
        BINARY=$(realpath -e "$BINARY")
        test -x "$BINARY" || fail 'benchmark binary must be executable'
        cmd=(env -u GGML_CUDA_TP_OVERLAP timeout --kill-after=5s 110s
            "$BINARY" -m "$MODEL" --offline -ngl 999 -sm tensor
            -dev ROCm0/ROCm1 -ts 1/1 -fa on -b 2048 -ub 2048
            -t 20 -p 2048 -n 128 -d 0 -r 1 -o json)
        printf 'Command (warmup enabled; one repetition per PP/TG test):\n'
        printf '%q ' "${cmd[@]}"
        printf '\n'
        if ! "$APPROVED"; then
            printf 'Preview only. Obtain explicit user OK before adding --run-approved.\n'
            exit 0
        fi
        OUT=$(mktemp -d "$SESSION/bench.XXXXXX")
        printf '%q ' "${cmd[@]}" > "$OUT/command.txt"
        printf '\n' >> "$OUT/command.txt"
        date -u +%FT%TZ > "$OUT/date.txt"
        printf '%s\n' "$MODEL" > "$OUT/model-path.txt"
        stat -- "$MODEL" > "$OUT/model-stat.txt"
        sha256sum -- "$BINARY" > "$OUT/binary-sha256.txt"
        ldd "$BINARY" > "$OUT/libraries.txt"
        env | LC_ALL=C sort | sed -n '/^\(GGML_\|LLAMA_\|HIP_\|ROCR_\|CUDA_\|HSA_\|LD_LIBRARY_PATH=\|PATH=\)/p' > "$OUT/environment.txt"
        BUILD_DIR=$(dirname "$(dirname "$BINARY")")
        if test -f "$BUILD_DIR/CMakeCache.txt"; then
            cp "$BUILD_DIR/CMakeCache.txt" "$OUT/CMakeCache.txt"
        fi
        git -C "$PORT" rev-parse HEAD > "$OUT/port-head.txt"
        git -C "$PORT" diff --binary HEAD > "$OUT/port-changes.patch"
        git -C "$SOURCE_REPO" rev-parse HEAD > "$OUT/source-head.txt"
        status=0
        "${cmd[@]}" > "$OUT/result.json" 2> "$OUT/stderr.log" || status=$?
        printf '%s\n' "$status" > "$OUT/exit-status.txt"
        printf 'Benchmark exit %s; logs: %s\n' "$status" "$OUT"
        if ((status == 0)); then cat "$OUT/result.json"; fi
        exit "$status"
        ;;
    *) usage >&2; exit 2 ;;
esac
