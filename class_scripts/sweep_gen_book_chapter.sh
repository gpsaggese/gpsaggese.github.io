#!/usr/bin/env bash
# Run `gen_book_chapter.py` over several --llm_backend / --model combos and
# save each run's stdout+stderr and generated chapter under a tag that
# encodes the exact command line used, so results can be compared side by
# side.
#
# Usage:
#   class_scripts/sweep_gen_book_chapter.sh [lesson] [mode] [out_dir]
#
# Examples:
#   # Default: msml610/01.2, typst_aima mode.
#   class_scripts/sweep_gen_book_chapter.sh
#
#   # Custom lesson / mode.
#   class_scripts/sweep_gen_book_chapter.sh msml610/08.1 md
#
# Edit the COMBOS array below to add/remove (backend, model) pairs to sweep.
set -euo pipefail

LESSON="${1:-msml610/01.2}"
MODE="${2:-typst_aima}"
OUT_DIR="${3:-./sweep_results}"

# Each entry is "llm_backend:model". Leave model empty to use the
# backend's default model (e.g., "hllm_cli_lib:").
# openrouter/anthropic/claude-opus-4.6
COMBOS=(
    "hllm_cli_exec:gpt-4o"
    "hllm_cli_exec:openrouter/anthropic/claude-opus-4.6"
    "hllm:"
    "hllm:gpt-4o"
)

mkdir -p "$OUT_DIR"

# Slugify a lesson spec like "msml610/01.2" -> "msml610_01.2".
lesson_slug=$(echo "$LESSON" | tr '/' '_')

# Extension must match --mode (gen_book_chapter.py picks the compiler/linter
# based on the --output file extension).
case "$MODE" in
    springer_latex) ext="tex" ;;
    typst_aima) ext="typ" ;;
    md) ext="md" ;;
    *)
        echo "Unknown mode '$MODE'" >&2
        exit 1
        ;;
esac

echo "Sweeping gen_book_chapter.py over ${#COMBOS[@]} backend/model combos"
echo "lesson=$LESSON mode=$MODE out_dir=$OUT_DIR"
echo

for combo in "${COMBOS[@]}"; do
    backend="${combo%%:*}"
    model="${combo#*:}"

    # Build a filesystem-safe tag, e.g. "hllm_cli__gpt-4o" or
    # "hllm_cli__default".
    model_tag="${model:-default}"
    model_tag=$(echo "$model_tag" | tr '/' '_')
    tag="${lesson_slug}__${MODE}__${backend}__${model_tag}"

    cmd=(python class_scripts/gen_book_chapter.py \
        --input "$LESSON" \
        --mode "$MODE" \
        --llm_backend "$backend" \
        --no_incremental \
        --no_report_command_line)
    if [ -n "$model" ]; then
        cmd+=(--model "$model")
    fi
    # Write each run's chapter to its own tagged file so runs don't clobber
    # each other or the "real" book/ output.
    cmd+=(--output "${OUT_DIR}/${tag}.${ext}")

    log_file="${OUT_DIR}/${tag}.log"

    echo "################################################################################"
    echo "# $tag"
    echo "# ${cmd[*]}"
    echo "################################################################################"
    {
        echo "# Command line: ${cmd[*]}"
        echo "# Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo
    } > "$log_file"
    if "${cmd[@]}" >> "$log_file" 2>&1; then
        echo "OK   -> $log_file"
    else
        echo "FAIL -> $log_file"
    fi
    echo
done

echo "Done. Results in $OUT_DIR"
