#!/bin/bash
# """
# Run the search over prompt variants of Milestone 4: 1 greedy one-shot answer
# of DeepSeek Coder 1.3B to every task of HumanEval+ and MBPP+ with each of
# the 20 prompt variants in `PROMPT_VARIANTS`, then score the answers.
#
# HumanEval+ goes first for all the variants, so a stopped job leaves complete
# benchmarks. The generation skips the cells that are done, so running the
# script again continues where it stopped.
#
# Keep the Mac awake and plugged in, and generate the answers:
# > caffeinate -dimsu ./run_prompt_variants.sh generate
#
# Then score them, with 4 workers as for the other runs:
# > ./run_prompt_variants.sh score
# """

# Exit immediately if any command exits with a non-zero status.
set -e

MODEL="deepseek-coder:1.3b-instruct-q4_K_M"
BENCHMARKS="humaneval mbpp"
# The 2 x 2 x 5 grid of `PROMPT_VARIANTS` in `skill_vs_luck_runs_utils.py`.
SYSTEMS="none expert"
INSTRUCTIONS="evalplus short"
GUIDANCES="none edge examples clean signature"
RUNS_DIR="data/raw/coding_runs_variants"
SCORES_DIR="data/coding_runs_variants"

variants() {
    # """
    # Print the names of the 20 prompt variants.
    # """
    for system in $SYSTEMS; do
        for instruction in $INSTRUCTIONS; do
            for guidance in $GUIDANCES; do
                echo "$system-$instruction-$guidance"
            done
        done
    done
}

generate() {
    # """
    # Generate the missing answers of every variant on both benchmarks.
    # """
    for benchmark in $BENCHMARKS; do
        for variant in $(variants); do
            echo "# $(date +%H:%M:%S) $benchmark $variant"
            ./docker_cmd_sandbox.sh --ollama "python run_coding_agents.py \
                --benchmark $benchmark --model $MODEL --temperature 0 \
                --runs 1 --max_repairs 0 --variant $variant \
                --dst_dir $RUNS_DIR"
        done
    done
}

score() {
    # """
    # Score the answers of every variant twice, to find flaky tests.
    # """
    for runs_file in "$RUNS_DIR"/*/*.jsonl; do
        echo "# $(date +%H:%M:%S) scoring $runs_file"
        ./docker_cmd_sandbox.sh "python score_coding_runs.py \
            --runs_file $runs_file --n_checks 2 --num_workers 4 \
            --dst_dir $SCORES_DIR"
    done
}

case "$1" in
    generate)
        generate
        ;;
    score)
        score
        ;;
    *)
        echo "Usage: $0 generate|score"
        exit 1
        ;;
esac
