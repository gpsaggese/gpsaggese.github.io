#!/bin/bash
# """
# Run the coding-benchmark experiment: generate the answers of every agent on
# HumanEval+ and MBPP+ with Ollama, then score them with EvalPlus.
#
# The 3 models run 1 greedy run and then 5 runs at temperature 0.8, added 1
# run at a time for all the models, so a stopped job leaves the same number
# of runs for every agent. The 2 scaffolds come from the same cells. The
# generation skips the cells that are done, so running this script again
# continues where it stopped. The public EvalPlus samples are scored too.
#
# Every scoring uses 4 workers, since the time limits of the tests depend on
# how busy the CPU is.
#
# Keep the Mac awake and plugged in:
# > caffeinate -dimsu ./run_coding_experiment.sh
# """

# Exit immediately if any command exits with a non-zero status.
set -e

MODELS="deepseek-coder:1.3b-instruct-q4_K_M codeqwen:7b-chat-v1.5-q4_K_M opencoder:8b"
BENCHMARKS="humaneval mbpp"
SAMPLED_RUNS=5

generate() {
    # """
    # Generate the missing cells of 1 model on 1 benchmark.
    #
    # :param model: Ollama model
    # :param benchmark: `humaneval` or `mbpp`
    # :param temperature: sampling temperature
    # :param runs: runs per task to reach
    # """
    local model=$1
    local benchmark=$2
    local temperature=$3
    local runs=$4
    echo "# $(date +%H:%M:%S) $model $benchmark T=$temperature runs=$runs"
    ./docker_cmd_sandbox.sh --ollama "python run_coding_agents.py \
        --benchmark $benchmark --model $model --temperature $temperature \
        --runs $runs"
}

# Greedy run of every model first, to compare with the published scores early.
for model in $MODELS; do
    for benchmark in $BENCHMARKS; do
        generate $model $benchmark 0 1
    done
done

# Sampled runs, 1 more run per pass for every model.
for runs in $(seq 1 $SAMPLED_RUNS); do
    for model in $MODELS; do
        for benchmark in $BENCHMARKS; do
            generate $model $benchmark 0.8 $runs
        done
    done
done

# Score every answer twice, to find flaky tests.
for runs_file in data/raw/coding_runs/*/*.jsonl; do
    echo "# $(date +%H:%M:%S) scoring $runs_file"
    ./docker_cmd_sandbox.sh "python score_coding_runs.py \
        --runs_file $runs_file --n_checks 2 --num_workers 4"
done

# Score the public EvalPlus samples: all the greedy ones, and the first 50
# samples per task of the sampled ones.
for zip_file in data/raw/evalplus_samples/*/*.zip; do
    echo "# $(date +%H:%M:%S) scoring $zip_file"
    ./docker_cmd_sandbox.sh "python score_coding_runs.py \
        --zip_file $zip_file --max_samples 50 --num_workers 4"
done
