"""Launch the official evaluator in its OWN environment, never reimplement it."""
import argparse
import subprocess
from pathlib import Path
from common import atomic_json, file_hash, read_json
import json
from collections import defaultdict
from statistics import mean
def load_results(path):
    results = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Path(row["filename"])
        key = (int(image.parent.parent.name), int(image.stem))

        if key in results:
            raise ValueError(f"Duplicate result: {key}")
        if not isinstance(row["correct"], bool):
            raise ValueError(f"Invalid correctness value: {key}")

        if isinstance(row["metadata"], str):
            row["metadata"] = json.loads(row["metadata"])
        results[key] = row
    return results


def summarize_subset(results):
    groups = defaultdict(list)
    for row in results.values():
        groups[row["tag"]].append(row["correct"])

    per_task = {tag: mean(values) for tag, values in groups.items()}
    return {
        "images": len(results),
        "per_task": per_task,
        "image_micro": mean(row["correct"] for row in results.values()),
        "macro_over_present_tasks": mean(per_task.values()),
    }


def compare_results(before, after):
    # Only originals whose edited versions were evaluated.
    before = {key: before[key] for key in after}

    for key in after:
        if before[key]["metadata"] != after[key]["metadata"]:
            raise ValueError(f"Before/after metadata differs: {key}")

    repaired = sum(
        not before[k]["correct"] and after[k]["correct"] for k in after
    )
    regressed = sum(
        before[k]["correct"] and not after[k]["correct"] for k in after
    )
    wrong_before = sum(not row["correct"] for row in before.values())
    right_before = len(before) - wrong_before
    b, a = summarize_subset(before), summarize_subset(after)

    return {
        "scope": "edited samples only",
        "baseline": b,
        "after": a,
        "paired": {
            "wrong_to_right": repaired,
            "right_to_wrong": regressed,
            "repair_rate": repaired / wrong_before if wrong_before else None,
            "regression_rate": regressed / right_before if right_before else None,
            "delta_micro_pp": 100 * (a["image_micro"] - b["image_micro"]),
            "delta_macro_present_tasks_pp": 100 * (
                a["macro_over_present_tasks"] - b["macro_over_present_tasks"]
            ),
        },
    }
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--geneval-repo', required=True)
    p.add_argument('--geneval-python', required=True, help='Absolute Python path in official GenEval environment')
    p.add_argument('--detector-dir', required=True)
    p.add_argument('--benchmark-source', default='data/geneval/source.json')
    p.add_argument('--out', required=True)
    p.add_argument(
    "--before-results",
    help="Original baseline JSONL; enables edited-subset evaluation",)
    a = p.parse_args()
    root = Path(a.run).resolve()
    repo = Path(a.geneval_repo).resolve()
    source = read_json(a.benchmark_source)
    if a.before_results:
        before = load_results(a.before_results)
        selected = {}

        for image in root.glob("*/samples/*.png"):
            key = (int(image.parent.parent.name), int(image.stem))
            if key in selected:
                raise ValueError(f"Duplicate edited image ID: {key}")

            metadata = read_json(image.parent.parent / "metadata.jsonl")
            if key not in before:
                raise ValueError(f"No baseline result for edited image: {key}")
            if before[key]["metadata"] != metadata:
                raise ValueError(f"Original metadata changed: {key}")

            selected[key] = metadata

        if not selected:
            raise ValueError(f"No edited images found in {root}")
    else:
        manifest = read_json(root / "manifest.json")
        if manifest["spec"]["metadata_sha256"] != source["sha256"]:
            raise ValueError("Run input does not match the pinned benchmark source.")

        audit = read_json(root / "audit.json")
        if not audit["complete"] or audit["scope"] == "smoke":
            raise ValueError("Run a successful full-input audit before evaluation.")
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != source['revision']:
        raise ValueError('Evaluator checkout must match the benchmark source revision.')
    if subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD', '--', 'evaluation'], text=True).strip():
        raise ValueError('Official evaluator has local modifications.')
    out = Path(a.out).resolve()
    if out.exists():
        raise ValueError('Choose a fresh results filename; do not overwrite evaluation evidence.')
    out.parent.mkdir(parents=True, exist_ok=True)
    python = str(Path(a.geneval_python).resolve())
    command = [python, str(repo/'evaluation/evaluate_images.py'), str(root), '--outfile', str(out),
               '--model-path', str(Path(a.detector_dir).resolve())]
    subprocess.run(command, cwd=repo, check=True)
    if a.before_results:
        after = load_results(out)

        if set(after) != set(selected):
            raise ValueError("Evaluation results do not cover exactly the edited images.")

        for key in after:
            if after[key]["metadata"] != selected[key]:
                raise ValueError(f"Evaluated metadata differs: {key}")

        report = compare_results(before, after)
        atomic_json(out.with_suffix(".paired.json"), report)
        print(json.dumps(report, indent=2))
    packages = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
    # Weights can be large; checksum once per evaluation to record exact evaluator artifacts.
    weights = {str(p.relative_to(a.detector_dir)): file_hash(p) for p in Path(a.detector_dir).rglob('*') if p.is_file()}
    atomic_json(out.with_suffix('.provenance.json'), {'command': command, 'geneval_commit': commit,
                'detector_files_sha256': weights, 'pip_freeze': packages,
                'results_sha256': file_hash(out),"manifest_sha256": (file_hash(root / "manifest.json") if not a.before_results else None),"baseline_results_sha256": (file_hash(a.before_results) if a.before_results else None),})

if __name__ == '__main__':
    main()
