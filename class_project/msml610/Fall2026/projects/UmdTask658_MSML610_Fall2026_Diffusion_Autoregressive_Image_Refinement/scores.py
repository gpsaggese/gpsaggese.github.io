"""Summarize official GenEval JSONL, with strict coverage and paired prompt bootstrap.

No detector substitute: these results MUST come from official evaluate_images.py.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from common import TAGS, atomic_json, canonical, rows

def load_scores(path):
    output = {}
    for r in rows(path):
        f = Path(r['filename'])
        if f.parent.name != 'samples' or not f.parent.parent.name.isdigit() or not f.stem.isdigit():
            raise ValueError(f'Unexpected image path: {f}')
        key = (int(f.parent.parent.name), int(f.stem))
        if key in output:
            raise ValueError(f'Duplicate result: {key}')
        if not isinstance(r['correct'], bool):
            raise ValueError('correct must be a JSON boolean')
        meta = json.loads(r['metadata']) if isinstance(r['metadata'], str) else r['metadata']
        if r['tag'] != meta['tag'] or r['prompt'] != meta['prompt']:
            raise ValueError('Inconsistent evaluator record')
        output[key] = {'correct': int(r['correct']), 'tag': r['tag'], 'metadata': meta}
    if not output:
        raise ValueError('No evaluation results')
    return output

def validate_coverage(results, metadata, samples):
    expected = {(i, j) for i in range(len(metadata)) for j in range(samples)}
    if set(results) != expected:
        raise ValueError(f'Coverage mismatch: expected {len(expected)}, got {len(results)}; missing={len(expected-set(results))}, extra={len(set(results)-expected)}')
    for (i, _), r in results.items():
        if r['metadata'] != metadata[i]:
            raise ValueError(f'Prompt {i} metadata mismatch')
    if {r['tag'] for r in results.values()} != TAGS:
        raise ValueError('All six GenEval categories required; do not report smoke subsets.')

def summarize(results):
    groups = defaultdict(list)
    for r in results.values():
        groups[r['tag']].append(r['correct'])
    by_task = {tag: statistics.mean(vals) for tag, vals in sorted(groups.items())}
    return {'images': len(results), 'per_task': by_task,
            'overall_macro': statistics.mean(by_task.values()),
            'image_micro': statistics.mean(r['correct'] for r in results.values())}

def paired(before, after, iterations=2000):
    if set(before) != set(after):
        raise ValueError('Before/after image sets differ; paired comparison invalid.')
    prompt_delta = defaultdict(list)
    changed = {'wrong_to_right': 0, 'right_to_wrong': 0, 'wrong_before': 0, 'right_before': 0}
    for key, b in before.items():
        a = after[key]
        if a['metadata'] != b['metadata'] or a['tag'] != b['tag']:
            raise ValueError('Before/after prompt metadata differs.')
        prompt_delta[(b['tag'], key[0])].append(a['correct'] - b['correct'])
        changed['wrong_before'] += b['correct'] == 0
        changed['right_before'] += b['correct'] == 1
        changed['wrong_to_right'] += b['correct'] == 0 and a['correct'] == 1
        changed['right_to_wrong'] += b['correct'] == 1 and a['correct'] == 0
    groups = defaultdict(list)
    for (tag, _), ds in prompt_delta.items():
        groups[tag].append(statistics.mean(ds))
    rng = random.Random(20260922)
    distribution = []
    for _ in range(iterations):
        distribution.append(statistics.mean(statistics.mean(rng.choices(v, k=len(v))) for v in groups.values()))
    distribution.sort()
    return {**changed,
            'repair_rate': changed['wrong_to_right']/changed['wrong_before'] if changed['wrong_before'] else None,
            'damage_rate': changed['right_to_wrong']/changed['right_before'] if changed['right_before'] else None,
            'delta_macro': summarize(after)['overall_macro'] - summarize(before)['overall_macro'],
            'delta_macro_95ci': [distribution[int(.025*iterations)], distribution[min(iterations-1, int(.975*iterations))]],
            'bootstrap': 'stratified by category; paired resampling of prompts with all image seeds retained'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--after')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    base = load_scores(a.results)
    meta = rows(a.metadata)
    validate_coverage(base, meta, a.samples)
    report = {'baseline': summarize(base)}
    if a.after:
        after = load_scores(a.after)
        validate_coverage(after, meta, a.samples)
        report.update(after=summarize(after), paired=paired(base, after))
    atomic_json(a.out, report)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
