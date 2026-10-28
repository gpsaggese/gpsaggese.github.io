"""Validate full coverage/checksums and summarize measured baseline costs."""
import argparse
import statistics
from pathlib import Path
from common import atomic_json, complete_sample, file_hash, read_json, rows, sample_paths, sample_seed

def audit(root, metadata):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    spec = manifest['spec']
    if file_hash(metadata) != spec['metadata_sha256']:
        raise ValueError('Metadata differs from generation input.')
    selected = list(enumerate(rows(metadata)))[:spec['limit']]
    records, missing = [], []
    for i, m in selected:
        folder, _, _ = sample_paths(root, i, 0)
        if (folder / 'metadata.jsonl').exists() and read_json(folder / 'metadata.jsonl') != m:
            raise ValueError(f'Metadata mismatch at prompt {i}')
        for j in range(spec['samples']):
            if not complete_sample(root, i, j, manifest['signature'], spec['save_latents']):
                missing.append([i, j])
                continue
            _, image, record = sample_paths(root, i, j)
            r = read_json(record)
            if r['seed'] != sample_seed(i, j, spec['seed']):
                raise ValueError('Incorrect seed record')
            from PIL import Image
            with Image.open(image) as im:
                if im.size != (spec['config']['width'], spec['config']['height']):
                    raise ValueError(f'Wrong image dimensions: {image}')
                im.verify()
            records.append(r)
    expected_paths = {f'{i:05d}/samples/{j:04d}.png' for i, _ in selected for j in range(spec['samples'])}
    actual_paths = {str(p.relative_to(root)) for p in root.glob('[0-9]*/samples/*.png')}
    extras = sorted(actual_paths - expected_paths)
    times = sorted(r['pipeline_seconds'] for r in records)
    summary = {'complete': not missing and not extras, 'expected': len(selected)*spec['samples'],
               'completed': len(records), 'missing_or_corrupt': missing, 'unexpected_images': extras,
               'scope': spec['scope'], 'model': spec['model'], 'offload': spec['offload'],
               'hardware': spec.get('environment', {}).get('gpu'),
               'p50_seconds': statistics.median(times) if times else None,
               'p95_seconds': times[min(len(times)-1, int(0.95*len(times)))] if times else None,
               'total_pipeline_seconds': sum(times),
               'max_peak_allocated_GiB': max((r['peak_allocated_bytes'] for r in records), default=0)/2**30,
               'max_peak_reserved_GiB': max((r['peak_reserved_bytes'] for r in records), default=0)/2**30}
    atomic_json(root / 'audit.json', summary)
    return summary

if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--metadata', required=True)
    a = p.parse_args()
    s = audit(a.run, a.metadata)
    print(json.dumps(s, indent=2))
    raise SystemExit(0 if s['complete'] else 1)
