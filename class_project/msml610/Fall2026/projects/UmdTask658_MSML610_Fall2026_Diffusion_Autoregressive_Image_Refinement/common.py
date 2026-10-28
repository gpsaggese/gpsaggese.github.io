"""Shared manifest and integrity helpers; standard library only."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAGS = {'single_object', 'two_object', 'counting', 'colors', 'position', 'color_attr'}

def read_json(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)

def sample_seed(prompt_id, sample_id, base_seed=20260922):
    # Stable across models/processes, independent of Python's randomized hash().
    return int(digest([base_seed, prompt_id, sample_id])[:8], 16)

def sample_paths(root, prompt_id, sample_id):
    folder = Path(root) / f'{prompt_id:05d}'
    return folder, folder / 'samples' / f'{sample_id:04d}.png', folder / 'records' / f'{sample_id:04d}.json'

def complete_sample(root, prompt_id, sample_id, signature, save_latents):
    _, image, record = sample_paths(root, prompt_id, sample_id)
    if not (image.exists() and record.exists()):
        return False
    r = read_json(record)
    if r['signature'] != signature:
        raise ValueError('Sample belongs to a different run configuration')
    if file_hash(image) != r['image_sha256']:
        return False
    if save_latents:
        latent = Path(root) / r['latent_file']
        if not latent.exists() or file_hash(latent) != r['latent_sha256']:
            return False
    return True
