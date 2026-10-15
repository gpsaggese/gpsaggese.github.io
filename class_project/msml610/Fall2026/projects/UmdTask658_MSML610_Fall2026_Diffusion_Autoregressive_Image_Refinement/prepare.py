"""Fetch the exact official GenEval prompt set and create a model revision lock."""
import argparse
import json
import re
import urllib.request
from pathlib import Path
from common import ROOT, atomic_json, file_hash, read_json, rows, TAGS

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'diffusion-ar-study/1.0'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    g = s.add_parser('geneval')
    g.add_argument('--out', default='data/geneval')
    g.add_argument('--revision', help='Full git commit; default resolves main once and records it')
    m = s.add_parser('models')
    m.add_argument('--out', default='model_lock.json')
    m.add_argument('--models', nargs='+', default=['sd15', 'sdxl', 'sd35', 'flux'])
    a = p.parse_args()
    if a.command == 'geneval':
        out = Path(a.out)
        if out.exists() and any(out.iterdir()):
            raise SystemExit('Output directory is not empty; preserve existing benchmark lock.')
        rev = a.revision or json.loads(fetch('https://api.github.com/repos/djghosh13/geneval/commits/main'))['sha']
        if not re.fullmatch('[0-9a-f]{40}', rev):
            raise ValueError('Use a full 40-character git commit.')
        url = f'https://raw.githubusercontent.com/djghosh13/geneval/{rev}/prompts/evaluation_metadata.jsonl'
        data = fetch(url)
        parsed = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
        if len(parsed) != 553 or {x['tag'] for x in parsed} != TAGS:
            raise ValueError('Unexpected official benchmark layout; review before proceeding.')
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'evaluation_metadata.jsonl'
        path.write_bytes(data)
        atomic_json(out / 'source.json', {'repo': 'https://github.com/djghosh13/geneval', 'revision': rev,
                    'url': url, 'sha256': file_hash(path), 'prompts': len(parsed), 'split': 'official-test'})
        print(f'Saved {len(parsed)} official test prompts at {path}; revision {rev}')
    else:
        from huggingface_hub import HfApi
        if Path(a.out).exists():
            raise SystemExit('Lock already exists; use a new filename to create another experiment.')
        configs = read_json(ROOT / 'configs/models.json')
        api = HfApi()
        lock = {}
        for key in a.models:
            config = dict(configs[key])
            config['revision'] = api.model_info(config['repo']).sha
            lock[key] = config
        atomic_json(a.out, lock)
        print(f'Saved exact model revisions to {a.out}. Gated weight access is checked when loading.')

if __name__ == '__main__':
    main()
