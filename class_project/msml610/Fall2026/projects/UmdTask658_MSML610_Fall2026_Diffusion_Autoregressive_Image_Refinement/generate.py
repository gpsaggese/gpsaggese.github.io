"""Phase 1: frozen baselines, native latents, GenEval output, measured CUDA cost."""
import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from common import ROOT, atomic_json, complete_sample, digest, file_hash, read_json, rows, sample_paths, sample_seed

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['sd15', 'sdxl', 'sd35', 'flux'])
    p.add_argument('--lock', default='model_lock.json')
    p.add_argument('--metadata', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--limit', type=int, help='Smoke only: first N prompts, NOT a benchmark score')
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--offload', choices=['none', 'model', 'sequential'], default='none')
    p.add_argument('--save-latents', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.samples < 1 or (a.limit is not None and a.limit < 1):
        p.error('Samples and limit must be positive.')
    return a

def run(a):
    metadata = rows(a.metadata)
    if not metadata or any('prompt' not in m or 'tag' not in m for m in metadata):
        raise ValueError('Invalid metadata')
    selected = list(enumerate(metadata))[:a.limit]
    lock = Path(a.lock)
    if lock.exists():
        config = read_json(lock)[a.model]
    elif a.dry_run:
        config = read_json(ROOT / 'configs/models.json')[a.model]
    else:
        raise ValueError('Run prepare.py models first to pin checkpoint revisions.')
    spec = {'model': a.model, 'config': config, 'metadata_sha256': file_hash(a.metadata),
            'samples': a.samples, 'limit': a.limit, 'seed': a.seed, 'offload': a.offload,
            'save_latents': a.save_latents, 'batch_size': 1,
            'scope': 'smoke' if a.limit else 'full-input', 'code_sha256': file_hash(__file__),
            'common_sha256': file_hash(ROOT / 'common.py')}
    if a.dry_run:
        print(json.dumps({'configuration': spec, 'images': len(selected)*a.samples,
                          'first_seed': sample_seed(0, 0, a.seed)}, indent=2))
        return
    import torch
    import diffusers
    from safetensors.torch import save_file
    if not torch.cuda.is_available():
        raise RuntimeError('This research runner requires CUDA for comparable latency measurements.')
    dtype = getattr(torch, config['dtype'])
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported. Do not silently change precision; use suitable hardware.')
    env = {'python': sys.version, 'platform': platform.platform(), 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(0), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
           'packages': {name: importlib.metadata.version(name) for name in
                        ['torch', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'huggingface-hub']}}
    spec['environment'] = env
    signature = digest(spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path)['signature'] != signature:
            raise ValueError('Run configuration/environment changed. Choose a new output directory.')
    else:
        if any(out.iterdir()):
            raise ValueError('Nonempty directory without a manifest; choose a clean output directory.')
        atomic_json(manifest_path, {'signature': signature, 'spec': spec})
    pending = [(i, m, j) for i, m in selected for j in range(a.samples)
               if not complete_sample(out, i, j, signature, a.save_latents)]
    if not pending:
        print('All samples already complete and checksummed.')
        return
    pipeline_type = getattr(diffusers, config['pipeline'])
    load_start = time.perf_counter()
    pipe = pipeline_type.from_pretrained(config['repo'], revision=config['revision'],
                 torch_dtype=dtype, use_safetensors=True)
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.requires_grad_(False).eval()
    if a.offload == 'none':
        pipe.to('cuda')
    elif a.offload == 'model':
        pipe.enable_model_cpu_offload()
    else:
        pipe.enable_sequential_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_start
    components = {name: sum(p.numel() for p in value.parameters())
                  for name, value in pipe.components.items() if isinstance(value, torch.nn.Module)}
    kwargs = dict(height=config['height'], width=config['width'],
                  num_inference_steps=config['steps'], guidance_scale=config['guidance'])
    # Keep all default model components, text encoders, safety checks and scheduler.
    capture = {}
    def callback(pipeline, step, timestep, values):
        if step == len(pipeline.scheduler.timesteps) - 1:
            capture['latent'] = values['latents'].detach()
        return values
    callbacks = {'callback_on_step_end': callback, 'callback_on_step_end_tensor_inputs': ['latents']} if a.save_latents else {}
    with torch.inference_mode():
        # Same shape and full step count; excluded from per-image measurements.
        warm = pipe(prompt='a photo of a red cup', generator=torch.Generator('cpu').manual_seed(0), **kwargs)
        del warm
        torch.cuda.synchronize()
        scheduler = dict(pipe.scheduler.config)
        runtime_path = out / 'runtime.json'
        if not runtime_path.exists():
            atomic_json(runtime_path, {'load_seconds': load_seconds, 'scheduler_class': type(pipe.scheduler).__name__,
                   'scheduler_config': json.loads(json.dumps(scheduler, default=str)),
                   'component_parameters': components, 'vae_config': dict(pipe.vae.config),
                   'timing': 'CUDA-synchronized pipeline call incl. text encoding, denoising, VAE decode and postprocess; excludes disk writes and warmup'})
        for i, m, j in pending:
            folder, image_path, record_path = sample_paths(out, i, j)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(folder / 'metadata.jsonl', m)
            capture.clear()
            seed = sample_seed(i, j, a.seed)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            result = pipe(prompt=m['prompt'], generator=torch.Generator('cpu').manual_seed(seed), **kwargs, **callbacks)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            record = {'signature': signature, 'prompt_id': i, 'sample_id': j, 'seed': seed, 'tag': m['tag'],
                      'pipeline_seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'image_file': str(image_path.relative_to(out)),
                      'nsfw_content_detected': getattr(result, 'nsfw_content_detected', None)}
            tmp = image_path.with_suffix('.tmp.png')
            result.images[0].save(tmp)
            os.replace(tmp, image_path)
            record['image_sha256'] = file_hash(image_path)
            if a.save_latents:
                if 'latent' not in capture:
                    raise RuntimeError('Final latent callback did not fire; sample is not complete.')
                latent = capture.pop('latent').cpu().contiguous()
                latent_path = folder / 'latents' / f'{j:04d}.safetensors'
                latent_path.parent.mkdir(exist_ok=True)
                temp_latent = latent_path.with_suffix('.tmp')
                save_file({'latent': latent}, str(temp_latent), metadata={'layout': config['latent_layout'], 'model': a.model})
                os.replace(temp_latent, latent_path)
                record.update(latent_file=str(latent_path.relative_to(out)), latent_sha256=file_hash(latent_path),
                              latent_shape=list(latent.shape), latent_dtype=str(latent.dtype), latent_layout=config['latent_layout'])
                del latent
            atomic_json(record_path, record)  # Completion marker written LAST.
            del result
            print(f'{a.model} prompt={i} sample={j} seconds={elapsed:.3f}', flush=True)

if __name__ == '__main__':
    run(arguments())
