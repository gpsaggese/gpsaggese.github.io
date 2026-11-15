"""Bounded, registered offline price-label BERT experiment; never a trading runner.

Each company-day is one target and one optimizer step. All distinct headlines
are averaged before MSE. Heldout features/outcomes open only after every company
finishes its sole epoch. This is retrospective development, not an alpha claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import math
from pathlib import Path
import random
import signal
import sys
import time

from grouped_collation import align_training_labels, collate_features, sample_ids
from grouped_loss import TrainingTargets
from price_transformer import (Budget, BudgetStop, bag_prediction, new_output,
                               parameter_hashes, partial_model, read, save, sha)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_bindings(repo, reg):
    """Hash-only access is allowed for heldout files; no parsing before fit."""
    for relative, digest in reg['bound_files_sha256'].items():
        if sha(repo / relative) != digest:
            raise ValueError('Registered artifact/source bytes changed: ' + relative)
    for name, version in reg['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Registered package version changed: ' + name)
    for name, item in reg['checkpoint']['files'].items():
        if sha(repo / reg['checkpoint']['local_directory'] / name) != item['sha256']:
            raise ValueError('Registered checkpoint asset changed: ' + name)
    if reg['companies'] != ['A', 'AMZN', 'MSFT'] or reg['training']['epochs'] != 1:
        raise ValueError('Unexpected company set or epoch policy')
    if reg['evaluation_kind'] != 'retrospective_development':
        raise ValueError('Historical receipt availability is not certified')


def validate_features(features, rows, expected, heldout_ids, tokenizer=None):
    if sample_ids(features) != expected or sample_ids(rows) != expected:
        raise ValueError('Missing, extra, or reordered company-day IDs')
    if any('target' in row for row in features):
        raise ValueError('Features must not carry targets')
    training = 'target' in rows[0]
    if training:
        align_training_labels(features, rows, heldout_ids)
    tokens = {}
    for feature, row in zip(features, rows):
        if not training and set(row) != {'sample_id', 'headlines'}:
            raise ValueError('Heldout feature packet contains outcomes')
        collate_features([feature])  # complete group, no truncation, valid masks
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in row['headlines']]
        if hashes != [h['input_sha256'] for h in feature['headlines']]:
            raise ValueError('Original headline identity or budget changed')
        for text, token in zip(row['headlines'], feature['headlines']):
            if tokenizer is not None:
                encoded = tokenizer.encode(text)
                if (encoded.ids != token['input_ids'] or encoded.type_ids != token['token_type_ids']
                        or encoded.attention_mask != token['attention_mask']):
                    raise ValueError('Official BERT tokenizer differs from prepared tokens')
            old = tokens.setdefault(token['input_sha256'], token)
            if old != token:
                raise ValueError('Conflicting tokens for one headline')
    return tokens


def check_updates(model, before):
    after = parameter_hashes(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    changed = {n for n in before if before[n] != after[n]}
    if changed - trainable:
        raise ValueError('Frozen parameters changed')
    if not any(n.startswith('backbone.encoder.layer.') for n in changed):
        raise ValueError('No encoder parameter update')
    if not any(n.startswith('head.') for n in changed):
        raise ValueError('No regression head update')
    return {'frozen_parameter_tensors_unchanged': len(before) - len(trainable),
            'trainable_parameter_tensors': len(trainable),
            'changed_trainable_parameter_tensors': len(changed),
            'before_sha256': before, 'after_sha256': after}


def fit_company(model, rows, tokens, heldout_ids, config, budget, seed, on_step):
    import torch
    targets = TrainingTargets(rows, heldout_ids)
    model.eval()
    before = parameter_hashes(model)
    selected = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(selected, lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    losses = []
    for row in ordered:
        budget.check()
        optimizer.zero_grad(set_to_none=True)
        prediction = bag_prediction(model, row['headlines'], tokens,
                                    config['headline_chunk_size'], budget)
        loss = (prediction - (row['target'] - targets.mean) / targets.scale) ** 2
        if not bool(torch.isfinite(loss)):
            raise BudgetStop('Nonfinite training loss')
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected):
            raise BudgetStop('Missing or nonfinite trainable gradient')
        torch.nn.utils.clip_grad_norm_(selected, config['gradient_clip_norm'])
        budget.check()
        optimizer.step()
        losses.append({'sample_id': row['sample_id'], 'normalized_training_loss': float(loss.detach())})
        on_step(losses[-1])
    budget.check()
    update = check_updates(model, before)
    return {'training_days': len(rows), 'optimizer_steps': len(losses),
            'training_mean': targets.mean, 'training_scale': targets.scale,
            'mean_normalized_training_loss': sum(x['normalized_training_loss'] for x in losses) / len(losses),
            'steps': losses, 'parameter_update_audit': update}


def metrics(truth, values, band=.005):
    if len(truth) != len(values) or not truth or not all(math.isfinite(x) for x in truth + values):
        raise ValueError('Require complete finite same-row predictions')
    def sign(x): return (x > 0) - (x < 0)
    def cls(x): return 1 if x > band else -1 if x < -band else 0
    y, p = list(map(cls, truth)), list(map(cls, values))
    recalls, f1 = [], []
    for c in (-1, 0, 1):
        tp = sum(a == c and b == c for a, b in zip(y, p))
        actual, predicted = y.count(c), p.count(c)
        if actual: recalls.append(tp / actual)
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0)
    n = len(truth)
    return {'samples': n, 'mae': sum(abs(a-b) for a,b in zip(truth,values))/n,
            'rmse': math.sqrt(sum((a-b)**2 for a,b in zip(truth,values))/n),
            'direction_accuracy': sum(sign(a)==sign(b) for a,b in zip(truth,values))/n,
            'price_class_balanced_accuracy': sum(recalls)/len(recalls),
            'price_class_macro_f1': sum(f1)/3}


def load_model(directory, seed):
    import torch
    from transformers import BertModel
    torch.manual_seed(seed)
    backbone, info = BertModel.from_pretrained(str(directory), local_files_only=True,
        use_safetensors=True, attn_implementation='eager', add_pooling_layer=False,
        output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Pretrained encoder was not completely restored')
    unused_pooler = {'bert.pooler.dense.bias', 'bert.pooler.dense.weight'}
    if any(not n.startswith('cls.') and n not in unused_pooler for n in info['unexpected_keys']):
        raise ValueError('Unexpected pretrained checkpoint architecture')
    if (backbone.config.num_hidden_layers, backbone.config.hidden_size) != (12, 768):
        raise ValueError('Checkpoint is not the registered BERT base architecture')
    return partial_model(backbone)


def public_summary(receipt, reg):
    """Export actual progress separately from the registered company-day pool.

    One optimizer step consumes one company-day in the registered single epoch.
    The legacy ``training_rows``/``heldout_rows`` fields now count actual training
    steps and completed evaluation rows. A stopped run must not inherit the full
    272/154 pool sizes. Older complete receipts retain their original counts via
    their recorded evaluation cases; an unknown planned heldout count stays null.
    No text, local paths, state tensors, or source IDs are exported.
    """
    cases=[]
    for case in receipt.get('evaluation_cases', []):
        costs=case['cost_evaluation']
        cases.append({'company':case['company'],'training_rows':case['training_rows'],
            'test_rows':case['test_rows'],'metrics':case['metrics'],
            'illustrative_cost_scenarios':{'mode':costs['mode'],
                'eligible_nonoverlapping_intervals':costs['eligible_intervals'],
                'models':costs['models'],'is_executable_strategy_evidence':False}})
    training_rows = receipt['actual_optimizer_steps']
    evaluated_rows = receipt.get('actual_evaluated_rows', sum(case['test_rows'] for case in cases))
    planned_training_rows = receipt.get('planned_training_rows', reg['training']['max_total_steps'])
    planned_heldout_rows = receipt.get('planned_heldout_rows')
    if planned_heldout_rows is None and receipt['status'] == 'complete':
        # Legacy completed receipts predate explicit prepared-pool counts.
        planned_heldout_rows = sum(case['test_rows'] for case in cases)
    return {'schema_version':1,'generated_at':stamp(),'status':receipt['status'],
        'evaluation_kind':'retrospective_development','model':'price_label_bert',
        'model_label':'Price-label BERT (final encoder layer + regression head)',
        'checkpoint_repository':reg['checkpoint']['repository'],
        'checkpoint_revision':reg['checkpoint']['revision'],
        'checkpoint_license':reg['checkpoint']['license'],
        'registration_sha256':receipt['registration_sha256'],
        'actual_optimizer_steps':receipt['actual_optimizer_steps'],
        'planned_optimizer_steps':reg['training']['max_total_steps'],
        'completed_companies':receipt['actual_finetuned_companies'],
        'heldout_predictions_completed':bool(receipt.get('heldout_prediction_sha256')),
        'heldout_outcomes_opened':receipt['heldout_outcomes_opened'],
        'training_rows':training_rows,'heldout_rows':evaluated_rows,
        'actual_training_rows':training_rows,'actual_evaluated_rows':evaluated_rows,
        'planned_training_rows':planned_training_rows,'planned_heldout_rows':planned_heldout_rows,
        'row_count_unit':'company_day','horizon_sessions':10,
        'test_start':'2023-09-01','target':'company minus SPY split-adjusted price return',
        'cases':cases,'promoted':False,'alpha_established':False,
        'failure_types':[f['type'] for f in receipt['failures']],
        'elapsed_seconds':receipt['elapsed_seconds'],
        'limitations':[
            'This is one fixed retrospective development run, not a fresh prospective test.',
            '2023 headlines were retrieved later; historical receipt availability is not verified.',
            'Only A, AMZN and MSFT are included; this does not establish general stock selection skill.',
            'Targets are 10-session SPY-excess split-adjusted price returns, not human sentiment labels.',
            'Cash dividends, realistic fills, borrow, portfolio constraints and taxes are excluded.',
            'Illustrative long/cash interval returns are not the live Quant.ai trading strategy or realized profit.',
            'Only the final BERT encoder layer and scalar head are updated; this is not full-model fine-tuning.',
            'Frozen FinBERT and previously saved controls are separate comparators; their parameters were not refitted.',
            'LM dictionary comparator has separate research-use and historical-version limitations.',
            'No hyperparameter search or deployment promotion was performed.']}


def run(repo, registration, output):
    import torch
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer

    repo, registration = Path(repo).resolve(), Path(registration).resolve()
    reg = read(registration)
    verify_bindings(repo, reg)
    out = new_output(output)
    started = time.monotonic()
    receipt = {'schema_version': 1, 'status': 'running', 'started_at': stamp(),
        'evaluation_kind': reg['evaluation_kind'], 'model': 'price_label_bert',
        'registration_sha256': sha(registration), 'actual_optimizer_steps': 0,
        'actual_finetuned_companies': [], 'heldout_features_opened': False,
        'heldout_outcomes_opened': False, 'companies': {}, 'promoted': False,
        'actual_evaluated_rows': 0,
        'alpha_established': False, 'strict_labels_created': 0, 'failures': []}
    resources = reg['resources']
    torch.set_num_threads(resources['threads'])
    torch.set_num_interop_threads(resources['interop_threads'])
    torch.use_deterministic_algorithms(True)
    budget = Budget(resources['train_seconds'], resources['max_rss_bytes'], resources['min_available_bytes'])
    previous_alarm = signal.getsignal(signal.SIGALRM)
    def outer_stop(signum, frame):
        raise BudgetStop('Registered outer wall-clock budget exhausted')
    signal.signal(signal.SIGALRM, outer_stop)
    signal.setitimer(signal.ITIMER_REAL, resources['outer_seconds'])
    try:
        manifest = read(repo / reg['prepared'] / 'preparation_receipt.json')
        receipt['planned_training_rows'] = sum(len(manifest['companies'][c]['train_ids']) for c in reg['companies'])
        receipt['planned_heldout_rows'] = sum(len(manifest['companies'][c]['test_ids']) for c in reg['companies'])
        training = read(repo / reg['prepared'] / 'training_packet.json')['companies']
        train_tokens = read(repo / reg['tokenized'] / 'training_tokens.json')['companies']
        heldout_ids = {s for c in manifest['companies'].values() for s in c['test_ids']}
        checkpoint = repo / reg['checkpoint']['local_directory']
        tokenizer = Tokenizer.from_file(str(checkpoint / 'tokenizer.json'))
        if list(training) != reg['companies'] or list(train_tokens) != reg['companies']:
            raise ValueError('Unexpected training company order')
        for index, company in enumerate(reg['companies']):
            tokens = validate_features(train_tokens[company], training[company],
                manifest['companies'][company]['train_ids'], heldout_ids, tokenizer)
            budget.check()
            model = load_model(checkpoint, reg['training']['seed'])
            def on_step(step):
                receipt['actual_optimizer_steps'] += 1
                receipt['active_company'] = company
                receipt['last_completed_sample_id'] = step['sample_id']
                receipt['elapsed_seconds'] = time.monotonic() - started
                save(out / 'training_receipt.json', receipt)
                if receipt['actual_optimizer_steps'] % 10 == 0:
                    print({'steps': receipt['actual_optimizer_steps'], 'company': company,
                           'seconds': round(receipt['elapsed_seconds'], 1)}, flush=True)
            result = fit_company(model, training[company], tokens, heldout_ids,
                reg['training'], budget, reg['training']['seed'] + index, on_step)
            state = {n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
            save_file(state, str(out / (company + '.trainable.safetensors')))
            result['trainable_state_sha256'] = sha(out / (company + '.trainable.safetensors'))
            receipt['companies'][company] = result
            receipt['actual_finetuned_companies'].append(company)
            save(out / 'training_receipt.json', receipt)
            del state, model, tokens
            gc.collect()
        if receipt['actual_optimizer_steps'] != reg['training']['max_total_steps']:
            raise ValueError('Incomplete registered epoch')
        receipt['training_complete_at'] = stamp()
        receipt['training_elapsed_seconds'] = time.monotonic() - started
        receipt['training_peak_observed_rss_bytes'] = budget.peak_observed_rss
        receipt['training_minimum_observed_available_bytes'] = budget.minimum_observed_available
        save(out / 'training_receipt.json', receipt)
        # First feature parse for the entire heldout set occurs after ALL fits.
        budget = Budget(resources['evaluation_seconds'], resources['max_rss_bytes'], resources['min_available_bytes'])
        heldout = read(repo / reg['prepared'] / 'heldout_text_packet.json')['companies']
        test_tokens = read(repo / reg['tokenized'] / 'heldout_tokens.json')['companies']
        receipt['heldout_features_opened'] = True
        predicted = {}
        for company in reg['companies']:
            tokens = validate_features(test_tokens[company], heldout[company],
                manifest['companies'][company]['test_ids'], heldout_ids, tokenizer)
            model = load_model(checkpoint, reg['training']['seed'])
            partial = load_file(str(out / (company + '.trainable.safetensors')))
            expected = {n for n,p in model.named_parameters() if p.requires_grad}
            if set(partial) != expected: raise ValueError('Incomplete trainable checkpoint')
            loaded = model.load_state_dict(partial, strict=False)
            expected_missing = set(model.state_dict()) - expected
            if loaded.unexpected_keys or set(loaded.missing_keys) != expected_missing:
                raise ValueError('Partial checkpoint state schema mismatch')
            trained = receipt['companies'][company]
            predictions = []
            with torch.inference_mode():
                for row in heldout[company]:
                    value = float(bag_prediction(model, row['headlines'], tokens,
                        reg['training']['headline_chunk_size'], budget))
                    value = value * trained['training_scale'] + trained['training_mean']
                    if not math.isfinite(value): raise ValueError('Nonfinite heldout prediction')
                    predictions.append({'sample_id':row['sample_id'], 'prediction':value})
            predicted[company] = predictions
            del model, partial, tokens
            gc.collect()
        save(out / 'heldout_predictions_without_outcomes.json', predicted)
        receipt['heldout_prediction_sha256'] = sha(out / 'heldout_predictions_without_outcomes.json')
        # Predictions are frozen on disk before the first heldout outcome parse.
        from run_historical import cost_evaluation
        cases = []
        for company in reg['companies']:
            budget.check()
            relative = f'{company}/h10/horizon_specific/excess_price_return'
            baseline = read(repo / reg['baseline_evaluation'] / relative / 'predictions.json')
            receipt['heldout_outcomes_opened'] = True
            if [r['sample_id'] for r in baseline] != [r['sample_id'] for r in predicted[company]]:
                raise ValueError('Heldout baseline row identity mismatch')
            combined = deepcopy(baseline)
            for row, prediction in zip(combined, predicted[company]):
                row['predictions']['price_label_bert'] = prediction['prediction']
            truth = [r['target_return'] for r in combined]
            all_metrics = {name:metrics(truth,[r['predictions'][name] for r in combined])
                           for name in combined[0]['predictions']}
            for item in all_metrics.values():
                item['mae_relative_to_train_mean'] = item['mae'] / all_metrics['train_mean']['mae']
            case = {'company':company,'training_rows':len(training[company]),'test_rows':len(combined),
                    'metrics':all_metrics,'cost_evaluation':cost_evaluation(combined,[0,10,20])}
            cases.append(case)
            save(out / (company + '.paired_predictions.json'), combined)
            receipt['actual_evaluated_rows'] += len(combined)
            save(out / 'training_receipt.json', receipt)
        receipt.update(status='complete', completed_at=stamp(), evaluation_cases=cases,
            evaluation_peak_observed_rss_bytes=budget.peak_observed_rss,
            evaluation_minimum_observed_available_bytes=budget.minimum_observed_available)
    except Exception as exc:
        receipt.update(status='incomplete', failed_at=stamp())
        receipt['failures'].append({'type':type(exc).__name__,'reason':str(exc)})
        print({'status':'incomplete','steps':receipt['actual_optimizer_steps'], 'reason':str(exc)},flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        receipt['elapsed_seconds'] = time.monotonic() - started
        receipt['last_budget_peak_observed_rss_bytes'] = budget.peak_observed_rss
        save(out / 'training_receipt.json', receipt)
        save(out / 'public_summary.json', public_summary(receipt, reg))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=run(args.repo,args.registration,args.output)
    print({'status':result['status'],'actual_optimizer_steps':result['actual_optimizer_steps']},flush=True)
    sys.exit(0 if result['status']=='complete' else 2)
