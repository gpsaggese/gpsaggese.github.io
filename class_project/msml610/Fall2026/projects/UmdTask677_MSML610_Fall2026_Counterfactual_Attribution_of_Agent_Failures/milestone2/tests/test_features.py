import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from m2_runtime.features import extract_features


def test_features_use_saved_before_state_and_exclude_labels():
    import csv
    d=next(x for x in csv.DictReader((ROOT/'experiments/m2_v1/checkpoint_inventory.csv').open()) if x['task_id']=='krk-m4-1' and x['trial']=='0' and x['step']=='0')
    ctx=json.loads((ROOT/d['context_path']).read_text());game=json.loads((ROOT/d['env_path']).read_text())
    f=extract_features(ctx,game)
    assert f['step_before']==0 and f['halfmove_clock']==85 and f['halfmoves_left']==15
    assert f['white_king_file']==4 and f['white_king_rank']==1
    assert f['last_white_move_was_king']==0 and f['last_white_move_was_rook']==0
    assert f['legal_moves_before']>0
    assert not any(any(w in k for w in ['oracle','outcome','reward','gain','after','future']) for k in f)


def test_extra_model_metadata_is_not_a_feature():
    import csv,copy
    d=next(csv.DictReader((ROOT/'experiments/m2_v1/checkpoint_inventory.csv').open()))
    ctx=json.loads((ROOT/d['context_path']).read_text());game=json.loads((ROOT/d['env_path']).read_text())
    changed=copy.deepcopy(ctx)
    for m in changed['messages']:m['extra']={'future_label':999,'oracle':True}
    assert extract_features(ctx,game)==extract_features(changed,game)
