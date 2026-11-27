import gzip
import hashlib
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/pr917')]
import caaf_store as base
from m2_runtime.compact_store import CompactStore,seed_pilot_store


def test_lossless_blob_identity_and_tampering(tmp_path):
    st=CompactStore(tmp_path/'store')
    data=base.canonical({'full_context':'repeat '*1000})
    key=st.put_blob('context',data)
    assert key==hashlib.sha256(data).hexdigest()
    zipped=st.root/'blobs/context'/f'{key}.json.gz'
    assert zipped.stat().st_size<len(data)
    assert st.get_blob('context',key)==data
    zipped.chmod(0o600);zipped.write_bytes(gzip.compress(b'changed'))
    with pytest.raises(ValueError):st.get_blob('context',key)
    st.db.close()


def test_minimal_seed_preserves_original_snapshots_and_source(tmp_path):
    source=tmp_path/'source';old=base.Store(source)
    branches=[];snapshots=[]
    for i in range(2):
        rid=old.new_run({'i':i});bid=old.new_branch(rid)
        snap=old.put_snapshot(run_id=rid,branch_id=bid,seq=0,step=0,parent_id=None,
                              context={'messages':[],'i':i},env_state=base.canonical({'i':i}))
        old.finish_branch(bid,'finished',{'won':False})
        tape=base.Tape(old,rid,bid);tape.close()
        branches.append(bid);snapshots.append(snap)
    old.db.close()
    before=hashlib.sha256((source/'manifest.sqlite').read_bytes()).hexdigest()
    target=tmp_path/'new'
    assert seed_pilot_store(source,target,[branches[0]])=={'runs':1,'branches':1,'snapshots':1}
    new=CompactStore(target);snap=new.snapshot(snapshots[0]['id'])
    assert snap['context_hash']==snapshots[0]['context_hash']
    assert new.get_blob('context',snap['context_hash'])==base.canonical({'messages':[],'i':0})
    assert new.db.execute('select count(*) from branches').fetchone()[0]==1
    assert hashlib.sha256((source/'manifest.sqlite').read_bytes()).hexdigest()==before
    new.db.close()
