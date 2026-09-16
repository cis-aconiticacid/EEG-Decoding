import json
import math
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from run_d061_paper_latent import lr_for_step, CONFIG


def test_d061_configuration_changes_only_declared_fields():
    before=json.loads((ROOT/'config/d060_paper_latent_s17.json').read_text())
    after=json.loads(CONFIG.read_text())
    assert after['physical_batch_size']==64
    assert after['gradient_accumulation']==1
    assert {k for k in before if before[k]!=after[k]}=={
        'task_id','gradient_accumulation','run_dir','status','stop_request'}
    current=(ROOT/'scripts/run_d061_paper_latent.py').read_text(encoding='utf-8')
    assert 'ensure_trial_manifest' in current
    assert "CONFIG = ROOT / 'config/d061_paper_latent_s17.json'" in current


def test_new_update_counts_and_schedule():
    cfg=json.loads(CONFIG.read_text())
    effective=cfg['physical_batch_size']*cfg['gradient_accumulation']
    for n,expected in [(2400,38),(1200,19),(240,4),(2370,38),(1170,19),(210,4)]:
        steps=math.ceil(n/effective)
        assert steps==expected
        assert lr_for_step(1,steps,cfg)==pytest.approx(3e-5)
        assert lr_for_step(3*steps,steps,cfg)==pytest.approx(3e-4)
        assert lr_for_step(70*steps,steps,cfg)==pytest.approx(3e-5)
    assert list(range(cfg['checkpoint_every_epochs'],cfg['epochs']+1,cfg['checkpoint_every_epochs']))==[10,20,30,40,50,60,70]
