"""Frozen train-set diagnosis; same test-image readout, no new test EEG pass."""
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from d057_common import ROOT, sha, atomic_json, emit
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))
from eegdecoding.data import ensure_trial_manifest, load_part
from eegdecoding.paper_latent_alignment import PaperWaveformRAEEncoder
from eegdecoding.rae_latent_alignment import normalized_flat


def main():
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cfg=json.loads((ROOT/'config/d061_paper_latent_s17.json').read_text())
    ensure_trial_manifest()
    with (ROOT/cfg['manifest']).open(newline='') as f: rows=list(csv.DictReader(f))
    tc=json.loads((ROOT/cfg['target_dir']/'targets.json').read_text())
    targets=torch.empty((4000,256,1024),dtype=torch.float32)
    image_records=[]
    for split in ('train','monitor_val','final_holdout'):
        source=np.load(ROOT/cfg['target_dir']/(split+'.npy'),mmap_mode='r')
        offset=len(image_records)
        for lo in range(0,len(source),64):
            targets[offset+lo:offset+min(lo+64,len(source))].copy_(torch.from_numpy(np.array(source[lo:lo+64])))
        image_records.extend(tc['splits'][split]['images'])
        del source
    gallery=torch.empty((4000,256*1024),dtype=torch.float32)
    for lo in range(0,4000,64): gallery[lo:lo+64]=normalized_flat(targets[lo:lo+64])
    lookup={r['image_id']:i for i,r in enumerate(image_records)}
    labels=torch.tensor([int(r['label_index']) for r in image_records])
    results=[]
    for job_id in ('all_s00','all_s04'):
        started=time.monotonic(); run=ROOT/cfg['run_dir']/job_id
        saved=torch.load(run/'epoch_070.pt',map_location='cpu',weights_only=False)
        job=saved['job'];state=saved['model']
        assert saved['completed_epoch']==70 and saved['step']==2660
        model=PaperWaveformRAEEncoder(state['coordinates'],state['eeg_mean'],state['eeg_scale'],
                                      dim=cfg['model_dim'],depth=cfg['depth'],heads=cfg['heads'],
                                      dropout=cfg['dropout'],query_depth=cfg['query_depth'])
        model.load_state_dict(state);model.eval()
        train_rows=[rows[i] for i in job['train_indices']]
        train_ids=torch.tensor([lookup[r['image']] for r in train_rows])
        test_ids=torch.tensor([lookup[rows[i]['image']] for i in job['test_indices']])
        assert not set(train_ids.tolist()) & set(test_ids.tolist())
        raw=torch.empty((len(train_rows),62,400),dtype=torch.float32)
        for source in sorted({r['source_file'] for r in train_rows}):
            data=load_part(ROOT/source)
            for i,r in enumerate(train_rows):
                if r['source_file']!=source: continue
                item=data['dataset'][int(r['source_index'])]
                assert str(item['image'])==r['image'] and int(item['subject'])==job['subject']
                raw[i]=item['eeg_data'].float()[:,40:440]
            del data
        del saved,state
        gc.collect()
        counts={'same_test_gallery_class_correct':0,'own_train_gallery_class_correct':0,
                'own_train_gallery_image_correct':0,'train_gallery_excluding_paired_class_correct':0}
        mse=0.
        with torch.inference_mode():
            for lo in range(0,len(raw),32):
                z=model(raw[lo:lo+32]);truth=train_ids[lo:lo+32]
                similarities=normalized_flat(z) @ gallery.T
                mse+=float((z-targets[truth]).square().mean((1,2)).sum())
                nearest_test=test_ids[similarities[:,test_ids].argmax(1)]
                train_sim=similarities[:,train_ids]
                nearest_train=train_ids[train_sim.argmax(1)]
                counts['same_test_gallery_class_correct']+=int(labels[nearest_test].eq(labels[truth]).sum())
                counts['own_train_gallery_class_correct']+=int(labels[nearest_train].eq(labels[truth]).sum())
                counts['own_train_gallery_image_correct']+=int(nearest_train.eq(truth).sum())
                train_sim[torch.arange(len(truth)),torch.arange(lo,lo+len(truth))]=-torch.inf
                nearest_other=train_ids[train_sim.argmax(1)]
                counts['train_gallery_excluding_paired_class_correct']+=int(labels[nearest_other].eq(labels[truth]).sum())
                if (lo+32)%320==0 or lo+32>=len(raw):
                    emit('train_diagnostic_progress',job_id=job_id,completed=min(lo+32,len(raw)),
                         total=len(raw),seconds=time.monotonic()-started)
        test=json.loads((run/'result.json').read_text())
        result={'job_id':job_id,'epoch':70,'device':'cpu','eval_mode':True,'train_trials':len(raw),
                'checkpoint_sha256':sha(run/'epoch_070.pt'),
                'clean_train_latent_mse':mse/len(raw),
                'train_metrics':{k.replace('_correct','_accuracy'):v/len(raw) for k,v in counts.items()},
                'test_class_accuracy_from_existing_pass':test['class_accuracy'],
                'test_latent_mse_from_existing_pass':test['latent_mse'],
                'same_gallery_size':len(test_ids),'own_train_gallery_size':len(train_ids),
                'new_test_eeg_evaluations':0,'seconds':time.monotonic()-started}
        result['same_gallery_generalization_gap_pp']=100*(counts['same_test_gallery_class_correct']/len(raw)-test['class_accuracy'])
        results.append(result)
        atomic_json(ROOT/'reports/d061_fit_diagnosis.json',{'results':results,'completed':len(results)==2})
        emit('diagnostic_complete',**result)
        del model,raw
        gc.collect()


if __name__=='__main__': main()
