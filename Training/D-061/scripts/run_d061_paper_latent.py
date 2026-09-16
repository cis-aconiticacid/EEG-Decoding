"""Independent subject/task fits, 30/20 source-order split, seven checkpoints.

The frozen test-image gallery is used only for final nearest-latent readout.
This is an image-assisted readout, not the paper's parametric classifier head.
"""
from __future__ import annotations
import csv
import gc
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from d057_common import ROOT, sha, atomic_json, emit, require_gpu0, code_contract
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from eegdecoding.data import ensure_trial_manifest, load_part
from eegdecoding.paper_latent_alignment import PaperWaveformRAEEncoder
from eegdecoding.rae_latent_alignment import latent_loss, normalized_flat

CONFIG = ROOT / 'config/d061_paper_latent_s17.json'


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def make_jobs(rows, cfg):
    """Use each subject's own class-block order, never a global image order."""
    if len(rows) != 63850 or len({r['trial_id'] for r in rows}) != len(rows):
        raise ValueError('unexpected full corpus')
    jobs = []
    for task in cfg['tasks']:
        allowed = (set(range(80)) if task == 'all' else set(range(40, 80))
                   if task == 'coarse' else set(range(int(task[-1])*8, int(task[-1])*8+8)))
        for subject in cfg['subjects']:
            selected = [i for i, r in enumerate(rows)
                        if int(r['subject']) == subject and int(r['label_index']) in allowed]
            present = sorted({int(rows[i]['label_index']) for i in selected})
            for label in present:
                block = [i for i in selected if int(rows[i]['label_index']) == label]
                if sorted(int(rows[i]['position_in_class_block']) for i in block) != list(range(50)):
                    raise ValueError('incomplete or duplicated within-class positions')
                if len({rows[i]['image'] for i in block}) != 50:
                    raise ValueError('repeated image in subject/class block')
            train = [i for i in selected if int(rows[i]['position_in_class_block']) < 30]
            test = [i for i in selected if int(rows[i]['position_in_class_block']) >= 30]
            if (not train or not test or set(train) & set(test)
                    or {rows[i]['image'] for i in train} & {rows[i]['image'] for i in test}
                    or len(train) != len(present)*30 or len(test) != len(present)*20):
                raise ValueError('subject/task train-test isolation failed')
            jobs.append(dict(job_id=f'{task}_s{subject:02d}', task=task, subject=subject,
                             present_classes=present, missing_classes=sorted(allowed-set(present)),
                             train_indices=train, test_indices=test))
    if len(jobs) != 112:
        raise ValueError('expected 16 subjects x 7 independent tasks')
    return jobs


def lr_for_step(step, steps_per_epoch, cfg):
    warm = cfg['warmup_epochs'] * steps_per_epoch
    end = cfg['epochs'] * steps_per_epoch
    if step <= warm:
        return cfg['lr_start'] + (cfg['lr_peak']-cfg['lr_start'])*(step-1)/max(1, warm-1)
    phase = min(1., (step-warm)/(end-warm))
    return cfg['lr_end'] + .5*(cfg['lr_peak']-cfg['lr_end'])*(1+math.cos(math.pi*phase))


def publish_summary(run, cfg):
    results = [json.loads(p.read_text()) for p in sorted(run.glob('*/result.json'))]
    per_subject = []
    for subject in cfg['subjects']:
        subset = {r['task']: r for r in results if r['subject'] == subject}
        row = {'subject': subject}
        for task in ('all', 'coarse'):
            if task in subset:
                row[task] = subset[task]['class_accuracy']
        fine = [subset[f'fine{k}']['class_accuracy'] for k in range(5) if f'fine{k}' in subset]
        if len(fine) == 5:
            row['fine'] = float(np.mean(fine))
        per_subject.append(row)
    aggregate = {}
    for task in ('all', 'coarse', 'fine'):
        values = [r[task] for r in per_subject if task in r]
        if values:
            aggregate[task] = {'subjects_completed': len(values), 'mean': float(np.mean(values)),
                               'sample_std': float(np.std(values, ddof=1)) if len(values)>1 else None}
    summary = dict(completed_jobs=len(results), total_jobs=112, aggregate=aggregate,
                   per_subject=per_subject, results=results,
                   protocol='30/20 within-subject source order; canonical task labels; fixed epoch70',
                   readout='nearest latent among task/subject test images; image-assisted gallery',
                   not_identical_to_paper_classifier=True)
    atomic_json(ROOT / 'reports/d061_summary.json', summary)
    return summary


def main():
    cfg = json.loads(CONFIG.read_text())
    ensure_trial_manifest()
    if (cfg['epochs'], cfg['checkpoint_every_epochs'], cfg['checkpoint_count'],
        cfg['evaluate_test_at_epoch'], cfg['early_stopping']) != (70,10,7,70,False):
        raise ValueError('user requested exactly 70 epochs and 7 ten-epoch checkpoints')
    if cfg['crop'] != [40,440] or cfg['train_positions'] != [0,30] or cfg['test_positions'] != [30,50]:
        raise ValueError('unsupported split/crop change')
    for key in ('manifest', 'channel_map', 'teacher_config'):
        if sha(ROOT / cfg[key]) != cfg[key+'_sha256']:
            raise ValueError(key+' hash mismatch')
    rows = read_csv(ROOT / cfg['manifest'])
    jobs = make_jobs(rows, cfg)
    target_path = ROOT / cfg['target_dir'] / 'targets.json'
    tc = json.loads(target_path.read_text())
    teacher_cfg = json.loads((ROOT / cfg['teacher_config']).read_text())
    if (tc['shape_per_image'] != [256,1024] or tc['all_image_count'] != 4000
        or tc['identity']['config_sha256'] != cfg['teacher_config_sha256']
        or tc['teacher_sha256'] != teacher_cfg['teacher_sha256']
        or tc['stats_sha256'] != teacher_cfg['stats_sha256']):
        raise ValueError('frozen teacher contract mismatch')
    contract = dict(config_sha256=sha(CONFIG), target_contract_sha256=sha(target_path),
                    torch=torch.__version__, code=code_contract([
                        'scripts/run_d061_paper_latent.py', 'scripts/d057_common.py',
                        'src/eegdecoding/paper_latent_alignment.py', 'src/eegdecoding/rae_latent_alignment.py',
                        'src/eegdecoding/waveform_alignment.py', 'src/eegdecoding/data.py']))
    run = ROOT / cfg['run_dir']; run.mkdir(parents=True, exist_ok=True)
    if (run/'contract.json').exists() and json.loads((run/'contract.json').read_text()) != contract:
        raise ValueError('campaign contract changed; refusing to mix runs')
    atomic_json(run/'contract.json', contract)
    atomic_json(run/'split_manifest.json', {'jobs': jobs, 'trial_ids': [r['trial_id'] for r in rows]})
    require_gpu0(cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    started = time.monotonic()

    def status(state, **fields):
        atomic_json(ROOT/cfg['status'], dict(task_id='D061', status=state,
                    updated_unix=time.time(), pid=os.getpid(), **fields))
        emit(state, **fields)

    status('loading_resident_data', total_jobs=112)
    targets = torch.empty((4000,256,1024), device='cuda', dtype=torch.float32)
    image_records = []
    for split in ('train','monitor_val','final_holdout'):
        part = tc['splits'][split]
        path = ROOT/cfg['target_dir']/(split+'.npy')
        if not part['complete'] or sha(path) != part['array_sha256']:
            raise ValueError('teacher array incomplete or changed')
        mapped = np.load(path, mmap_mode='r')
        if list(mapped.shape) != part['shape'] or mapped.dtype != np.float32:
            raise ValueError('teacher shape/dtype mismatch')
        offset = len(image_records)
        for lo in range(0,len(mapped),64):
            targets[offset+lo:offset+min(lo+64,len(mapped))].copy_(torch.from_numpy(np.array(mapped[lo:lo+64])))
        image_records.extend(part['images'])
        del mapped
    if len(image_records)!=4000 or len({r['image_id'] for r in image_records})!=4000:
        raise ValueError('missing/duplicated image targets')
    lookup = {r['image_id']: i for i,r in enumerate(image_records)}
    ids = torch.tensor([lookup[r['image']] for r in rows], device='cuda')
    labels = torch.tensor([int(r['label_index']) for r in image_records], device='cuda')
    for r in rows:
        if int(image_records[lookup[r['image']]]['label_index']) != int(r['label_index']):
            raise ValueError('EEG-image-label identity mismatch')
    raw = torch.empty((len(rows),62,400), device='cuda', dtype=torch.float32)
    # One sequential read per archive per segment. No disk reads in training batches.
    for source in sorted({r['source_file'] for r in rows}):
        loaded = load_part(ROOT/source)
        selected = [i for i,r in enumerate(rows) if r['source_file']==source]
        if len(selected) != len(loaded['dataset']):
            raise ValueError('archive trial count differs')
        for lo in range(0,len(selected),128):
            ix = selected[lo:lo+128]; batch = []
            for i in ix:
                r = rows[i]; item = loaded['dataset'][int(r['source_index'])]
                if (str(item['image']) != r['image'] or str(item['label']) != r['label']
                        or int(item['subject']) != int(r['subject'])):
                    raise ValueError('archive row identity differs')
                batch.append(item['eeg_data'].float()[:,40:440])
            values = torch.stack(batch)
            if values.shape[1:] != (62,400) or not torch.isfinite(values).all():
                raise ValueError('invalid raw EEG')
            raw[torch.tensor(ix,device='cuda')] = values.cuda()
        del loaded, batch, values
        gc.collect()
    gallery = torch.empty((4000,256*1024),device='cuda',dtype=torch.float32)
    for lo in range(0,4000,64):
        if not torch.isfinite(targets[lo:lo+64]).all():
            raise ValueError('nonfinite teacher targets')
        gallery[lo:lo+64] = normalized_flat(targets[lo:lo+64])
    channels = sorted(read_csv(ROOT/cfg['channel_map']), key=lambda r:int(r['tensor_index']))
    coordinates = [[float(r[f'{a}_m_template_fit'])/.1 for a in 'xyz'] for r in channels]
    status('resident_data_ready', raw_trials=len(rows), target_images=4000,
           raw_shape=list(raw.shape), raw_device=str(raw.device), targets_device=str(targets.device),
           allocated_gib=torch.cuda.memory_allocated()/2**30, load_seconds=time.monotonic()-started)

    for job in jobs:
        jd = run/job['job_id']; jd.mkdir(exist_ok=True)
        if (jd/'result.json').exists():
            continue
        if stop[0] or (ROOT/cfg['stop_request']).exists():
            status('paused_by_user', job_id=job['job_id']); return 0
        train = torch.tensor(job['train_indices'],device='cuda')
        test = torch.tensor(job['test_indices'],device='cuda')
        # Train-only FP64 reductions; convert buffers to FP32. Test never enters these statistics.
        sums = torch.zeros((62,),device='cuda',dtype=torch.float64)
        squares = torch.zeros_like(sums)
        for lo in range(0,len(train),128):
            values = raw[train[lo:lo+128]].double()
            sums += values.sum((0,2)); squares += values.square().sum((0,2))
        count = len(train)*400
        mean = sums/count
        scale = (squares/count-mean.square()).clamp_min(1e-12).sqrt()
        del values
        torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        model = PaperWaveformRAEEncoder(coordinates,mean,scale,dim=cfg['model_dim'],depth=cfg['depth'],
                     heads=cfg['heads'],dropout=cfg['dropout'],query_depth=cfg['query_depth']).cuda()
        decay, no_decay = [], []
        for name,p in model.named_parameters():
            (no_decay if p.ndim==1 or name.endswith('.bias') else decay).append(p)
        optimizer = torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},
                                      {'params':no_decay,'weight_decay':0.}],lr=cfg['lr_start'])
        completed_epoch, step, active_seconds = 0, 0, 0.
        order = None
        checkpoints = sorted(jd.glob('epoch_*.pt'))
        if checkpoints:
            saved = torch.load(checkpoints[-1],map_location='cpu',weights_only=False)
            if saved['contract']!=contract or saved['job']!=job:
                raise ValueError('checkpoint identity mismatch')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            completed_epoch, step = saved['completed_epoch'],saved['step']
            active_seconds = saved['active_seconds']
            torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved
        atomic_json(jd/'job.json', dict(**job, contract=contract, eeg_statistics='this job training rows only'))
        physical=cfg['physical_batch_size']; effective=physical*cfg['gradient_accumulation']
        steps_per_epoch=math.ceil(len(train)/effective)
        status('training',job_id=job['job_id'], completed_epoch=completed_epoch,step=step,
               train_trials=len(train),test_trials=len(test),parameters=sum(p.numel() for p in model.parameters()),
               initialization='random_seed17' if not checkpoints else 'own_ten_epoch_checkpoint')
        for epoch in range(completed_epoch+1,cfg['epochs']+1):
            tick=time.monotonic(); model.train()
            gen=torch.Generator(device='cuda').manual_seed(cfg['seed']+epoch*1000003)
            order=train[torch.randperm(len(train),generator=gen,device='cuda')]
            order_sha=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            totals={k:0. for k in ('loss','latent_mse','contrastive')}; seen=0
            for lo in range(0,len(order),effective):
                hi=min(lo+effective,len(order)); lr=lr_for_step(step+1,steps_per_epoch,cfg)
                for group in optimizer.param_groups: group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                for start in range(lo,hi,physical):
                    ix=order[start:min(start+physical,hi)]
                    prediction=model(raw[ix])
                    loss,parts=latent_loss(prediction,targets[ids[ix]],ids[ix],
                                           cfg['temperature'],cfg['contrastive_weight'])
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
                    (loss*(len(ix)/(hi-lo))).backward()
                    for key in totals: totals[key]+=float(parts[key])*len(ix)
                    seen+=len(ix)
                    del prediction,loss,parts
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); step+=1
            torch.cuda.synchronize()
            seconds=time.monotonic()-tick; active_seconds+=seconds
            if seen!=len(train): raise RuntimeError('epoch coverage incomplete')
            record=dict(job_id=job['job_id'],epoch=epoch,step=step,train_samples_seen=seen,
                        online_train={k:v/seen for k,v in totals.items()},learning_rate=lr,
                        epoch_seconds=seconds,active_seconds=active_seconds,order_sha256=order_sha,
                        test_evaluated=False,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            atomic_json(jd/f'epoch_{epoch:03d}.json',record)
            status('epoch_complete',**record)
            if epoch % cfg['checkpoint_every_epochs']==0:
                snapshot=dict(contract=contract,job=job,model=model.state_dict(),optimizer=optimizer.state_dict(),
                              completed_epoch=epoch,step=step,active_seconds=active_seconds,
                              cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                              test_evaluated=False)
                temp=jd/f'epoch_{epoch:03d}.tmp'
                torch.save(snapshot,temp); os.replace(temp,jd/f'epoch_{epoch:03d}.pt')
                del snapshot
                # Yield only on a saved boundary; no extra recovery/best/latest checkpoints.
                requested=stop[0] or (ROOT/cfg['stop_request']).exists()
                due=time.monotonic()-started>=cfg['segment_seconds']
                if requested or (due and epoch<cfg['epochs']):
                    status('paused_by_user' if requested else 'yielding',job_id=job['job_id'],completed_epoch=epoch)
                    return 0 if requested else 75
        expected=[f'epoch_{e:03d}.pt' for e in range(10,71,10)]
        if sorted(p.name for p in jd.glob('*.pt'))!=expected:
            raise RuntimeError('checkpoint count/names differ from seven-snapshot contract')
        # One final test pass. Candidates are frozen external image features, never gradient targets from test.
        status('final_test',job_id=job['job_id'],completed_epoch=70,test_trials=len(test))
        model.eval(); candidates=ids[test]; candidate_labels=labels[candidates]
        predictions=[]; hits=0; image_hits=0; total_mse=0.
        with torch.inference_mode():
            for lo in range(0,len(test),cfg['evaluation_batch_size']):
                ix=test[lo:lo+cfg['evaluation_batch_size']]; z=model(raw[ix]); q=normalized_flat(z)
                similarities=torch.cat([q @ gallery[candidates[g:g+cfg['gallery_chunk_size']]].T
                                         for g in range(0,len(candidates),cfg['gallery_chunk_size'])],1)
                best=similarities.argmax(1); pred_ids=candidates[best]
                pred_labels=candidate_labels[best]; truth_labels=labels[ids[ix]]
                if not torch.isfinite(similarities).all(): raise RuntimeError('nonfinite test output')
                hits+=int(pred_labels.eq(truth_labels).sum()); image_hits+=int(pred_ids.eq(ids[ix]).sum())
                total_mse+=float((z-targets[ids[ix]]).square().mean((1,2)).sum())
                for global_i,pi,pl,tl in zip(ix.tolist(),pred_ids.tolist(),pred_labels.tolist(),truth_labels.tolist()):
                    predictions.append(dict(trial_id=rows[global_i]['trial_id'],true_image=rows[global_i]['image'],
                                            predicted_image=image_records[pi]['image_id'],true_label=tl,predicted_label=pl))
        atomic_json(jd/'predictions.json',predictions)
        result=dict(job_id=job['job_id'],task=job['task'],subject=job['subject'],epoch=70,
                    train_trials=len(train),test_trials=len(test),gallery_size=len(candidates),
                    class_accuracy=hits/len(test),image_top1=image_hits/len(test),latent_mse=total_mse/len(test),
                    present_classes=job['present_classes'],missing_classes=job['missing_classes'],
                    active_train_seconds=active_seconds,checkpoint_count=7,contract=contract)
        atomic_json(jd/'result.json',result)
        summary=publish_summary(run,cfg)
        status('job_complete',job_id=job['job_id'],completed_jobs=summary['completed_jobs'],total_jobs=112,
               class_accuracy=result['class_accuracy'],active_train_seconds=active_seconds)
        del model,optimizer,decay,no_decay,order
        gc.collect(); torch.cuda.empty_cache()
        if time.monotonic()-started>=cfg['segment_seconds']:
            return 75
    summary=publish_summary(run,cfg)
    if summary['completed_jobs']!=112: raise RuntimeError('missing campaign results')
    atomic_json(run/'completed.json',summary)
    status('completed',completed_jobs=112,total_jobs=112,aggregate=summary['aggregate'])
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        previous_path=ROOT/'reports/d061_status.json'
        previous=json.loads(previous_path.read_text()) if previous_path.exists() else {}
        atomic_json(previous_path,{**previous,'status':'failed','updated_unix':time.time(),
                                  'error_type':type(exc).__name__,'error':str(exc)})
        raise
