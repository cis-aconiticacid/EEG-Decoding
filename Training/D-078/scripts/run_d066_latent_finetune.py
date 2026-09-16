"""D066 supervised image-latent fine-tuning from D065 masked EEG encoders."""
import argparse
import csv
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eegdecoding.data import ensure_trial_manifest
import run_d064_mask_ratio as b
from eegdecoding.rae_latent_alignment import QueryBlock, grid_encoding, latent_loss, normalized_flat

ROOT=b.ROOT
STOP=False

class ImageDecoder(nn.Module):
    def __init__(self, encoder, seed):
        super().__init__()
        self.encoder=encoder
        # The masked reconstruction head and mask token are unused in full-input fine-tuning.
        self.encoder.output.requires_grad_(False)
        self.encoder.mask_token.requires_grad_(False)
        torch.manual_seed(seed+66000)
        self.memory_norm=nn.LayerNorm(128)
        self.queries=nn.Parameter(torch.empty(256,128))
        nn.init.trunc_normal_(self.queries,std=.02)
        self.register_buffer('grid',grid_encoding(16,128))
        self.blocks=nn.ModuleList([QueryBlock(128,4,.1) for _ in range(2)])
        self.projector=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,512),nn.GELU(),nn.Dropout(.1),nn.Linear(512,1024))

    def forward(self,x):
        memory=self.memory_norm(self.encoder.encode(x))
        q=(self.queries+self.grid)[None].expand(len(x),-1,-1)
        for block in self.blocks: q=block(q,memory)
        return self.projector(q)

def prepare(cfg,report):
    ensure_trial_manifest()
    base=json.loads((ROOT/'config/d064_local_mask_ratio.json').read_text())
    small,_,_,_,coords,small_rows,parent=b.data(base,ROOT/base['report_dir'])
    stats=torch.load(ROOT/base['report_dir']/'normalization.pt',map_location='cpu',weights_only=True)
    with (ROOT/base['manifest']).open(newline='',encoding='utf-8') as f:
        rows=sorted([r for r in csv.DictReader(f) if int(r['subject'])==0],key=lambda r:int(r['source_index']))
    assert len(rows)==4000
    train=torch.tensor([i for i,r in enumerate(rows) if int(r['position_in_class_block'])<30])
    test=torch.tensor([i for i,r in enumerate(rows) if int(r['position_in_class_block'])>=30])
    assert (len(train),len(test))==(2400,1600)
    assert not ({rows[i]['image'] for i in train.tolist()} & {rows[i]['image'] for i in test.tolist()})
    archive=b.load_part(ROOT/base['archive'])
    raw=torch.empty(4000,62,400)
    for i,r in enumerate(rows):
        item=archive['dataset'][int(r['source_index'])]
        assert int(item['subject'])==0 and str(item['image'])==r['image'] and str(item['label'])==r['label']
        raw[i]=item['eeg_data'][:,40:440].float()
    assert torch.isfinite(raw).all()
    x=((b.frequency(raw)-stats['mean'])/stats['std']).reshape(4000,496,4)
    assert torch.equal(x[train],small), 'Pretrained input transform changed'
    assert [rows[i]['trial_id'] for i in train.tolist()]==[r['trial_id'] for r in small_rows]
    tcpath=ROOT/cfg['target_dir']/'targets.json'
    tc=json.loads(tcpath.read_text())
    teacher=json.loads((ROOT/cfg['teacher_config']).read_text())
    assert tc['shape_per_image']==[256,1024] and tc['all_image_count']==4000
    assert tc['identity']['config_sha256']==b.sha(ROOT/cfg['teacher_config'])
    assert tc['teacher_sha256']==teacher['teacher_sha256'] and tc['stats_sha256']==teacher['stats_sha256']
    sources={str(s):ROOT/'runs/d065-mask-continue'/f'mask60-s{s}/epoch_140.pt' for s in cfg['seeds']}
    for seed,p in sources.items():
        saved=torch.load(p,map_location='cpu',weights_only=False)
        assert saved['epoch']==140 and saved['seed']==int(seed) and saved['arm']=='mask60'
        assert saved['contract']['parent']==parent
        enc=b.MaskedFrequency(coords,int(seed));enc.load_state_dict(saved['model'],strict=True)
    contract={'config':b.sha(ROOT/'config/d066_latent_finetune.json'),'code':b.sha(__file__),
              'latent_modules':b.sha(PROJECT_ROOT/'src/eegdecoding/rae_latent_alignment.py'),
              'parent':parent,'sources':{s:b.sha(p) for s,p in sources.items()},
              'target_contract':b.sha(tcpath),'raw4000':b.thash(raw),'train':b.thash(train),'test':b.thash(test)}
    if (report/'contract.json').exists(): assert json.loads((report/'contract.json').read_text())==contract
    b.dump(report/'contract.json',contract)
    b.dump(report/'split.json',{'train_trials':[rows[i]['trial_id'] for i in train.tolist()],
                                'test_trials':[rows[i]['trial_id'] for i in test.tolist()]})
    return x,train,test,coords,rows,tc,sources,contract,base

def teacher_targets(cfg,tc,rows):
    targets=torch.empty(4000,256,1024,device='cuda')
    records=[]
    for split in ['train','monitor_val','final_holdout']:
        meta=tc['splits'][split];p=ROOT/cfg['target_dir']/(split+'.npy')
        assert meta['complete'] and b.sha(p)==meta['array_sha256']
        array=np.load(p,mmap_mode='r')
        assert list(array.shape)==meta['shape'] and array.dtype==np.float32
        offset=len(records)
        for lo in range(0,len(array),64):
            targets[offset+lo:offset+min(lo+64,len(array))].copy_(torch.from_numpy(np.array(array[lo:lo+64])))
        records.extend(meta['images'])
    lookup={r['image_id']:i for i,r in enumerate(records)}
    assert len(lookup)==4000
    ids=torch.tensor([lookup[r['image']] for r in rows],device='cuda')
    for r in rows: assert int(records[lookup[r['image']]]['label_index'])==int(r['label_index'])
    gallery=torch.empty(4000,256*1024,device='cuda')
    for lo in range(0,4000,64):
        assert torch.isfinite(targets[lo:lo+64]).all()
        gallery[lo:lo+64]=normalized_flat(targets[lo:lo+64])
    labels=torch.tensor([int(r['label_index']) for r in records],device='cuda')
    return targets,gallery,ids,labels,records

@torch.no_grad()
def evaluate(model,x,indices,ids,targets,gallery,labels,candidates,records,rows,out,split):
    model.eval();predictions=[];mse=0.;correct=image_correct=0
    for ix in indices.split(32):
        pred=model(x[ix]);mse+=float((pred-targets[ids[ix]]).square().mean())*len(ix)
        q=normalized_flat(pred)
        best=torch.full((len(ix),),-float('inf'),device='cuda');winner=torch.zeros(len(ix),device='cuda',dtype=torch.long)
        for chunk in candidates.split(256):
            scores=q@gallery[chunk].T
            values,local=scores.max(1);take=values>best
            winner=torch.where(take,chunk[local],winner);best=torch.maximum(best,values)
        c=labels[winner]==labels[ids[ix]];im=winner==ids[ix]
        correct+=int(c.sum());image_correct+=int(im.sum())
        for j,i in enumerate(ix.cpu().tolist()):
            w=int(winner[j]);predictions.append({'trial_id':rows[i]['trial_id'],'image':rows[i]['image'],
              'actual_label':int(rows[i]['label_index']),'predicted_image':records[w]['image_id'],
              'predicted_label':int(labels[w]),'similarity':float(best[j])})
    b.dump(out/(split+'_predictions.json'),predictions)
    return {'n':len(indices),'latent_mse':mse/len(indices),'class_accuracy':correct/len(indices),
            'image_top1':image_correct/len(indices),'gallery_size':len(candidates)}

def checks():
    torch.set_num_threads(4)
    enc=b.MaskedFrequency(torch.randn(62,3),17)
    model=ImageDecoder(enc,17)
    out=model(torch.randn(2,496,4))
    assert out.shape==(2,256,1024)
    loss,_=latent_loss(out,torch.randn_like(out),torch.arange(2))
    loss.backward()
    assert enc.projection.weight.grad is not None and enc.layers[-1].linear2.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder.output.weight.grad is None
    print(json.dumps({'checks':'PASS','shape':list(out.shape),'encoder_receives_gradient':True,
                     'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    checks()
    if not (args.train or args.prepare):return 0
    cfg=json.loads((ROOT/'config/d066_latent_finetune.json').read_text())
    report=ROOT/cfg['report_dir'];report.mkdir(parents=True,exist_ok=True)
    x,train,test,coords,rows,tc,sources,contract,base=prepare(cfg,report)
    b.dump(report/'preflight.json',{'status':'PASS','train':len(train),'test':len(test),'contract':contract})
    if not args.train:return 0
    b.guard(base)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    def status(**kw):
        d={'task':'D066','pid':os.getpid(),'updated_unix':time.time(),**kw};b.dump(report/'status.json',d);print(json.dumps(d),flush=True)
    status(phase='loading_image_targets')
    targets,gallery,ids,labels,image_records=teacher_targets(cfg,tc,rows)
    x,train,test=[v.cuda() for v in (x,train,test)]
    candidates=ids[test];assert len(candidates.unique())==1600
    status(phase='resident_data_ready',features_device=str(x.device),targets_device=str(targets.device),allocated_gib=torch.cuda.memory_allocated()/2**30)
    segment=time.monotonic();results=[]
    for seed in cfg['seeds']:
        if STOP or (ROOT/'.runtime/d066_stop_requested').exists():return 76
        out=ROOT/cfg['run_dir']/f'all_s00_seed{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():
            d=json.loads((out/'result.json').read_text());assert d['contract']==contract;results.append(d);continue
        saved=torch.load(sources[str(seed)],map_location='cpu',weights_only=False)
        enc=b.MaskedFrequency(coords,seed);enc.load_state_dict(saved['model']);del saved
        model=ImageDecoder(enc,seed).cuda()
        pretrained_hash=b.thash(model.encoder.projection.weight)
        enc_params=[p for p in model.encoder.parameters() if p.requires_grad]
        new_params=[p for name,p in model.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
        opt=torch.optim.AdamW([{'params':enc_params,'lr':3e-6,'scale':.1},{'params':new_params,'lr':3e-5,'scale':1.}],weight_decay=.001)
        start=0;history=[]
        ckpts=sorted(out.glob('epoch_*.pt'))
        if ckpts:
            saved=torch.load(ckpts[-1],map_location='cuda',weights_only=False)
            assert saved['contract']==contract and saved['seed']==seed
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history'];del saved
        for epoch in range(start+1,71):
            tick=time.perf_counter();torch.manual_seed(seed*10000+epoch);model.train()
            perm=b.order(seed,epoch,2400).cuda();total=mse_sum=contrast_sum=0.;seen=steps=0
            for group in opt.param_groups:group['lr']=b.lr(epoch)*group['scale']
            for ix in train[perm].split(64):
                opt.zero_grad(set_to_none=True)
                loss,parts=latent_loss(model(x[ix]),targets[ids[ix]],ids[ix],temperature=.07,contrastive_weight=.1)
                assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                n=len(ix);total+=float(loss.detach())*n;mse_sum+=float(parts['latent_mse'])*n;contrast_sum+=float(parts['contrastive'])*n;seen+=n;steps+=1
            torch.cuda.synchronize();assert seen==2400 and steps==38
            rec={'epoch':epoch,'loss':total/seen,'latent_mse':mse_sum/seen,'contrastive':contrast_sum/seen,'coverage':seen,'steps':steps,
                 'sampler_hash':b.thash(perm),'head_lr':b.lr(epoch),'encoder_lr':b.lr(epoch)*.1,'seconds':time.perf_counter()-tick}
            history.append(rec);b.dump(out/'epochs.json',history);status(phase='finetuning',seed=seed,**rec)
            if epoch%10==0:
                b.save(out/f'epoch_{epoch:03d}.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'seed':seed,'contract':contract,'history':history})
                if STOP or (ROOT/'.runtime/d066_stop_requested').exists():return 76
                if time.monotonic()-segment>1800 and epoch<70:return 75
        assert len(list(out.glob('epoch_*.pt')))==7
        assert b.thash(model.encoder.projection.weight)!=pretrained_hash, 'Encoder was not fine-tuned'
        status(phase='final_evaluation',seed=seed)
        test_result=evaluate(model,x,test,ids,targets,gallery,labels,candidates,image_records,rows,out,'test')
        train_result=evaluate(model,x,train,ids,targets,gallery,labels,candidates,image_records,rows,out,'train_same_test_gallery')
        result={'seed':seed,'subject':0,'epoch':70,'test':test_result,'train_same_test_gallery':train_result,'contract':contract,
                'readout':'image-assisted nearest latent among 1600 subject0 test images; not paper CE classifier',
                'checkpoint_count':7,'training_seconds':sum(d['seconds'] for d in history)}
        b.dump(out/'result.json',result);results.append(result);b.dump(report/'completed_runs.json',results)
        del model,opt;torch.cuda.empty_cache()
        if time.monotonic()-segment>1800 and seed!=cfg['seeds'][-1]:return 75
    b.dump(report/'summary.json',{'results':results,'mean_class_accuracy':statistics.mean(d['test']['class_accuracy'] for d in results),
            'sample_std':statistics.stdev(d['test']['class_accuracy'] for d in results),'subject_count':1,'seed_count':3})
    b.dump(ROOT/cfg['run_dir']/'completed.json',{'runs':3,'contract':contract});status(phase='completed',runs=3)
    return 0

def stop(*_):
    global STOP
    STOP=True

if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    raise SystemExit(main())
