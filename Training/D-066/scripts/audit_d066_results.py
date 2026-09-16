"""CPU/file-only audit of completed D066 predictions; never rerun model inference."""
import argparse
import json
import statistics
from pathlib import Path

def audit(root):
    run=root/'runs/d066-latent-finetune'
    summary=json.loads((root/'reports/d066_latent_finetune/summary.json').read_text())
    assert len(summary['results'])==3
    rows_checked=0
    samplers=set()
    results=[]
    for seed in [17,23,41]:
        out=run/f'all_s00_seed{seed}'
        d=json.loads((out/'result.json').read_text())
        epochs=json.loads((out/'epochs.json').read_text())
        assert [e['epoch'] for e in epochs]==list(range(1,71))
        assert all(e['coverage']==2400 and e['steps']==38 for e in epochs)
        # Seed-dependent samplers must be reproducible; do not require different seeds to match.
        samplers.add(tuple(e['sampler_hash'] for e in epochs))
        if (out/'epoch_010.pt').exists():
            assert [p.name for p in sorted(out.glob('epoch_*.pt'))]==[f'epoch_{i:03d}.pt' for i in range(10,71,10)]
        sets={}
        for split,n in [('test',1600),('train_same_test_gallery',2400)]:
            rows=json.loads((out/(split+'_predictions.json')).read_text())
            assert len(rows)==n and len({r['trial_id'] for r in rows})==n
            assert len({r['image'] for r in rows})==n
            correct=sum(r['actual_label']==r['predicted_label'] for r in rows)/n
            image=sum(r['image']==r['predicted_image'] for r in rows)/n
            assert abs(correct-d[split]['class_accuracy'])<1e-12
            assert abs(image-d[split]['image_top1'])<1e-12
            assert d[split]['gallery_size']==1600
            sets[split]={r['image'] for r in rows};rows_checked+=n
        assert not sets['test'] & sets['train_same_test_gallery']
        for split in sets:
            rows=json.loads((out/(split+'_predictions.json')).read_text())
            assert all(r['predicted_image'] in sets['test'] for r in rows)
        results.append({'seed':seed,'test':d['test'],'train_same_gallery':d['train_same_test_gallery']})
    avg=statistics.mean(d['test']['class_accuracy'] for d in results)
    sd=statistics.stdev(d['test']['class_accuracy'] for d in results)
    assert abs(avg-summary['mean_class_accuracy'])<1e-12 and abs(sd-summary['sample_std'])<1e-12
    result={'audit':'PASS','prediction_rows_verified':rows_checked,'completed_seeds':3,'epochs_per_seed':70,
            'results':results,'mean_class_accuracy':avg,'sample_std':sd,'new_model_evaluations':0}
    (root/'reports/d066_latent_finetune/audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args()
    audit(args.root)
