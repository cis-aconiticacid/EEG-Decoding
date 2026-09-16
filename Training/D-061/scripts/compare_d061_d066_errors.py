"""Compare saved predictions only; no inference or training."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def compare(old,new,kind):
    if kind=='class':
        old_bad={i for i,r in old.items() if r['predicted_label']!=r['true_label']}
        new_bad={i for i,r in new.items() if r['predicted_label']!=r['actual_label']}
    else:
        old_bad={i for i,r in old.items() if r['predicted_image']!=r['true_image']}
        new_bad={i for i,r in new.items() if r['predicted_image']!=r['image']}
    joint=old_bad&new_bad
    union=old_bad|new_bad
    expected=len(old_bad)*len(new_bad)/len(old)
    return {'old_errors':len(old_bad),'new_errors':len(new_bad),'both_wrong':len(joint),
            'old_wrong_now_correct':len(old_bad-new_bad),'old_correct_now_wrong':len(new_bad-old_bad),
            'both_correct':len(old)-len(union),'jaccard':len(joint)/len(union),
            'fraction_new_errors_also_old':len(joint)/len(new_bad),
            'fraction_old_errors_still_wrong':len(joint)/len(old_bad),
            'both_wrong_fraction_of_test':len(joint)/len(old),
            'expected_both_wrong_if_independent':expected,
            'both_wrong_same_prediction':sum(old[i]['predicted_label' if kind=='class' else 'predicted_image']==new[i]['predicted_label' if kind=='class' else 'predicted_image'] for i in joint),
            'error_ids':sorted(new_bad)}

def main():
    out=ROOT/'reports/d067_error_overlap';out.mkdir(parents=True,exist_ok=True)
    old_list=json.loads((ROOT/'runs/d061-paper-latent-s17/all_s00/predictions.json').read_text())
    old={r['trial_id']:r for r in old_list}
    assert len(old)==len(old_list)==1600 and len({r['true_image'] for r in old_list})==1600
    results=[];detail=[];sets=[]
    for seed in [17,23,41]:
        rows=json.loads((ROOT/f'runs/d066-latent-finetune/all_s00_seed{seed}/test_predictions.json').read_text())
        new={r['trial_id']:r for r in rows}
        assert len(new)==len(rows)==1600 and old.keys()==new.keys()
        for i in old:
            assert old[i]['true_image']==new[i]['image'] and old[i]['true_label']==new[i]['actual_label']
        d={'seed':seed,'n':1600,'class':compare(old,new,'class'),'image':compare(old,new,'image')}
        sets.append(set(d['class']['error_ids']))
        results.append(d)
        for i,a in old.items():
            z=new[i];detail.append({'seed':seed,'trial_id':i,'image':a['true_image'],'true_label':a['true_label'],
              'old_predicted_label':a['predicted_label'],'new_predicted_label':z['predicted_label'],
              'old_class_wrong':int(a['predicted_label']!=a['true_label']),
              'new_class_wrong':int(z['predicted_label']!=z['actual_label']),
              'old_predicted_image':a['predicted_image'],'new_predicted_image':z['predicted_image'],
              'old_image_wrong':int(a['predicted_image']!=a['true_image']),
              'new_image_wrong':int(z['predicted_image']!=z['image'])})
    common=set.intersection(*sets)
    old_bad={i for i,r in old.items() if r['predicted_label']!=r['true_label']}
    summary={'baseline':'D061 waveform subject0 seed17 epoch70',
       'current':'D066 masked-frequency to image latent fine-tuning subject0 epoch70 seeds17/23/41',
       'identical_test_trials_images_labels':True,'test_n':1600,'pairs':results,
       'class_wrong_all_three_current':len(common),'class_wrong_old_and_all_three_current':len(common&old_bad),
       'definition':'Jaccard=both_wrong/(old_wrong union new_wrong); conditional rates supplied separately',
       'new_inference_runs':0}
    (out/'overlap.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'per_image_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    for d in results:
        for kind in ['class','image']:d[kind].pop('error_ids')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
