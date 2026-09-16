"""Read-only T1 fitted-SVM diagnosis; no fitting or T3 follow-up."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import csv
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/a077_pair_tasks/T1_support_diagnostic'
OUT.mkdir(exist_ok=True)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

bundle = joblib.load(ROOT / 'artifacts/a077_pair_features/models/T1/svm.joblib')
clf = bundle['model']
x = bundle['x_trainval'].astype(np.float64)
sv = x[clf.support_]
alpha = clf.dual_coef_[0]
gamma = bundle['gamma']
pair_lookup = {r['pair_id']: r for r in read(ROOT / 'reports/a077_pair_tasks/T1_pairs.csv')}
rows = [pair_lookup[i] for i in bundle['trainval_pair_ids']]
y = np.array([int(r['target']) for r in rows])
sv_mask = np.zeros(len(x), dtype=bool)
sv_mask[clf.support_] = True

def kernel(q):
    d = np.maximum((q*q).sum(1)[:, None] + (sv*sv).sum(1)[None, :] - 2*q@sv.T, 0)
    return np.exp(-gamma*d)

with threadpool_limits(limits=4):
    scores = np.concatenate([kernel(x[i:i+128])@alpha + clf.intercept_[0] for i in range(0, len(x), 128)])
    rng = np.random.default_rng(77019)
    # Balanced 512 fitting-set probes; no new use of test-set feature rankings.
    probe = np.concatenate([rng.choice(np.flatnonzero(y == t), 256, replace=False) for t in (0, 1)])
    q = x[probe]
    weighted = kernel(q)*alpha
    grad = 2*gamma*(weighted@sv - q*weighted.sum(1)[:, None])
    checks = []
    for j in (0, 79, 777, 1983):
        p, m = q[:1].copy(), q[:1].copy()
        p[0,j] += 1e-4
        m[0,j] -= 1e-4
        fd = ((kernel(p)@alpha - kernel(m)@alpha)/2e-4)[0]
        checks.append(float(abs(fd-grad[0,j])))
    assert max(checks) < 1e-6, checks

margin = (2*y-1)*scores
abs_alpha = np.abs(alpha)
bounded = np.isclose(abs_alpha, clf.C, atol=1e-6, rtol=1e-6)
summary = dict(task='T1', scope='512 balanced train+validation probes; local standardized-pair-feature gradient, not causal localization',
               fit_pairs=len(x), support_count=len(sv), support_fraction=float(sv_mask.mean()),
               bounded_at_C=int(bounded.sum()), free_support=int((~bounded).sum()),
               negative_fit_margin=int((margin<0).sum()), fit_accuracy=float(((scores>=0)==y).mean()),
               top_1pct_support_abs_coefficient_share=float(np.sort(abs_alpha)[-max(1, int(len(alpha)*.01)):].sum()/abs_alpha.sum()),
               finite_difference_max_error=max(checks))
support_rows = []
for idx, a in zip(clf.support_, alpha):
    r = dict(rows[int(idx)])
    r.update(dual_coefficient=float(a), signed_margin=float(margin[idx]), bounded_at_C=bool(np.isclose(abs(a), clf.C)))
    support_rows.append(r)
save('support_margins.csv', support_rows)

channels = sorted(read(ROOT / 'config/channel_map.csv'), key=lambda r:int(r['tensor_index']))
assert len(channels) == 62
energy = np.mean(grad.reshape(-1,62,32)**2, axis=0)
energy /= energy.sum()
freq_rows = [dict(frequency_hz=(j+1)*2.5, gradient_energy_share=float(energy[:,j].sum())) for j in range(32)]
ch_rows = [dict(tensor_index=j, channel=channels[j]['canonical_name'], gradient_energy_share=float(energy[j].sum())) for j in range(62)]
save('frequency_gradient.csv', freq_rows)
save('electrode_gradient.csv', ch_rows)
save('electrode_frequency_gradient.csv', [dict(tensor_index=i, channel=channels[i]['canonical_name'], frequency_hz=(j+1)*2.5, gradient_energy_share=float(energy[i,j])) for i in range(62) for j in range(32)])
subject_rows = []
for s in range(16):
    mask = np.array([s in (int(r['subject_a']), int(r['subject_b'])) for r in rows])
    subject_rows.append(dict(subject=s, fit_pairs_involving_subject=int(mask.sum()), support_pairs=int((mask&sv_mask).sum()), support_rate=float(sv_mask[mask].mean())))
save('subject_support_rates.csv', subject_rows)
summary['top_frequencies'] = sorted(freq_rows,key=lambda r:r['gradient_energy_share'],reverse=True)[:8]
summary['top_electrodes'] = sorted(ch_rows,key=lambda r:r['gradient_energy_share'],reverse=True)[:8]
summary['frequency_top8_share'] = float(sum(r['gradient_energy_share'] for r in summary['top_frequencies']))
summary['electrode_top8_share'] = float(sum(r['gradient_energy_share'] for r in summary['top_electrodes']))
summary['channel_identity_note'] = 'Names follow existing official-RGNN inferred tensor-axis contract, not embedded archive metadata.'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(11,10))
im=ax.imshow(energy*100,aspect='auto',origin='upper',cmap='magma')
ax.set_yticks(range(62), [c['canonical_name'] for c in channels],fontsize=6)
ax.set_xticks(range(1,32,2), [str((j+1)*2.5) for j in range(1,32,2)],fontsize=8)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Electrode (inferred axis labels)')
ax.set_title('T1 same-person SVM: local gradient energy\n512 fitting-set probes; standardized pair differences; not causal importance')
fig.colorbar(im, ax=ax,label='Share of total squared gradient (%)')
fig.tight_layout()
fig.savefig(OUT/'sensitivity.png',dpi=160)
print(json.dumps(summary,indent=2),flush=True)
