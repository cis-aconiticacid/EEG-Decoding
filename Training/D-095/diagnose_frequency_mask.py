"""Frozen-checkpoint spectral-input occlusion; retrospective, not model selection."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from model import D095HybridClassifier, aligned_hann_log_power, frequency_bin_centres
from run import atomic_json, load_coordinates, sha256
from eegdecoding.subject_data import load_subject_data


@torch.inference_mode()
def main():
    cfg = json.loads((HERE / 'config/config.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != cfg['gpu_uuid'] or not torch.cuda.is_available():
        raise RuntimeError('Run using the configured GPU lock')
    torch.set_num_threads(4)
    checkpoint_path = HERE / cfg['run_dir'] / 'epoch100.pt'
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    coords, _ = load_coordinates(cfg)
    model = D095HybridClassifier(coords, classes=cfg['classes'], dropout=cfg['block_dropout'],
                                 head_dropout=cfg['head_dropout']).cuda().eval()
    model.load_state_dict(ckpt['model'], strict=True)
    data = load_subject_data(cfg)
    train_idx, test_idx = data['official_train'], data['test']
    x = data['waveform'][test_idx]
    y = data['labels'][test_idx].cpu()
    centres = frequency_bin_centres()
    total = torch.zeros(62, 16, len(centres), device='cuda')
    for ids in train_idx.split(64):
        total += aligned_hann_log_power(model.standardize(data['waveform'][ids].cuda())).sum(0)
    replacement = total / len(train_idx)
    active = {'indices': [], 'mode': 'zero'}

    def occlude(_module, args):
        value = args[0].clone()
        indices = active['indices']
        if indices:
            value[..., indices] = 0 if active['mode'] == 'zero' else replacement[..., indices]
        return (value,)

    def predict():
        values = []
        for batch in x.split(64):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values.append(model(batch.cuda()).float().cpu())
        return torch.cat(values)

    baseline = predict()
    handle = model.frequency_projection.register_forward_pre_hook(occlude)
    torch.testing.assert_close(predict(), baseline, rtol=0, atol=0)
    cases = [('all', list(range(len(centres))))]
    cases += [(f'{lo}-{hi}Hz', torch.where((centres >= lo) & (centres < hi))[0].tolist())
              for lo, hi in [(12, 28), (28, 44), (44, 60), (60, 80)]]
    cases += [(f'bin_{hz:g}Hz', [i]) for i, hz in enumerate(centres.tolist())]
    base_pred = baseline.argmax(1)
    base_prob = baseline.softmax(1)
    base_loss = F.cross_entropy(baseline, y, reduction='none')
    base_correct = base_pred.eq(y)
    rows, arrays = [], {'baseline_logits': baseline.numpy(), 'labels': y.numpy(),
                        'test_indices': test_idx.cpu().numpy()}
    output = HERE / cfg['report_dir'] / 'frequency_mask_diagnostic.json'
    report = {
        'status': 'running', 'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': sha256(checkpoint_path), 'script_sha256': sha256(Path(__file__)),
        'model_sha256': sha256(HERE / 'model.py'), 'n': len(y),
        'baseline_accuracy': float(base_correct.float().mean()),
        'baseline_loss': float(base_loss.mean()), 'frequency_bins_hz': centres.tolist(),
        'protocol': 'Frozen epoch100, all electrodes and time tokens; mask log1p power before Linear. Waveform intact; gate recomputed. Mean replacement is training-only per electrode/time/bin.',
        'limitations': 'Retrospective official-test sensitivity, not model selection or physiological importance. Zeroing may be out of distribution. FFT bins are correlated; 97ms window does not resolve independent 3.90625Hz bands.',
        'rows': rows,
    }
    try:
        for name, indices in cases:
            for mode in ('zero', 'train_mean'):
                active.update(indices=indices, mode=mode)
                logits = predict()
                pred, prob = logits.argmax(1), logits.softmax(1)
                row = {
                    'name': name, 'mode': mode, 'bins_hz': centres[indices].tolist(),
                    'accuracy': float(pred.eq(y).float().mean()),
                    'accuracy_drop_pp': float((base_correct.float() - pred.eq(y).float()).mean() * 100),
                    'loss': float(F.cross_entropy(logits, y)),
                    'loss_increase': float((F.cross_entropy(logits, y, reduction='none') - base_loss).mean()),
                    'prediction_flip_rate': float(pred.ne(base_pred).float().mean()),
                    'true_class_probability_drop': float((base_prob - prob)[torch.arange(len(y)), y].mean()),
                    'correct_to_wrong': int((base_correct & pred.ne(y)).sum()),
                    'wrong_to_correct': int((~base_correct & pred.eq(y)).sum()),
                }
                rows.append(row)
                arrays[f'{name}_{mode}'] = logits.numpy()
                atomic_json(output, report)
                print(json.dumps(row), flush=True)
    finally:
        handle.remove()
    np.savez_compressed(output.with_suffix('.npz'), **arrays)
    report['status'] = 'complete'
    report['test_forward_rounds'] = len(rows) + 2
    atomic_json(output, report)
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
