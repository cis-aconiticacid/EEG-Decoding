"""D064: controlled masked frequency modelling on GPU0, no official test access."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
import torch
from torch import nn
from eegdecoding.data import ensure_trial_manifest, load_part

STOP = False
ARMS = ['mask15', 'mask30', 'mask45', 'mask60', 'mask75', 'schedule15to60']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def thash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)

def frequency(x):
    w = torch.hann_window(400, periodic=True, dtype=x.dtype, device=x.device)
    z = torch.fft.rfft((x - x.mean(-1, keepdim=True)) * w)
    return (2 * z[..., 1:33].abs().square() / (1000 * w.square().sum())).clamp_min(1e-30).log10()

def ratio(arm, epoch):
    if arm == 'schedule15to60':
        return .15 + .45 * min(1., max(0., (epoch - 7) / 42))
    return int(arm.removeprefix('mask')) / 100

def masks(n, r, seed):
    generator = torch.Generator().manual_seed(seed)
    ranking = torch.rand(n, 496, generator=generator).argsort(1)
    out = torch.zeros(n, 496, dtype=torch.bool)
    out.scatter_(1, ranking[:, :int(math.floor(496 * r + .5))], True)
    return out

def order(seed, epoch, n):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100000 + epoch))

def lr(epoch):
    if epoch <= 3:
        return 3e-5 + (epoch - 1) * 2.7e-4 / 2
    return 3e-5 + 1.35e-4 * (1 + math.cos(math.pi * (epoch - 3) / 67))

class MaskedFrequency(nn.Module):
    def __init__(self, coordinates, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer('coordinates', coordinates.clone())
        self.projection = nn.Linear(4, 128)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 128))
        nn.init.normal_(self.mask_token, std=.02)
        self.electrode = nn.Embedding(62, 128)
        self.position = nn.Linear(3, 128)
        self.frequency_position = nn.Embedding(8, 128)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(
            128, 4, 384, .1, activation='gelu', batch_first=True) for _ in range(2)])
        self.output = nn.Linear(128, 4)

    def tokens(self, x, mask):
        h = self.projection(x)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token, h)
        p = ((self.electrode.weight + self.position(self.coordinates))[:, None, :]
             + self.frequency_position.weight[None, :, :]).reshape(496, 128)
        return h + p

    def encode(self, x, mask=None):
        h = self.tokens(x, mask)
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x, mask):
        return self.output(self.encode(x, mask))

def masked_loss(pred, target, mask):
    return (pred - target).square().mean(-1)[mask].mean()

def data(cfg, report):
    with (ROOT / cfg['manifest']).open(newline='', encoding='utf-8') as f:
        all_rows = sorted([r for r in csv.DictReader(f) if int(r['subject']) == 0],
                          key=lambda r: int(r['source_index']))
    assert len(all_rows) == 4000
    rows = []
    for label in range(80):
        group = [r for r in all_rows if int(r['label_index']) == label]
        assert len(group) == 50 and len({r['image'] for r in group}) == 50
        assert [int(r['position_in_class_block']) for r in group] == list(range(50))
    for r in all_rows:
        pos = int(r['position_in_class_block'])
        if pos < 30:
            rows.append(dict(r, d064_split='fit' if pos < 24 else 'validation'))
    assert len(rows) == 2400
    fit = torch.tensor([i for i, r in enumerate(rows) if r['d064_split'] == 'fit'])
    val = torch.tensor([i for i, r in enumerate(rows) if r['d064_split'] == 'validation'])
    assert (len(fit), len(val)) == (1920, 480)
    assert not ({rows[i]['image'] for i in fit.tolist()} & {rows[i]['image'] for i in val.tolist()})
    official_test = {r['image'] for r in all_rows if int(r['position_in_class_block']) >= 30}
    assert not (official_test & {r['image'] for r in rows})
    # Archive storages are mmap'ed. Only the 2400 eligible trial tensors are read.
    archive = load_part(ROOT / cfg['archive'])
    raw = torch.empty(2400, 62, 400)
    for i, r in enumerate(rows):
        item = archive['dataset'][int(r['source_index'])]
        assert r['source_file'] == 'EEG-ImageNet_1.pth'
        assert int(item['subject']) == 0 and str(item['image']) == r['image']
        assert str(item['label']) == r['label'] == str(archive['labels'][int(r['label_index'])])
        raw[i] = item['eeg_data'][:, 40:440].float()
    assert torch.isfinite(raw).all()
    freq = frequency(raw)
    mean = freq[fit].mean(0, keepdim=True)
    std = freq[fit].std(0, correction=0, keepdim=True).clamp_min(1e-6)
    x = ((freq - mean) / std).reshape(2400, 496, 4)
    y = torch.tensor([int(r['label_index']) for r in rows])
    with (ROOT / cfg['channel_map']).open(newline='', encoding='utf-8') as f:
        channels = sorted(csv.DictReader(f), key=lambda r: int(r['tensor_index']))
    assert [int(r['tensor_index']) for r in channels] == list(range(62))
    coords = torch.tensor([[float(r[f'{a}_m_template_fit']) for a in 'xyz'] for r in channels])
    coords -= coords.mean(0)
    coords /= torch.pdist(coords).median()
    assert torch.isfinite(coords).all() and torch.isfinite(x).all()
    contract = {'config': sha(ROOT / 'config/d064_local_mask_ratio.json'),
                'code': sha(__file__), 'loader': sha(PROJECT_ROOT / 'src/eegdecoding/data.py'),
                'manifest': sha(ROOT / cfg['manifest']), 'montage': sha(ROOT / cfg['channel_map']),
                'eligible_raw': thash(raw), 'fit_indices': thash(fit), 'val_indices': thash(val),
                'normalization_mean': thash(mean), 'normalization_std': thash(std)}
    if (report / 'contract.json').exists():
        assert json.loads((report / 'contract.json').read_text()) == contract, 'Contract changed'
    dump(report / 'contract.json', contract)
    save(report / 'normalization.pt', {'mean': mean, 'std': std, 'fit': fit, 'val': val})
    with (report / 'split.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return x, y, fit, val, coords, rows, contract

@torch.no_grad()
def evaluate(model, x, ids, mask):
    model.eval()
    total = 0.
    baseline = 0.
    for positions in torch.arange(len(ids), device=x.device).split(64):
        batch = x[ids[positions]]
        m = mask[positions]
        total += float(masked_loss(model(batch, m), batch, m)) * len(positions)
        baseline += float(masked_loss(torch.zeros_like(batch), batch, m)) * len(positions)
    return {'masked_mse': total / len(ids), 'mean_fill_mse': baseline / len(ids)}

@torch.no_grad()
def embeddings(model, x):
    model.eval()
    return torch.cat([model.encode(b).mean(1) for b in x.split(64)])

def probe(model, x, y, fit, val, seed, out, rows, cfg):
    model.eval().requires_grad_(False)
    before = {k: thash(v) for k, v in model.state_dict().items()}
    features = embeddings(model, x).detach()
    torch.manual_seed(seed + 64000)
    head = nn.Linear(128, 80).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg['probe']['lr'], weight_decay=cfg['probe']['weight_decay'])
    for epoch in range(1, 101):
        for ids in fit[order(seed, epoch, len(fit)).to(x.device)].split(64):
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features[ids]), y[ids])
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
    assert before == {k: thash(v) for k, v in model.state_dict().items()}, 'Encoder changed during probe'
    head.eval()
    result = {}
    for name, ids in [('fit', fit), ('validation', val)]:
        with torch.no_grad():
            logits = head(features[ids]).cpu()
        labels = y[ids].cpu()
        pred = logits.argmax(1)
        accuracy = int((pred == labels).sum()) / len(labels)
        result[name] = {'accuracy': accuracy, 'loss': float(nn.functional.cross_entropy(logits, labels)), 'n': len(ids)}
        save(out / f'{name}_predictions.pt', {'logits': logits, 'labels': labels,
             'trial_ids': [rows[i]['trial_id'] for i in ids.cpu().tolist()]})
    save(out / 'linear_probe.pt', {'model': head.state_dict(), 'config': cfg['probe']})
    return result

def status(report, **kw):
    record = dict(task='D064', pid=os.getpid(), updated_unix=time.time(), **kw)
    dump(report / 'status.json', record)
    print(json.dumps(record), flush=True)

def train_run(cfg, arm, seed, loaded, banks, report, segment_start):
    x, y, fit, val, coords, rows, contract = loaded
    out = ROOT / cfg['run_dir'] / f'{arm}-s{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        result = json.loads((out / 'result.json').read_text())
        assert result['contract'] == contract
        return result
    model = MaskedFrequency(coords, seed).to(x.device)
    initialization_hash = hashlib.sha256(''.join(thash(v) for v in model.state_dict().values()).encode()).hexdigest()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
    start_epoch = 0
    records = []
    ckpts = sorted(out.glob('epoch_*.pt'))
    if ckpts:
        saved = torch.load(ckpts[-1], map_location=x.device, weights_only=False)
        assert saved['contract'] == contract and saved['arm'] == arm and saved['seed'] == seed
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        start_epoch = saved['epoch']
        records = saved['records']
        dump(out / 'epochs.json', records)
    for epoch in range(start_epoch + 1, 71):
        t = time.perf_counter()
        model.train()
        # Independent RNG for sampler, masks and dropout; reproducible checkpoint resume.
        torch.manual_seed(seed * 10000 + epoch)
        perm = order(seed, epoch, len(fit)).to(x.device)
        mask = masks(len(fit), ratio(arm, epoch), seed * 1000000 + epoch).to(x.device)
        for group in opt.param_groups:
            group['lr'] = lr(epoch)
        total = 0.
        seen = steps = 0
        for positions in perm.split(64):
            batch = x[fit[positions]]
            opt.zero_grad(set_to_none=True)
            loss = masked_loss(model(batch, mask[positions]), batch, mask[positions])
            assert torch.isfinite(loss), 'Nonfinite loss'
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            total += float(loss.detach()) * len(positions)
            seen += len(positions)
            steps += 1
        torch.cuda.synchronize()
        assert seen == 1920 and steps == 30 and len(perm.unique()) == 1920
        rec = {'epoch': epoch, 'loss': total / seen, 'coverage': seen, 'steps': steps,
               'sampler_sha256': thash(perm), 'mask_sha256': thash(mask), 'ratio': ratio(arm, epoch),
               'mask_count': int(mask[0].sum()), 'lr': lr(epoch), 'seconds': time.perf_counter() - t}
        if epoch % 10 == 0:
            rec['validation_60'] = evaluate(model, x, val, banks['0.6'])
        records.append(rec)
        dump(out / 'epochs.json', records)
        status(report, phase='pretrain', arm=arm, seed=seed, **rec,
               gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
               features_device=str(x.device))
        if epoch % 10 == 0:
            save(out / f'epoch_{epoch:03d}.pt', dict(epoch=epoch, model=model.state_dict(),
                 optimizer=opt.state_dict(), records=records, contract=contract, arm=arm, seed=seed))
            if STOP or (ROOT / '.runtime/d064_stop_requested').exists():
                raise SystemExit(76)
            if time.monotonic() - segment_start > 1800 and epoch < 70:
                raise SystemExit(75)
    assert len(list(out.glob('epoch_*.pt'))) == 7
    status(report, phase='evaluate_and_probe', arm=arm, seed=seed)
    reconstruction = {r: evaluate(model, x, val, bank) for r, bank in banks.items()}
    probe_result = probe(model, x, y, fit, val, seed, out, rows, cfg)
    result = {'arm': arm, 'seed': seed, 'contract': contract, 'parameters': sum(p.numel() for p in model.parameters()),
              'initialization_hash': initialization_hash, 'reconstruction': reconstruction, 'probe': probe_result,
              'pretrain_seconds': sum(r['seconds'] for r in records), 'checkpoint_count': 7}
    dump(out / 'result.json', result)
    return result

def checks():
    torch.set_num_threads(4)
    model = MaskedFrequency(torch.randn(62, 3), 17).eval()
    x = torch.randn(2, 496, 4)
    m = masks(2, .6, 1)
    changed = x.clone()
    changed[m] += 100
    with torch.no_grad():
        assert torch.equal(model.tokens(x, m), model.tokens(changed, m))
        assert torch.equal(model(x, m), model(changed, m))
    p = torch.randn_like(x, requires_grad=True)
    loss = masked_loss(p, x, m)
    loss.backward()
    assert p.grad[~m].abs().sum() == 0 and p.grad[m].abs().sum() > 0
    assert masks(2, .15, 1).sum(1).tolist() == [74, 74]
    assert (masks(2, .15, 1) & ~m).sum() == 0
    assert ratio('schedule15to60', 1) == .15 and ratio('schedule15to60', 7) == .15
    assert abs(ratio('schedule15to60', 49) - .6) < 1e-9
    assert abs(ratio('schedule15to60', 70) - .6) < 1e-9
    t = torch.arange(400) / 1000
    assert int(frequency(torch.sin(2 * math.pi * 20 * t)).argmax()) == 7
    assert len(order(17, 1, 1920).unique()) == 1920
    print(json.dumps({'checks': 'PASS', 'mask_leakage': False, 'masked_only_gradient': True,
                     'frequency_peak_hz': 20, 'parameters': sum(p.numel() for p in model.parameters())}), flush=True)

def guard(cfg):
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == cfg['gpu_uuid']
    lock = ROOT / cfg['lock_root'] / f"gpu-{cfg['gpu_uuid']}.lock"
    assert lock.is_dir(), 'External UUID lock required'
    # The wrapper writes worker_pid after Popen; allow that short handoff race.
    for _ in range(20):
        for path in lock.glob('*.json'):
            meta = json.loads(path.read_text())
            if meta.get('worker_pid') == os.getpid():
                assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
                return
        time.sleep(.1)
    raise RuntimeError('GPU lock does not identify this worker')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    checks()
    if not (args.train or args.prepare):
        return
    cfg = json.loads((ROOT / 'config/d064_local_mask_ratio.json').read_text())
    ensure_trial_manifest()
    assert cfg['arms'] == ARMS and cfg['epochs'] == 70 and cfg['batch_size'] == 64
    report = ROOT / cfg['report_dir']
    report.mkdir(parents=True, exist_ok=True)
    loaded = data(cfg, report)
    x, y, fit, val, coords, rows, contract = loaded
    banks = {str(r): masks(len(val), r, cfg['validation_bank_seed']) for r in cfg['validation_ratios']}
    dump(report / 'validation_mask_banks.json', {r: thash(m) for r, m in banks.items()})
    save(report / 'validation_mask_banks.pt', banks)
    dump(report / 'preflight.json', {'fit': len(fit), 'validation': len(val), 'official_test_evaluations': 0,
          'shape': list(x.shape), 'contract': contract, 'torch': str(torch.__version__), 'status': 'PASS'})
    if not args.train:
        return
    guard(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    x, y, fit, val = [v.cuda() for v in (x, y, fit, val)]
    loaded = x, y, fit, val, coords, rows, contract
    banks = {r: m.cuda() for r, m in banks.items()}
    segment_start = time.monotonic()
    results = []
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            if STOP or (ROOT / '.runtime/d064_stop_requested').exists():
                return 76
            results.append(train_run(cfg, arm, seed, loaded, banks, report, segment_start))
            dump(report / 'completed_runs.json', results)
            torch.cuda.empty_cache()
            if time.monotonic() - segment_start > 1800:
                return 75
        out = ROOT / cfg['run_dir'] / f'random_control-s{seed}'
        out.mkdir(parents=True, exist_ok=True)
        if not (out / 'result.json').exists():
            status(report, phase='random_encoder_probe', seed=seed)
            model = MaskedFrequency(coords, seed).cuda()
            result = probe(model, x, y, fit, val, seed, out, rows, cfg)
            dump(out / 'result.json', {'seed': seed, 'contract': contract, 'probe': result})
            del model
    aggregates = {}
    for arm in ARMS:
        values = [r['probe']['validation']['accuracy'] for r in results if r['arm'] == arm]
        aggregates[arm] = {'validation_accuracy_mean': statistics.mean(values),
                           'validation_accuracy_sample_std': statistics.stdev(values)}
    dump(report / 'summary.json', {'results': results, 'aggregates': aggregates,
         'selection_split': 'internal validation, exploratory', 'official_test_evaluations': 0})
    dump(ROOT / cfg['run_dir'] / 'completed.json', {'runs': len(results), 'contract': contract})
    status(report, phase='completed', completed_runs=len(results))
    return 0

def stop_handler(*_):
    global STOP
    STOP = True

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    raise SystemExit(main())
