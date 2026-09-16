"""D063 fixed local frequency probe; default mode performs CPU checks only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
import torch
from torch import nn
from eegdecoding.data import ensure_trial_manifest, load_part
from eegdecoding.gpu_lock import AtomicDirectoryLock, local_wddm_snapshot

ARMS = ['raw-only', 'frequency-only', 'raw+frequency']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_hash(model, prefixes):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith(prefixes):
            h.update(name.encode())
            h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def frequency(x):
    window = torch.hann_window(400, periodic=True, dtype=x.dtype, device=x.device)
    fft = torch.fft.rfft((x - x.mean(-1, keepdim=True)) * window, n=400)
    psd = 2 * fft[..., 1:33].abs().square() / (1000 * window.square().sum())
    return psd.clamp_min(1e-30).log10()

def fit_stats(x, train, raw=False):
    dims = (0, 2) if raw else (0,)
    subset = x[train]
    return subset.mean(dims, keepdim=True), subset.std(dims, correction=0, keepdim=True).clamp_min(1e-6)

def order(seed, epoch, n=2400):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100000 + epoch))

def learning_rate(epoch):
    if epoch <= 3:
        return 3e-5 + (epoch - 1) * (3e-4 - 3e-5) / 2
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (epoch - 3) / 67))

class Probe(nn.Module):
    def __init__(self, arm, coordinates, seed):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        # All common modules precede arm-specific random draws.
        torch.manual_seed(seed)
        self.register_buffer('coordinates', coordinates.clone())
        self.electrode = nn.Embedding(62, 128)
        self.position = nn.Linear(3, 128)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            128, 4, 384, .1, activation='gelu', batch_first=True), 2, enable_nested_tensor=False)
        self.head = nn.Linear(128, 80)
        def branch(width, branch_seed):
            torch.manual_seed(branch_seed)
            return nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1))
        self.raw = branch(400, seed + 1000) if arm != 'frequency-only' else None
        self.freq = branch(32, seed + 2000) if arm != 'raw-only' else None
        torch.manual_seed(seed + 3000)

    def forward(self, raw, freq):
        tokens = self.raw(raw) if self.raw is not None else self.freq(freq)
        if self.raw is not None and self.freq is not None:
            tokens = (tokens + self.freq(freq)) / math.sqrt(2)
        tokens = tokens + self.electrode.weight + self.position(self.coordinates)
        return self.head(self.encoder(tokens).mean(1))

def data(cfg):
    with (ROOT / cfg['manifest']).open(newline='', encoding='utf-8') as f:
        rows = sorted([r for r in csv.DictReader(f) if int(r['subject']) == 0], key=lambda r: int(r['source_index']))
    assert len(rows) == 4000 and len({r['trial_id'] for r in rows}) == 4000
    archive = load_part(cfg['archive'])
    original = [(i, item) for i, item in enumerate(archive['dataset']) if int(item['subject']) == 0]
    assert [i for i, _ in original] == [int(r['source_index']) for r in rows]
    counts = {}
    raw = torch.empty(4000, 62, 400)
    for i, r in enumerate(rows):
        assert r['source_file'] == 'EEG-ImageNet_1.pth'
        item = archive['dataset'][int(r['source_index'])]
        assert int(item['subject']) == 0 and str(item['image']) == r['image'] and str(item['label']) == r['label']
        label = int(r['label_index'])
        assert str(archive['labels'][label]) == r['label']
        position = counts.get(label, 0)
        assert int(r['position_in_class_block']) == position
        counts[label] = position + 1
        r['d063_split'] = 'train' if position < 30 else 'test'
        raw[i] = item['eeg_data'][:, 40:440].float()
    assert counts == dict.fromkeys(range(80), 50) and torch.isfinite(raw).all()
    for label in range(80):
        assert len({r['image'] for r in rows if int(r['label_index']) == label}) == 50
    assert not ({r['image'] for r in rows if r['d063_split'] == 'train'} &
                {r['image'] for r in rows if r['d063_split'] == 'test'})
    train = torch.tensor([i for i, r in enumerate(rows) if r['d063_split'] == 'train'])
    test = torch.tensor([i for i, r in enumerate(rows) if r['d063_split'] == 'test'])
    assert (len(train), len(test)) == (2400, 1600)
    with (ROOT / cfg['channel_map']).open(newline='', encoding='utf-8') as f:
        channels = sorted(csv.DictReader(f), key=lambda r: int(r['tensor_index']))
    assert [int(r['tensor_index']) for r in channels] == list(range(62))
    coordinates = torch.tensor([[float(r[f'{axis}_m_template_fit']) for axis in 'xyz'] for r in channels])
    coordinates -= coordinates.mean(0)
    coordinates /= torch.pdist(coordinates).median()
    assert torch.isfinite(coordinates).all()
    return rows, raw, frequency(raw), torch.tensor([int(r['label_index']) for r in rows]), train, test, coordinates

def prepared(arm, raw, freq, train):
    stats = {}
    features = []
    for name, x, enabled in [('raw', raw, arm != 'frequency-only'), ('frequency', freq, arm != 'raw-only')]:
        if enabled:
            mean, scale = fit_stats(x, train, raw=name == 'raw')
            stats[name] = {'mean': mean, 'scale': scale}
            features.append((x - mean) / scale)
        else:
            features.append(None)
    return features, stats

@torch.inference_mode()
def evaluate(model, features, labels, indices, rows, path):
    model.eval()
    logits = torch.cat([model(*[x[idx] if x is not None else None for x in features]).cpu()
                        for idx in indices.split(64)])
    y = labels[indices].cpu()
    loss = nn.functional.cross_entropy(logits, y, reduction='none')
    pred = logits.argmax(1)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial_id', 'source_index', 'image', 'label', 'prediction', 'loss', 'correct'])
        writer.writeheader()
        for k, idx in enumerate(indices.cpu().tolist()):
            writer.writerow(dict(trial_id=rows[idx]['trial_id'], source_index=rows[idx]['source_index'],
                                 image=rows[idx]['image'], label=int(y[k]), prediction=int(pred[k]),
                                 loss=float(loss[k]), correct=int(pred[k] == y[k])))
    torch.save({'logits': logits, 'labels': y, 'trial_ids': [rows[i]['trial_id'] for i in indices.cpu().tolist()]}, path.with_suffix('.pt'))
    return {'accuracy': float((pred == y).float().mean()), 'loss': float(loss.mean()), 'n': len(y)}

def train_one(cfg, arm, seed, loaded):
    rows, raw, freq, labels, train, test, coordinates = loaded
    out = ROOT / cfg['run_dir'] / f'{arm}-s{seed}'
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    features, stats = prepared(arm, raw, freq, train)
    torch.save(stats, out / 'normalization.pt')
    features = [x.cuda() if x is not None else None for x in features]
    labels, train, test = labels.cuda(), train.cuda(), test.cuda()
    model = Probe(arm, coordinates, seed).cuda()
    initialization_hashes = {name: state_hash(model, prefixes) for name, prefixes in {
        'transformer_head': ('encoder.', 'head.'), 'raw': ('raw.',), 'frequency': ('freq.',)}.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    for epoch in range(1, 71):
        snapshot = local_wddm_snapshot(cfg['gpu_uuid'])
        foreign = [p for p in snapshot['existing_processes'] if p['type'] == 'C' and p['pid'] != os.getpid()]
        if foreign:
            raise RuntimeError(f'External compute processes detected: {foreign}')
        epoch_start = time.perf_counter()
        model.train()
        permutation = order(seed, epoch)
        assert len(permutation.unique()) == 2400
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(epoch)
        total_loss = total_correct = total_seen = steps = 0
        for indices in train[permutation.to(train.device)].split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*[x[indices] if x is not None else None for x in features])
            loss = nn.functional.cross_entropy(logits, labels[indices])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip'], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(indices)
            total_correct += (logits.argmax(1) == labels[indices]).sum().item()
            total_seen += len(indices)
            steps += 1
        torch.cuda.synchronize()
        record = dict(epoch=epoch, loss=total_loss / total_seen, accuracy=total_correct / total_seen,
                      coverage=total_seen, unique_trials=len(permutation.unique()), steps=steps,
                      sampler_sha256=hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
                      lr=learning_rate(epoch), wall_seconds=time.perf_counter() - epoch_start)
        with (out / 'epochs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(dict(arm=arm, seed=seed, **record)), flush=True)
        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'config': cfg, 'arm': arm, 'seed': seed, 'normalization': stats}, out / f'epoch{epoch:03d}.pt')
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - training_start
    eval_start = time.perf_counter()
    metrics = {split: evaluate(model, features, labels, ids, rows, out / f'{split}_predictions.csv')
               for split, ids in [('train', train), ('test', test)]}
    torch.cuda.synchronize()
    result = dict(arm=arm, seed=seed, parameters=sum(p.numel() for p in model.parameters()), **metrics,
                  train_seconds=train_seconds, evaluation_seconds=time.perf_counter() - eval_start,
                  wall_seconds=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(0),
                  gpu_uuid=cfg['gpu_uuid'], initialization_hashes=initialization_hashes,
                  precision='FP32; TF32 disabled', total_optimizer_steps=70 * 38)
    dump(out / 'metrics.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Only after explicit main-agent approval')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/d063_local_frequency_probe.json').read_text())
    ensure_trial_manifest()
    report = ROOT / cfg['report_dir']
    report.mkdir(parents=True, exist_ok=True)
    for key in ['TEMP', 'TMP', 'TMPDIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH']:
        path = report / 'cache' / key
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    torch.set_num_threads(4)
    assert cfg['arms'] == ARMS and cfg['seeds'] == [17, 23, 41] and cfg['epochs'] == 70
    assert cfg['batch_size'] == 64 and cfg['sample_rate'] == 1000 and cfg['crop'] == [40, 440] and cfg['subject'] == 0
    assert (cfg['lr'], cfg['min_lr'], cfg['weight_decay'], cfg['clip'], cfg['dropout']) == (3e-4, 3e-5, 1e-3, 1., .1)
    loaded = data(cfg)
    rows, raw, freq, labels, train, test, coordinates = loaded
    with (report / 'split.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    checks = {'status': 'CPU checks passed; awaiting main-agent review', 'training_started': False,
              'train': len(train), 'test': len(test), 'bins_hz': [2.5 * i for i in range(1, 33)], 'arms': {}}
    common = None
    for arm in ARMS:
        features, stats = prepared(arm, raw, freq, train)
        model = Probe(arm, coordinates, 17).eval()
        with torch.inference_mode():
            logits = model(*[x[:2] if x is not None else None for x in features])
        assert logits.shape == (2, 80) and torch.isfinite(logits).all()
        current = {k: v for k, v in model.state_dict().items() if not k.startswith(('raw.', 'freq.'))}
        if common is not None:
            assert all(torch.equal(current[k], common[k]) for k in common)
        common = current
        checks['arms'][arm] = {'parameters': sum(p.numel() for p in model.parameters()), 'forward_shape': list(logits.shape)}
    checks['hashes'] = {name: digest(ROOT / name) for name in [cfg['manifest'], cfg['channel_map'],
                         'config/d063_local_frequency_probe.json', 'scripts/run_d063_local_frequency_probe.py']}
    checks['raw_float32_sha256'] = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    checks['environment'] = {'python': sys.version, 'torch': str(torch.__version__), 'platform': platform.platform()}
    checks['initialization_hashes'] = {}
    for seed in cfg['seeds']:
        hashes = {}
        for arm in ARMS:
            model = Probe(arm, coordinates, seed)
            hashes[arm] = {name: state_hash(model, prefixes) for name, prefixes in {
                'common': ('encoder.', 'head.', 'electrode.', 'position.', 'coordinates'),
                'transformer_head': ('encoder.', 'head.'), 'raw': ('raw.',), 'frequency': ('freq.',)}.items()}
        assert len({h['common'] for h in hashes.values()}) == 1
        assert hashes['raw-only']['raw'] == hashes['raw+frequency']['raw']
        assert hashes['frequency-only']['frequency'] == hashes['raw+frequency']['frequency']
        checks['initialization_hashes'][str(seed)] = hashes
    dump(report / 'cpu_checks.json', checks)
    print(json.dumps(checks), flush=True)
    if not args.train:
        return
    assert platform.system() == 'Windows'
    # Both historical local lock namespaces are respected, never stolen.
    with ExitStack() as stack:
        for i in range(3):
            snapshot = local_wddm_snapshot(cfg['gpu_uuid'])
            dump(report / f'gpu_preflight_{i}.json', snapshot)
            if not snapshot['eligible']:
                raise RuntimeError(f'GPU is not idle: {snapshot}')
            if i < 2:
                time.sleep(2)
        for root in cfg['lock_roots']:
            for name in ['queue-slot-0.lock', f"gpu-{cfg['gpu_uuid']}.lock"]:
                lock = AtomicDirectoryLock(ROOT / root / name, {'run_id': 'd063-local-frequency',
                    'holder_pid': os.getpid(), 'gpu_uuid': cfg['gpu_uuid'], 'host': platform.node()})
                lock.acquire()
                stack.callback(lock.release)
        snapshot = local_wddm_snapshot(cfg['gpu_uuid'])
        dump(report / 'gpu_inside_lock.json', snapshot)
        if not snapshot['eligible']:
            raise RuntimeError('GPU eligibility changed inside lock')
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_uuid']
        assert torch.cuda.device_count() == 1 and '4060' in torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        results = []
        for seed in cfg['seeds']:
            for arm in ARMS:
                results.append(train_one(cfg, arm, seed, loaded))
                torch.cuda.empty_cache()
                dump(report / 'completed_runs.json', results)
        paired = []
        for seed in cfg['seeds']:
            baseline = next(r for r in results if r['seed'] == seed and r['arm'] == 'raw-only')
            for arm in ARMS[1:]:
                other = next(r for r in results if r['seed'] == seed and r['arm'] == arm)
                predictions = []
                for selected in ['raw-only', arm]:
                    with (ROOT / cfg['run_dir'] / f'{selected}-s{seed}' / 'test_predictions.csv').open(newline='') as f:
                        predictions.append(list(csv.DictReader(f)))
                assert [r['trial_id'] for r in predictions[0]] == [r['trial_id'] for r in predictions[1]]
                raw_only_correct = sum(a['correct'] == '1' and b['correct'] == '0' for a, b in zip(*predictions))
                other_only_correct = sum(a['correct'] == '0' and b['correct'] == '1' for a, b in zip(*predictions))
                paired.append({'seed': seed, 'arm': arm, 'test_accuracy_delta': other['test']['accuracy'] - baseline['test']['accuracy'],
                               'raw_only_correct': raw_only_correct, 'other_only_correct': other_only_correct})
        aggregates = {}
        for arm in ARMS[1:]:
            delta = torch.tensor([r['test_accuracy_delta'] for r in paired if r['arm'] == arm], dtype=torch.float64)
            aggregates[arm] = {'mean_delta': delta.mean().item(), 'sample_std_delta': delta.std(correction=1).item()}
        arm_summary = {}
        for arm in ARMS:
            arm_summary[arm] = {}
            for split in ['train', 'test']:
                for metric in ['accuracy', 'loss']:
                    values = torch.tensor([r[split][metric] for r in results if r['arm'] == arm], dtype=torch.float64)
                    arm_summary[arm][f'{split}_{metric}'] = {'mean': values.mean().item(), 'sample_std': values.std(correction=1).item()}
        dump(report / 'summary.json', {'runs': results, 'paired_vs_raw': paired,
             'paired_seed_summary': aggregates, 'arm_seed_summary': arm_summary,
             'scope': 'One subject, fixed classification probe; no RAEv2 or 16-subject inference.'})

if __name__ == '__main__':
    main()
