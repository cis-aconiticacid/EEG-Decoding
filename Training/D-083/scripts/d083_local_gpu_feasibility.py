"""Local full-logical-batch capacity probe with disposable synthetic inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from d081_joint_waveform_frequency import D081LargeModel, parameter_audit  # noqa: E402
from d081_sampling import synchronous_temporal_mask  # noqa: E402
from d081_training import gradient_cache_latent_step, hidden_patch_mse  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--physical", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = D081LargeModel(seed=17, activation_checkpointing=True).cuda()
    audit = parameter_audit(model)
    # Local execution transfers only logical-batch targets to CUDA, not the full bank.
    torch.cuda.synchronize()
    stage_a_start = time.perf_counter()
    print(json.dumps({"phase": "capacity_probe_stage_a", "pid": os.getpid(), "device": torch.cuda.get_device_name(), "parameters": audit}), flush=True)
    model.stage_a(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    before = model.encoder.wave_patch[0].weight.detach().clone()
    for dataset, channels, count, weight in (("imagenet", 62, 40, .4), ("things", 63, 30, .3), ("things", 63, 15, .3)):
        for start in range(0, count, args.physical):
            size = min(args.physical, count - start)
            waveform = torch.randn(size, channels, 400, device="cuda")
            frequency_mean = torch.zeros(channels, 32, device="cuda")
            frequency_scale = torch.ones(channels, 32, device="cuda")
            mask = synchronous_temporal_mask(size, .5, torch.Generator(device="cuda").manual_seed(81000 + start + count), "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.encoder.reconstruct(waveform, dataset, mask, frequency_mean, frequency_scale)
            (hidden_patch_mse(prediction, waveform, mask) * weight * size / count).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    stage_a_seconds = time.perf_counter() - stage_a_start
    stage_a_peak = torch.cuda.max_memory_allocated() / 2**30
    stage_a_update = float((model.encoder.wave_patch[0].weight.detach() - before).abs().max())
    del optimizer
    model.zero_grad(set_to_none=True)
    del prediction, waveform
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"phase": "capacity_probe_stage_a_complete", "seconds": stage_a_seconds, "peak_allocated_gib": stage_a_peak, "weight_update": stage_a_update}), flush=True)

    torch.cuda.synchronize()
    stage_b_start = time.perf_counter()
    model.stage_b(); optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    optimizer.zero_grad(set_to_none=True)
    waveforms = [(torch.randn(40, 62, 400), "imagenet"), (torch.randn(30, 63, 400), "things")]
    forwards = []
    for cpu, dataset in waveforms:
        for start in range(0, len(cpu), args.physical):
            part = cpu[start:start + args.physical]
            channels = part.shape[1]
            def forward(part=part, dataset=dataset, channels=channels):
                value = part.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return model(value, dataset, torch.zeros(channels, 32, device="cuda"),
                                 torch.ones(channels, 32, device="cuda"))
            forwards.append(forward)
    target = torch.randn(70, 256, 1024, device="cuda")
    ids = torch.arange(70, device="cuda")
    before = model.image_head.projector[-1].weight.detach().clone()
    loss, parts = gradient_cache_latent_step(forwards, target, ids)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    stage_b_seconds = time.perf_counter() - stage_b_start
    stage_b_update = float((model.image_head.projector[-1].weight.detach() - before).abs().max())
    result = {"status": "PASS", "pid": os.getpid(), "physical_batch": args.physical,
              "parameter_audit": audit, "stage_a_update": stage_a_update,
              "stage_b_loss": float(loss), "stage_b_parts": {k: float(v) if torch.is_tensor(v) else v for k, v in parts.items()},
              "stage_b_update": stage_b_update, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
              "stage_a_seconds_one_logical_update": stage_a_seconds,
              "stage_b_seconds_one_logical_update": stage_b_seconds,
              "stage_a_peak_allocated_gib": stage_a_peak,
              "includes_formal_target_residency_gib": 0,
              "device": torch.cuda.get_device_name(),
              "torch_version": torch.__version__,
              "scope": "synthetic full logical updates, no input IO or evaluation, not formal training",
              "formal_state_reused": False}
    if stage_a_update <= 0 or stage_b_update <= 0:
        raise AssertionError(result)
    print(json.dumps(result), flush=True)
    output = ROOT / "reports/d083_local_feasibility"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    measurements = [main() for _ in range(3)]
    output = ROOT / "reports/d083_local_feasibility"
    (output / f"physical_{measurements[0]['physical_batch']}_repeats.json").write_text(
        json.dumps(measurements, indent=2), encoding="utf-8")

