import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d094_multiscale_cnn import (
    AntiAliasResample250,
    CHANNELS,
    D094Stage1Classifier,
    SEGMENTS,
    SPATIAL_FEATURES,
    TARGET_SAMPLES,
    TEMPORAL_KERNEL_SAMPLES,
)


def test_resampler_and_architecture_shapes():
    model = D094Stage1Classifier(dropout=0.0)
    raw = torch.randn(2, CHANNELS, 400)

    resampled = model.resampler(raw)
    assert resampled.shape == (2, CHANNELS, TARGET_SAMPLES)
    assert TEMPORAL_KERNEL_SAMPLES == (9, 17, 33, 65)
    assert [branch.groups for branch in model.classifier.temporal_branches] == [CHANNELS] * 4
    assert model.classifier.spatial.kernel_size == (CHANNELS, 1)
    assert model.classifier.spatial.out_channels == SPATIAL_FEATURES

    features = model.classifier.temporal_spatial_features(resampled)
    statistics = model.classifier.segment_statistics(features)
    assert features.shape == (2, SPATIAL_FEATURES, TARGET_SAMPLES)
    assert statistics.shape == (2, SEGMENTS * SPATIAL_FEATURES * 2)
    assert model(raw).shape == (2, 80)


def test_depthwise_blocks_preserve_time_and_training_updates():
    torch.manual_seed(17)
    model = D094Stage1Classifier(dropout=0.0)
    assert all(block.depthwise.groups == SPATIAL_FEATURES for block in model.classifier.temporal_blocks)
    assert all(block.depthwise.stride == (1,) for block in model.classifier.temporal_blocks)

    raw = torch.randn(2, CHANNELS, 400)
    target = torch.tensor([3, 7])
    before = model.classifier.head[-1].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(raw), target)
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.head[-1].weight)


def test_fixed_resampler_rejects_wrong_shape():
    resampler = AntiAliasResample250()
    try:
        resampler(torch.randn(1, CHANNELS, 399))
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong input length was accepted")
