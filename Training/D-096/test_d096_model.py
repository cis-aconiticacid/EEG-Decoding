import torch

from d096_model import (
    BRANCH_FEATURES,
    CHANNELS,
    CONCAT_FEATURES,
    DEPTH,
    DIM,
    D096CrossAttentionClassifier,
    FFT_SIZE,
    FFN_DIM,
    FREQUENCY_BINS,
    FREQUENCY_WINDOW_SAMPLES,
    FRONTEND_DIM,
    FUSED_FEATURES,
    MULTISCALE_KERNELS,
    PATCHES,
    SPATIAL_BLOCKS,
    aligned_hann_log_power,
    frequency_bin_centres,
    normalize_unit_sphere_coordinates,
)


def coordinates() -> torch.Tensor:
    torch.manual_seed(95)
    value = torch.randn(CHANNELS, 3)
    value[:, 2] += 2.0
    return value


def test_multiscale_frontend_and_full_architecture_shapes():
    model = D096CrossAttentionClassifier(coordinates(), dropout=0.0, head_dropout=0.0)
    waveform = torch.randn(2, CHANNELS, 400)

    assert len(model.frontend.branches) == 3
    assert [layer.kernel_size[0] for layer in model.frontend.branches] == list(MULTISCALE_KERNELS)
    assert all(layer.groups == CHANNELS for layer in model.frontend.branches)
    assert all(layer.out_channels == CHANNELS * BRANCH_FEATURES for layer in model.frontend.branches)
    assert model.frontend.fusion.in_channels == CHANNELS * CONCAT_FEATURES
    assert model.frontend.fusion.out_channels == CHANNELS * FUSED_FEATURES
    assert model.frontend.fusion.groups == CHANNELS
    assert model.frontend.patch_projection.in_features == FUSED_FEATURES
    assert model.frontend.patch_projection.out_features == FRONTEND_DIM

    frontend = model.frontend(waveform)
    tokens, auxiliary = model.tokens(waveform, return_aux=True)
    logits, full_auxiliary = model.forward_standardized(waveform, return_aux=True)
    assert frontend.shape == (2, CHANNELS, PATCHES, FRONTEND_DIM)
    assert tokens.shape == (2, CHANNELS, PATCHES, DIM)
    assert auxiliary["frequency_tokens"].shape == frontend.shape
    assert logits.shape == (2, 80)
    assert full_auxiliary["spatial_pooling_weights"].shape == (2, CHANNELS, PATCHES, 1)
    assert torch.allclose(
        full_auxiliary["spatial_pooling_weights"].sum(dim=1),
        torch.ones(2, PATCHES, 1),
        atol=1e-6,
    )


def test_frequency_windows_bins_and_cross_attention():
    model = D096CrossAttentionClassifier(coordinates(), dropout=0.0, head_dropout=0.0)
    waveform = torch.randn(1, CHANNELS, 400)
    spectrum = aligned_hann_log_power(waveform)
    bins = frequency_bin_centres()

    assert spectrum.shape == (1, CHANNELS, PATCHES, FREQUENCY_BINS)
    assert FREQUENCY_WINDOW_SAMPLES == 97
    assert FFT_SIZE == 256
    assert float(bins.min()) >= 12.0
    assert float(bins.max()) <= 80.0
    assert len(bins) == FREQUENCY_BINS == 17
    assert torch.isfinite(spectrum).all()

    _, auxiliary = model.tokens(waveform, return_aux=True)
    attention = auxiliary["cross_attention_weights"]
    assert attention.shape == (1, CHANNELS, PATCHES, PATCHES)
    assert torch.allclose(attention.sum(-1), torch.ones(1, CHANNELS, PATCHES), atol=1e-6)
    assert torch.isclose(model.cross_modal_fusion.residual_scale.detach(), torch.tensor(0.1))


def test_coordinate_normalization_and_block_contract():
    normalized = normalize_unit_sphere_coordinates(coordinates())
    model = D096CrossAttentionClassifier(normalized, dropout=0.0, head_dropout=0.0)

    assert torch.allclose(model.coordinates.norm(dim=-1), torch.ones(CHANNELS), atol=1e-6)
    assert len(model.blocks) == DEPTH == 4
    spatial = [index + 1 for index, block in enumerate(model.blocks) if block.has_spatial_attention]
    assert spatial == list(SPATIAL_BLOCKS) == [2, 4]
    assert all(block.local_conv.groups == DIM for block in model.blocks)
    assert all(block.ffn[0].in_features == DIM and block.ffn[0].out_features == FFN_DIM for block in model.blocks)
    assert FRONTEND_DIM == DIM == 192
    assert not hasattr(model, "input_projection")
    assert model.cross_modal_fusion.attention.embed_dim == 192
    assert model.cross_modal_fusion.attention.num_heads == 6
    assert model.electrode_embedding.embedding_dim == 192
    assert model.temporal_embedding.embedding_dim == 192
    assert model.coordinate_embedding[-1].out_features == 192
    assert model.head_norm.normalized_shape == (PATCHES * DIM,)
    assert model.classifier.in_features == PATCHES * DIM == 3072


def test_cross_attention_training_updates():
    torch.manual_seed(17)
    model = D096CrossAttentionClassifier(coordinates(), dropout=0.0, head_dropout=0.0)
    waveform = torch.randn(2, CHANNELS, 400)
    target = torch.tensor([3, 7])
    before = model.classifier.weight.detach().clone()
    logits, auxiliary = model.forward_standardized(waveform, return_aux=True)
    loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=0.0)
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=0.01).step()

    assert auxiliary["frequency_tokens"].shape == (2, CHANNELS, PATCHES, FRONTEND_DIM)
    assert auxiliary["cross_attention_weights"].shape == (2, CHANNELS, PATCHES, PATCHES)
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.weight)


def test_wrong_input_shape_is_rejected():
    model = D096CrossAttentionClassifier(coordinates())
    try:
        model(torch.randn(1, CHANNELS, 399))
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong input length was accepted")


def test_unified_width_frequency_branch_receives_gradients():
    model = D096CrossAttentionClassifier(coordinates(), dropout=0.0, head_dropout=0.0)
    logits = model(torch.randn(2, 62, 400))
    torch.nn.functional.cross_entropy(logits, torch.tensor([3, 7])).backward()
    for parameter in (
        model.frontend.patch_projection.weight,
        model.frequency_projection.weight,
        model.cross_modal_fusion.attention.in_proj_weight,
        model.cross_modal_fusion.residual_scale,
        model.electrode_embedding.weight,
        model.temporal_embedding.weight,
        model.coordinate_embedding[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
