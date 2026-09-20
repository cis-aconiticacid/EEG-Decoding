import torch
from d098_model import D098CrossAttentionClassifier, frequency_bin_centres


def test_notch_removes_target_preserves_other_dft_components():
    torch.set_num_threads(2)
    model = D098CrossAttentionClassifier(torch.ones(62, 3), excluded_band_hz=(18, 28)).eval()
    t = torch.arange(400) / 1000
    retained = torch.sin(2 * torch.pi * 10 * t) + torch.sin(2 * torch.pi * 75 * t)
    removed = sum(torch.sin(2 * torch.pi * f * t) for f in (20, 22.5, 25, 27.5))
    waveform = (retained + removed).expand(2, 62, 400).clone()
    filtered = model.filter_input(waveform)
    torch.testing.assert_close(filtered, retained.expand_as(filtered), atol=3e-5, rtol=1e-4)
    seen = []
    hook = model.frequency_projection.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
    with torch.no_grad():
        raw = model(waveform)
        preprocessed = model.forward_standardized(model.standardize(waveform))
    hook.remove()
    torch.testing.assert_close(raw, preprocessed)
    bins = frequency_bin_centres()
    assert (seen[0][..., (bins >= 18) & (bins <= 28)] == 0).all()


def test_default_is_identity_and_initialization_is_matched():
    torch.manual_seed(17)
    base = D098CrossAttentionClassifier(torch.ones(62, 3))
    torch.manual_seed(17)
    excluded = D098CrossAttentionClassifier(torch.ones(62, 3), excluded_band_hz=(18, 28))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, excluded.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(1, 62, 400)
    assert base.filter_input(x) is x


def test_factorial_frequency_mask_and_post_cross_position_switch():
    torch.manual_seed(17)
    waveform = torch.randn(1, 62, 400)
    masked = D098CrossAttentionClassifier(
        torch.ones(62, 3),
        dropout=0.0,
        head_dropout=0.0,
        frequency_mask_below_hz=25.0,
        add_position_after_cross_attention=False,
    ).eval()
    seen = []
    hook = masked.frequency_projection.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach())
    )
    with torch.no_grad():
        masked.tokens(waveform)
    hook.remove()

    bins = frequency_bin_centres()
    assert masked.frequency_mask_below_hz == 25.0
    assert masked.add_position_after_cross_attention is False
    assert (seen[0][..., bins < 25.0] == 0).all()
    assert torch.isfinite(seen[0][..., bins >= 25.0]).all()

    torch.manual_seed(17)
    with_position = D098CrossAttentionClassifier(
        torch.ones(62, 3), dropout=0.0, head_dropout=0.0,
        add_position_after_cross_attention=True,
    ).eval()
    without_position = D098CrossAttentionClassifier(
        torch.ones(62, 3), dropout=0.0, head_dropout=0.0,
        add_position_after_cross_attention=False,
    ).eval()
    without_position.load_state_dict(with_position.state_dict())
    with torch.no_grad():
        positioned = with_position.tokens(waveform)
        unpositioned = without_position.tokens(waveform)
    assert not torch.equal(positioned, unpositioned)
