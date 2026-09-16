import torch
from model import D095HybridClassifier, frequency_bin_centres


def test_notch_removes_target_preserves_other_dft_components():
    torch.set_num_threads(2)
    model = D095HybridClassifier(torch.ones(62, 3), excluded_band_hz=(18, 28)).eval()
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
    base = D095HybridClassifier(torch.ones(62, 3))
    torch.manual_seed(17)
    excluded = D095HybridClassifier(torch.ones(62, 3), excluded_band_hz=(18, 28))
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, excluded.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(1, 62, 400)
    assert base.filter_input(x) is x
