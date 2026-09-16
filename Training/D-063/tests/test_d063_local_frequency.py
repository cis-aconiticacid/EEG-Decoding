import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('d063', Path(__file__).resolve().parents[1] / 'scripts/run_d063_local_frequency_probe.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class D063Tests(unittest.TestCase):
    def test_periodogram_density(self):
        from scipy.signal import periodogram
        x = torch.randn(3, 62, 400, generator=torch.Generator().manual_seed(7))
        hz, psd = periodogram(x.numpy(), fs=1000, window='hann', detrend='constant', scaling='density', axis=-1)
        expected = torch.from_numpy(psd[..., 1:33]).clamp_min(1e-30).log10()
        torch.testing.assert_close(m.frequency(x), expected, atol=2e-5, rtol=2e-5)
        self.assertEqual(hz[32], 80)

    def test_peak(self):
        t = torch.arange(400) / 1000
        x = torch.sin(2 * torch.pi * 20 * t)[None, None]
        self.assertEqual(m.frequency(x).argmax().item(), 7)

    def test_train_only_stats(self):
        x = torch.arange(60.).reshape(5, 3, 4)
        ids = torch.tensor([0, 1, 2])
        for raw in [False, True]:
            a = m.fit_stats(x, ids, raw)
            x[3:] = 1e9
            b = m.fit_stats(x, ids, raw)
            for left, right in zip(a, b):
                self.assertTrue(torch.equal(left, right))

    def test_sampler_and_schedule(self):
        for seed in [17, 23, 41]:
            for epoch in range(1, 71):
                p = m.order(seed, epoch)
                self.assertEqual(p.unique().numel(), 2400)
                self.assertTrue(torch.equal(p, m.order(seed, epoch)))
        self.assertAlmostEqual(m.learning_rate(1), 3e-5)
        self.assertAlmostEqual(m.learning_rate(3), 3e-4)
        self.assertAlmostEqual(m.learning_rate(70), 3e-5)

    def test_initialization_and_forward(self):
        torch.set_num_threads(2)
        coords = torch.zeros(62, 3)
        for seed in [17, 23, 41]:
            models = [m.Probe(arm, coords, seed).eval() for arm in m.ARMS]
            states = [model.state_dict() for model in models]
            for key in states[2]:
                if key.startswith('raw.'):
                    self.assertTrue(torch.equal(states[0][key], states[2][key]))
                elif key.startswith('freq.'):
                    self.assertTrue(torch.equal(states[1][key], states[2][key]))
                else:
                    self.assertTrue(torch.equal(states[0][key], states[1][key]))
                    self.assertTrue(torch.equal(states[0][key], states[2][key]))
            with torch.inference_mode():
                for model in models:
                    output = model(torch.zeros(2, 62, 400), torch.zeros(2, 62, 32))
                    self.assertEqual(tuple(output.shape), (2, 80))
                    self.assertTrue(torch.isfinite(output).all())

if __name__ == '__main__':
    unittest.main()
