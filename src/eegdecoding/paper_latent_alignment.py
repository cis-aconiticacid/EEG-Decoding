"""D060: D057 architecture adapted to the official 40:440 EEG crop."""
import torch
from torch import nn
from .rae_latent_alignment import WaveformRAEEncoder


class PaperWaveformRAEEncoder(WaveformRAEEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_position = nn.Parameter(torch.empty(8, self.time_position.shape[1]))
        nn.init.trunc_normal_(self.time_position, std=.02)

    def forward(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (62, 400):
            raise ValueError("D060 EEG input must be [B,62,400]")
        x = ((raw - self.eeg_mean) / self.eeg_scale).reshape(-1, 62, 8, 50)
        x = self.patch_encoder(x)
        x = x + (self.electrode.weight + self.position(self.coordinates))[None, :, None]
        x = x + self.time_position[None, None]
        for block in self.blocks:
            x = block(x)
        memory = self.final_norm(x.reshape(len(raw), 496, -1))
        q = (self.queries + self.grid_position)[None].expand(len(raw), -1, -1)
        for block in self.query_blocks:
            q = block(q, memory)
        return self.projector(q)
