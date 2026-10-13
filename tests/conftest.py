from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pos_mamba import models


class TinySpatialBackbone(nn.Module):
    """Only mocks the documented AutoModel interface, never real MambaVision kernels."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=4, padding=1)

    def forward(self, images):
        spatial = self.conv(images)
        return spatial.mean(dim=(2, 3)), [spatial]


@pytest.fixture
def fake_backbone(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(models.AutoConfig, 'from_pretrained',
                        lambda *a, **k: SimpleNamespace(dim=8, depths=[1]))
    monkeypatch.setattr(models.AutoModel, 'from_pretrained',
                        lambda *a, **k: TinySpatialBackbone())
