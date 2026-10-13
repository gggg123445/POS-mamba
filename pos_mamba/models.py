"""POS-Mamba: spatial semantics, Haar texture, pixel gate, six tasks and POS.

The backbone uses NVIDIA's AutoModel feature interface, never classifier logits.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

class TaskTowers6(nn.Module):
    def __init__(self, in_dim=512, hidden=64):
        super().__init__()
        self.task_dims = [6, 1, 4, 1, 1, 3]
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden, d),
            )
            for d in self.task_dims
        ])

    def forward(self, x):
        return [tower(x) for tower in self.towers]


class HaarWaveletFreqBranch(nn.Module):
    """Haar-DWT frequency detail branch for long-image inputs."""
    def __init__(
        self,
        use_hh: bool = True,
        out_dim: int = 256,
        dropout: float = 0.1,
        levels: int = 2,
        use_ll: bool = True,
        band_norm: bool = False,
        band_attention: bool = True,
        band_attn_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_hh = bool(use_hh)
        self.levels = max(1, int(levels))
        self.use_ll = bool(use_ll)
        self.band_norm = bool(band_norm)
        self.band_attention = bool(band_attention)
        self.band_attn_temperature = float(band_attn_temperature)
        details_per_level = 3 if self.use_hh else 2
        in_ch = details_per_level * self.levels + (1 if self.use_ll else 0)
        self.num_bands = int(in_ch)
        if self.band_attention:
            self.band_attn = nn.Sequential(
                nn.Linear(self.num_bands, self.num_bands),
                nn.GELU(),
                nn.Linear(self.num_bands, self.num_bands),
            )
        else:
            self.band_attn = None
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, int(out_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def _haar_dwt2(self, x: torch.Tensor):
        # x: [B,1,H,W] -> ll/lh/hl/hh: [B,1,H/2,W/2]
        h = x.shape[-2]
        w = x.shape[-1]
        if (h % 2) != 0:
            x = x[:, :, :-1, :]
        if (w % 2) != 0:
            x = x[:, :, :, :-1]

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, lh, hl, hh

    def _norm_band(self, x: torch.Tensor) -> torch.Tensor:
        if not self.band_norm:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        # The deepest band may contain one pixel; unbiased std would be NaN.
        std = x.std(dim=(2, 3), keepdim=True, unbiased=x.shape[-2] * x.shape[-1] > 1).clamp_min(1e-6)
        return (x - mean) / std

    def forward(self, x_gray: torch.Tensor) -> torch.Tensor:
        # x_gray: [B,1,H,W]
        if min(x_gray.shape[-2:]) < 2 ** self.levels:
            raise ValueError(f'Input sides must be >= {2 ** self.levels} for {self.levels} Haar levels')
        cur = x_gray
        all_bands = []
        target_hw = None
        for _ in range(self.levels):
            ll, lh, hl, hh = self._haar_dwt2(cur)
            bands = [self._norm_band(lh), self._norm_band(hl)]
            if self.use_hh:
                bands.append(self._norm_band(hh))
            if target_hw is None:
                target_hw = bands[0].shape[-2:]
            bands = [
                F.interpolate(b, size=target_hw, mode='bilinear', align_corners=False)
                if b.shape[-2:] != target_hw else b
                for b in bands
            ]
            all_bands.extend(bands)
            cur = ll

        if self.use_ll:
            lln = self._norm_band(cur)
            if target_hw is not None and lln.shape[-2:] != target_hw:
                lln = F.interpolate(lln, size=target_hw, mode='bilinear', align_corners=False)
            all_bands.append(lln)

        freq = torch.cat(all_bands, dim=1)
        if self.band_attention and self.band_attn is not None:
            band_desc = freq.mean(dim=(2, 3))
            tau = max(float(self.band_attn_temperature), 1e-6)
            band_w = torch.softmax(self.band_attn(band_desc) / tau, dim=1)
            freq = freq * band_w.unsqueeze(-1).unsqueeze(-1)
        feat = self.encoder(freq)
        return self.head(feat)


class SpatialGatedFusion(nn.Module):
    """
    空间感知门控融合机制
    在 [B, C, H, W] 维度上计算门控权重，门控形状为 [B,1,H,W]，在通道维广播。
    """

    def __init__(self, main_dim: int, freq_dim: int):
        super().__init__()
        self.freq_to_main = nn.Sequential(
            nn.Conv2d(int(freq_dim), int(main_dim), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(main_dim)),
            nn.ReLU(inplace=True),
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(int(main_dim) * 2, int(main_dim), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(main_dim)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(main_dim), 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, main_spatial: torch.Tensor, freq_spatial: torch.Tensor) -> torch.Tensor:
        if main_spatial.shape[-2:] != freq_spatial.shape[-2:]:
            freq_spatial = F.interpolate(
                freq_spatial,
                size=main_spatial.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        freq_mapped = self.freq_to_main(freq_spatial)
        cat_feat = torch.cat([main_spatial, freq_mapped], dim=1)
        gate = self.gate_conv(cat_feat)
        return (1.0 - gate) * main_spatial + gate * freq_mapped


class HaarWaveletFreqBranchV2(HaarWaveletFreqBranch):
    """保留空间维度的小波频域分支。"""

    def __init__(
        self,
        use_hh: bool = True,
        out_dim: int = 256,
        dropout: float = 0.1,
        levels: int = 2,
        use_ll: bool = True,
        band_norm: bool = False,
        band_attention: bool = True,
        band_attn_temperature: float = 1.0,
    ):
        super().__init__(
            use_hh=use_hh,
            out_dim=out_dim,
            dropout=dropout,
            levels=levels,
            use_ll=use_ll,
            band_norm=band_norm,
            band_attention=band_attention,
            band_attn_temperature=band_attn_temperature,
        )
        # v2: 取消提前池化，输出 [B, out_dim, H, W]
        self.head = nn.Sequential(
            nn.Conv2d(128, int(out_dim), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_dim)),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
        )


class POSMamba(nn.Module):
    """Paper-aligned default model; alternative fusion modes support ablations."""

    def __init__(
        self, model_name='nvidia/MambaVision-T-1K', feature_proj_dim=512,
        towers_hidden=128, use_hlgauss=True, hl_bins_mode='cadrads',
        hl_aux_hidden=256, hl_aux_dropout=0.5, repeat_gray_to_3ch=True,
        use_wavelet_freq_branch=True, wavelet_use_hh=True,
        wavelet_freq_hidden=256, wavelet_fusion_mode='gated',
        wavelet_fusion_alpha=0.2, wavelet_levels=4, wavelet_use_ll=True,
        wavelet_band_norm=False, wavelet_band_attention=True,
        wavelet_band_attn_temperature=1.0, model_revision=None,
    ):
        super().__init__()
        if wavelet_fusion_mode not in {'gated', 'concat', 'sum'}:
            raise ValueError('Unknown fusion mode')
        if hl_bins_mode not in {'cadrads', 'uniform101'}:
            raise ValueError('Unknown histogram bins')
        self.use_hlgauss = bool(use_hlgauss)
        self.repeat_gray_to_3ch = bool(repeat_gray_to_3ch)
        self.wavelet_fusion_mode = wavelet_fusion_mode
        self.wavelet_fusion_alpha = float(wavelet_fusion_alpha)
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, revision=model_revision)
        self.backbone = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True, revision=model_revision)
        # MambaVision expands its channel count at every stage.
        self.main_spatial_dim = int(config.dim * 2 ** (len(config.depths) - 1))
        self.feat_dim = int(feature_proj_dim) if feature_proj_dim > 0 else self.main_spatial_dim
        self.feature_proj = nn.Identity() if self.feat_dim == self.main_spatial_dim else nn.Sequential(
            nn.Linear(self.main_spatial_dim, self.feat_dim), nn.GELU(), nn.Dropout(0.1))
        self.wavelet_branch = None
        if use_wavelet_freq_branch:
            self.wavelet_branch = HaarWaveletFreqBranchV2(
                use_hh=wavelet_use_hh, out_dim=wavelet_freq_hidden,
                levels=wavelet_levels, use_ll=wavelet_use_ll,
                band_norm=wavelet_band_norm, band_attention=wavelet_band_attention,
                band_attn_temperature=wavelet_band_attn_temperature)
            if wavelet_fusion_mode == 'gated':
                self.fusion = SpatialGatedFusion(self.main_spatial_dim, wavelet_freq_hidden)
            elif wavelet_fusion_mode == 'concat':
                self.fusion = nn.Sequential(
                    nn.Conv2d(self.main_spatial_dim + wavelet_freq_hidden, self.main_spatial_dim, 1, bias=False),
                    nn.BatchNorm2d(self.main_spatial_dim), nn.GELU(), nn.Dropout2d(0.1))
            else:
                self.fusion = nn.Sequential(
                    nn.Conv2d(wavelet_freq_hidden, self.main_spatial_dim, 1, bias=False),
                    nn.BatchNorm2d(self.main_spatial_dim), nn.GELU())
        self.towers = TaskTowers6(in_dim=self.feat_dim, hidden=towers_hidden)
        if self.use_hlgauss:
            # Half-integer edges retain the original integer-percentage convention.
            edges = (torch.linspace(-0.5, 100.5, 102) if hl_bins_mode == 'uniform101'
                     else torch.tensor([0., 0.5, 24.5, 49.5, 69.5, 99.5, 100.5]))
            self.register_buffer('hl_edges', edges)
            self.hl_num_bins = len(edges) - 1
            self.hl_head = nn.Sequential(
                nn.Linear(self.feat_dim, hl_aux_hidden), nn.ReLU(inplace=True),
                nn.Dropout(hl_aux_dropout), nn.Linear(hl_aux_hidden, self.hl_num_bins))

    def hl_gauss_to_probs(self, target, sigma=10.0):
        """Gaussian bin masses with all outside mass absorbed by the tail bins."""
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        target = target.clamp(0, 100)
        interior = self.hl_edges[1:-1].to(target)
        cdf = 0.5 * (1 + torch.erf((interior[None, :] - target[:, None]) / (2 ** 0.5 * sigma)))
        return torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]], dim=1)

    def hl_gauss_from_probs(self, probs):
        """Auxiliary percentage estimate, separate from the primary T1 grade head."""
        centers = ((self.hl_edges[:-1] + self.hl_edges[1:]) * 0.5).to(probs)
        return (probs * centers[None, :]).sum(dim=1)

    @staticmethod
    def stenosis_to_cadrads_label(st):
        return 0 if st <= 0 else 1 if st < 25 else 2 if st < 50 else 3 if st < 70 else 4 if st < 100 else 5

    def forward(self, images):
        gray = images if images.shape[1] == 1 else images.mean(dim=1, keepdim=True)
        rgb = images.repeat(1, 3, 1, 1) if images.shape[1] == 1 and self.repeat_gray_to_3ch else images
        # Official interface: (pooled feature, list of BCHW stage feature maps).
        output = self.backbone(rgb)
        if not (isinstance(output, (tuple, list)) and len(output) == 2
                and isinstance(output[1], (tuple, list)) and output[1]):
            raise RuntimeError('Expected MambaVision AutoModel spatial stage outputs')
        spatial = output[1][-1]
        if spatial.ndim != 4 or spatial.shape[1] != self.main_spatial_dim:
            raise RuntimeError('Unexpected spatial feature shape; do not substitute classifier logits')
        if self.wavelet_branch is not None:
            frequency = self.wavelet_branch(gray)
            if self.wavelet_fusion_mode == 'gated':
                spatial = self.fusion(spatial, frequency)
            else:
                frequency = F.interpolate(frequency, size=spatial.shape[-2:], mode='bilinear', align_corners=False)
                if self.wavelet_fusion_mode == 'concat':
                    spatial = self.fusion(torch.cat([spatial, frequency], dim=1))
                else:
                    alpha = self.wavelet_fusion_alpha
                    spatial = (1 - alpha) * spatial + alpha * self.fusion(frequency)
        feature = self.feature_proj(F.adaptive_avg_pool2d(spatial, 1).flatten(1))
        tasks = self.towers(feature)
        return (tasks, self.hl_head(feature)) if self.use_hlgauss else tasks


# Original public names remain importable; original checkpoints need retraining.
MambaVisionLongImage6TaskV2 = POSMamba
MambaVisionLongImage6Task = POSMamba
