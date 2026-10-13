import pytest
import torch
from torch import nn

from pos_mamba.models import POSMamba, SpatialGatedFusion, HaarWaveletFreqBranchV2
from pos_mamba.losses import get_task_labels, compute_multitask_loss, hl_soft_ce_with_probs


def test_paper_pixel_gate_is_shared_across_channels():
    fusion = SpatialGatedFusion(8, 4).eval()
    main, freq = torch.randn(2, 8, 5, 9), torch.randn(2, 4, 5, 9)
    mapped = fusion.freq_to_main(freq)
    gate = fusion.gate_conv(torch.cat([main, mapped], dim=1))
    assert gate.shape == (2, 1, 5, 9)
    assert isinstance(fusion.gate_conv[2], nn.ReLU)
    torch.testing.assert_close(fusion(main, freq), (1 - gate) * main + gate * mapped)


def test_tail_probability_is_absorbed_not_truncated(fake_backbone):
    model = POSMamba(feature_proj_dim=16)
    targets = torch.tensor([0., 25., 50., 70., 100.])
    probs = model.hl_gauss_to_probs(targets, sigma=10)
    normal = torch.distributions.Normal(0., 1.)
    expected_first = normal.cdf((model.hl_edges[1] - targets) / 10)
    expected_last = 1 - normal.cdf((model.hl_edges[-2] - targets) / 10)
    torch.testing.assert_close(probs[:, 0], expected_first)
    torch.testing.assert_close(probs[:, -1], expected_last)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(len(targets)))
    assert (probs >= 0).all() and probs[0, 0] > 0.5 and probs[-1, -1] > 0.5
    with pytest.raises(ValueError):
        model.hl_gauss_to_probs(targets, sigma=0)


@pytest.mark.parametrize('mode', ['gated', 'concat', 'sum'])
def test_spatial_model_joint_loss_backpropagates_to_both_branches(fake_backbone, mode):
    model = POSMamba(feature_proj_dim=16, towers_hidden=8, wavelet_freq_hidden=8,
                     wavelet_fusion_mode=mode, hl_aux_hidden=8)
    tasks, histogram = model(torch.randn(6, 1, 40, 128))
    assert [tuple(t.shape) for t in tasks] == [(6, 6), (6, 1), (6, 4), (6, 1), (6, 1), (6, 3)]
    labels, masks = get_task_labels(torch.arange(6))
    main, _ = compute_multitask_loss(tasks, labels, masks)
    hl = hl_soft_ce_with_probs(histogram, model.hl_gauss_to_probs(torch.tensor([0., 12., 37., 60., 85., 100.])))
    (0.1 * main + 0.9 * hl).backward()
    assert model.backbone.conv.weight.grad.abs().sum() > 0
    assert model.wavelet_branch.encoder[0].weight.grad.abs().sum() > 0
    assert model.hl_head[-1].weight.grad.abs().sum() > 0


def test_classifier_logits_cannot_silently_replace_spatial_maps(fake_backbone):
    model = POSMamba()
    class WrongBackbone(nn.Module):
        def forward(self, x):
            return {'logits': torch.ones(len(x), 1000)}
    model.backbone = WrongBackbone()
    with pytest.raises(RuntimeError, match='spatial stage'):
        model(torch.randn(2, 1, 40, 128))


def test_conditional_tasks_have_correct_masks_and_zero_empty_loss():
    labels, masks = get_task_labels(torch.arange(6))
    assert labels[2].tolist() == [0, 1, 1, 2, 2, 3]
    assert masks[3].tolist() == [True, True, True, False, False, False]
    assert masks[4].tolist() == masks[5].tolist() == [False, False, False, True, True, True]
    labels, masks = get_task_labels(torch.tensor([0, 1, 2]))
    preds = [torch.randn(3, c, requires_grad=True) for c in [6, 1, 4, 1, 1, 3]]
    loss, values = compute_multitask_loss(preds, labels, masks)
    assert values[4:] == [0, 0]
    loss.backward()
    assert preds[4].grad is None and preds[5].grad is None


def test_haar_odd_sizes_and_singleton_normalization_are_finite():
    branch = HaarWaveletFreqBranchV2(levels=4, band_norm=True).eval()
    assert torch.isfinite(branch(torch.randn(2, 1, 17, 19))).all()
    with pytest.raises(ValueError, match='Input sides'):
        branch(torch.randn(2, 1, 8, 8))
