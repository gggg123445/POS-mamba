"""Six task targets/masks and the optional histogram Gaussian loss."""
import torch
import torch.nn.functional as F

def get_task_labels(base_labels):
    l1 = base_labels
    l2 = (base_labels >= 3).float().view(-1, 1)
    l3 = torch.zeros_like(base_labels)
    l3[base_labels == 0] = 0
    l3[(base_labels == 1) | (base_labels == 2)] = 1
    l3[(base_labels == 3) | (base_labels == 4)] = 2
    l3[base_labels == 5] = 3

    mask4 = (base_labels <= 2)
    l4 = torch.zeros_like(base_labels).float().view(-1, 1)
    l4[(base_labels == 1) | (base_labels == 2)] = 1.0

    mask5 = (base_labels >= 3)
    l5 = torch.zeros_like(base_labels).float().view(-1, 1)
    l5[base_labels == 5] = 1.0

    mask6 = (base_labels >= 3)
    l6 = torch.zeros_like(base_labels)
    l6[base_labels == 3] = 0
    l6[base_labels == 4] = 1
    l6[base_labels == 5] = 2

    return [l1, l2, l3, l4, l5, l6], [None, None, None, mask4, mask5, mask6]


def compute_multitask_loss(preds, labels, masks):
    loss = 0.0
    losses = []

    loss1 = F.cross_entropy(preds[0], labels[0])
    loss += loss1
    losses.append(float(loss1.item()))

    loss2 = F.binary_cross_entropy_with_logits(preds[1], labels[1])
    loss += loss2
    losses.append(float(loss2.item()))

    loss3 = F.cross_entropy(preds[2], labels[2])
    loss += loss3
    losses.append(float(loss3.item()))

    if masks[3].sum() > 0:
        loss4 = F.binary_cross_entropy_with_logits(preds[3][masks[3]], labels[3][masks[3]])
        loss += loss4
        losses.append(float(loss4.item()))
    else:
        losses.append(0.0)

    if masks[4].sum() > 0:
        loss5 = F.binary_cross_entropy_with_logits(preds[4][masks[4]], labels[4][masks[4]])
        loss += loss5
        losses.append(float(loss5.item()))
    else:
        losses.append(0.0)

    if masks[5].sum() > 0:
        loss6 = F.cross_entropy(preds[5][masks[5]], labels[5][masks[5]])
        loss += loss6
        losses.append(float(loss6.item()))
    else:
        losses.append(0.0)

    return loss, losses


def hl_soft_ce_with_probs(logits, target_probs):
    logp = F.log_softmax(logits, dim=1)
    return -(target_probs * logp).sum(dim=1).mean()
