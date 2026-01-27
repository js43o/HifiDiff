from torch.nn import functional as F


def key_region_loss(pred, y, y_patch):
    loss = 0
    for batch in range(y.shape[0]):
        mask = y_patch[batch].sum(dim=0) > 0.0
        mask = mask.unsqueeze(0).expand(y_patch[batch].shape)
        pred_patch = pred[batch] * mask
        gt_patch = y[batch] * mask
        loss += F.mse_loss(pred_patch, gt_patch)

    return loss / y.shape[0]


def cr_loss(pred, y, y_patch):
    return F.mse_loss(pred, y) + key_region_loss(pred, y, y_patch)
