from torch.nn import functional as F


def key_region_loss(pred, y, y_patches):
    loss = 0
    for batch in range(y.shape[0]):
        for y_patch in y_patches[batch]:
            mask = y_patch > 0
            pred_patch = pred[batch] * mask

            loss += F.mse_loss(pred_patch, y_patch)

    return loss / y.shape[0]


def cr_loss(pred, y, y_patches):
    return F.mse_loss(pred, y) + key_region_loss(pred, y, y_patches)
