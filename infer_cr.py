import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import KfaceDataset
from models.cr.model import CoarseRestoration
from models.cr.loss import cr_loss


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = "cuda" if torch.cuda.is_available() else "cpu"


def infer_loop(dataloader, model, loss_fn):
    acc_loss = 0

    with torch.no_grad():
        for batch, (x, y, y_patches) in enumerate(dataloader):
            x, y, y_patches = x.to(device), y.to(device), y_patches.to(device)
            pred = model(x)
            loss = loss_fn(pred, y, y_patches).item()
            acc_loss += loss
            
            result = torch.cat([x[0], pred[0], y[0]], dim=-1)
            if not os.path.exists("output/infer"):
                os.makedirs("output/infer")
            save_image(result, os.path.join("output/infer/%d.png" % batch))
            print("loss=%.4f (%d/%d)" % (loss, batch, len(dataloader)))

    acc_loss /= len(dataloader)

    print("✅ done! (avg_loss=%.4f)" % (acc_loss))


CHECKPOINT_PATH = 'checkpoints/20.pt'
BATCH_SIZE = 8

infer_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)
infer_dataloader = DataLoader(dataset=infer_dataset, batch_size=BATCH_SIZE)

model = CoarseRestoration().to(device=device)
loss_fn = cr_loss

checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

infer_loop(dataloader=infer_dataloader, model=model, loss_fn=loss_fn)

print("✅ Done!")
