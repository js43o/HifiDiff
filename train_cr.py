import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import KfaceDataset
from models.cr.model import CoarseRestoration
from models.cr.loss import cr_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(dataloader, model, loss_fn, optimizer, current_epoch, loss_history=None):
    num_data = len(dataloader.dataset)
    model.train()

    for batch_idx, (x, y, y_patches) in enumerate(dataloader):
        x, y, y_patches = x.to(device), y.to(device), y_patches.to(device)
        pred = model(x)
        loss = loss_fn(pred, y, y_patches)
        
        loss_history.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            "loss=%.4f (batch: %d/%d)" % (loss, (batch_idx + 1) * BATCH_SIZE, num_data)
        )
        # save images
        if (batch_idx + 1) % 100 == 0:
            result = torch.cat([x[0], pred[0], y[0]], dim=-1)
            if not os.path.exists("output/%d" % current_epoch):
                os.makedirs("output/%d" % current_epoch)
            save_image(result, os.path.join("output/%d" % current_epoch, "%d.jpg" % (batch_idx + 1)))


def val_loop(dataloader, model, loss_fn, loss_history=None):
    acc_loss = 0
    model.eval()

    with torch.no_grad():
        for batch, (x, y, y_patches) in enumerate(dataloader):
            x, y, y_patches = x.to(device), y.to(device), y_patches.to(device)
            pred = model(x)
            loss = loss_fn(pred, y, y_patches).item()
            acc_loss += loss
            loss_history.append(loss)

    acc_loss /= len(dataloader)

    print("avg_loss: %.4f" % (acc_loss))


LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 24

train_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="train",
)
val_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

model = CoarseRestoration().to(device=device)
loss_fn = cr_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = [0.0]

for epoch in range(EPOCHS):
    print("🔄 %d epoch: loss=%.4f" % (epoch, train_losses[-1]))
    train_loop(
        dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, current_epoch=epoch, loss_history=train_losses
    )
    val_loop(dataloader=val_dataloader, model=model, loss_fn=loss_fn, loss_history=train_losses)
    
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            }, './checkpoints/%d.pt' % epoch)

print("✅ Done!")
