import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import KfaceDataset
from models.cr.model import CoarseRestoration


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(dataloader, model, loss_fn, optimizer, current_epoch, loss_history=None):
    num_data = len(dataloader.dataset)
    model.train()

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss_history.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx + 1) % 100 == 0:
            print(
                "loss=%.4f (batch: %d/%d)" % (loss, (batch_idx + 1) * BATCH_SIZE, num_data)
            )
            # save images
            result = torch.cat([X[0], pred[0], y[0]], dim=-1)
            if not os.path.exists("output/%d" % current_epoch):
                os.makedirs("output/%d" % current_epoch)
            save_image(result, os.path.join("output/%d" % current_epoch, "%d.jpg" % (batch_idx + 1)))


def test_loop(dataloader, model, loss_fn, loss_history=None):
    test_loss = 0
    model.eval()

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= len(dataloader)
    loss_history.append(test_loss)

    print("test loss: %.4f" % (test_loss))


LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

train_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="train",
)
test_dataset = KfaceDataset(
    dataroot="../../datasets/kface",
    use="val",
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

model = CoarseRestoration().to(device=device)
loss_fn = nn.L1Loss()  # 임시 Loss 함수
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = [0.0]

for epoch in range(EPOCHS):
    print("🔄 %d epoch: loss=%.4f" % (epoch, train_losses[-1]))
    train_loop(
        dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, current_epoch=epoch, loss_history=train_losses
    )
    test_loop(dataloader=test_dataloader, model=model, loss_fn=loss_fn, loss_history=train_losses)
    
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_losses[-1],
            }, './checkpoints/%d.pt' % epoch)

print("✅ Done!")
