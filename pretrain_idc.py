import torch
from torch.utils.data import DataLoader
from torch.nn.functional import triplet_margin_loss
from torchvision.utils import save_image

from dataset import KfaceDataset_IDC
from models.cr.model import CoarseRestoration
from models.idc.model import ResNet50

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(
    dataloader, cr_module, model, loss_fn, optimizer, current_epoch, loss_history=None
):
    num_data = len(dataloader.dataset)
    model.train()

    for batch_idx, (x, y, other) in enumerate(dataloader):
        x, y, other = x.to(device), y.to(device), other.to(device)
        cr_pred = cr_module(x)
        id_cr, id_hf, id_ck = model(cr_pred), model(y), model(other)
        loss = loss_fn(id_cr, id_hf, id_ck)

        loss_history.append(loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(
            "loss=%.8f (epoch=%d / batch=%d/%d)"
            % (loss, current_epoch + 1, (batch_idx + 1) * BATCH_SIZE, num_data)
        )

        if (batch_idx + 1) % 100 == 0:
            output = torch.concat((cr_pred, y, other))
            # print("😮 OUTPUT:", cr_pred.shape, y.shape, output.shape, id_cr.shape)
            save_image(output, "output/idc/%d.png" % batch_idx)


def val_loop(dataloader, cr_module, model, loss_fn, loss_history=None):
    acc_loss = 0
    model.eval()

    with torch.no_grad():
        for batch, (x, y, other) in enumerate(dataloader):
            x, y, other = x.to(device), y.to(device), other.to(device)
            cr_pred = cr_module(x)
            id_cr, id_hf, id_ck = model(cr_pred), model(y), model(other)
            loss = loss_fn(id_cr, id_hf, id_ck)

            acc_loss += loss
            loss_history.append(loss)

    acc_loss /= len(dataloader)

    print("avg_loss: %.4f" % (acc_loss))


LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 24
CR_CHECKPOINT_PATH = "checkpoints/cr/23.pt"

train_dataset = KfaceDataset_IDC(
    dataroot="../../datasets/kface",
    use="train",
)
val_dataset = KfaceDataset_IDC(
    dataroot="../../datasets/kface",
    use="val",
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

cr_module = CoarseRestoration().to(device=device)
cr_checkpoint = torch.load(CR_CHECKPOINT_PATH)
cr_module.load_state_dict(cr_checkpoint["model_state_dict"])
cr_module.eval()

model = ResNet50().to(device=device)
idc_loss = triplet_margin_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_losses = [0.0]

for epoch in range(EPOCHS):
    print("🔄 %d epoch: loss=%.4f" % (epoch, train_losses[-1]))
    train_loop(
        dataloader=train_dataloader,
        cr_module=cr_module,
        model=model,
        loss_fn=idc_loss,
        optimizer=optimizer,
        current_epoch=epoch,
        loss_history=train_losses,
    )
    val_loop(
        dataloader=val_dataloader,
        cr_module=cr_module,
        model=model,
        loss_fn=idc_loss,
        loss_history=train_losses,
    )

    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_losses[-1],
            },
            "./checkpoints/idc/%d.pt" % (epoch + 1),
        )

print("✅ Done!")
