import torch
from torch.utils.data import DataLoader
from torch.nn.functional import triplet_margin_loss
from torchvision.utils import save_image
from tqdm.auto import tqdm
import wandb
import gc

from dataset import KfaceCropDataset_IDC
from models.cr.model import CoarseRestoration
from models.idc.model import ResNet50

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(dataloader, cr_module, model, loss_fn, optimizer, current_epoch):
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description(f"Epoch {current_epoch}")
    global_step = 0

    model.train()

    for batch_idx, (x, y, other) in enumerate(dataloader):
        x, y, other = x.to(device), y.to(device), other.to(device)
        cr_pred = cr_module(x)
        id_cr, id_hf, id_ck = model(cr_pred), model(y), model(other)
        loss = loss_fn(id_cr, id_hf, id_ck)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        global_step += 1
        logs = {
            "loss": loss.detach().item(),
            "step": global_step,
        }
        progress_bar.set_postfix(**logs)
        wandb.log({"train_loss": loss.detach().item()})

        if (batch_idx + 1) % 100 == 0:
            output = torch.concat((cr_pred, y, other))
            save_image(output, "output/idc/%d.png" % (batch_idx + 1))

    torch.cuda.empty_cache()
    gc.collect()


def val_loop(dataloader, cr_module, model, loss_fn):
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description("Validating...")
    acc_loss = 0
    model.eval()

    with torch.no_grad():
        for batch, (x, y, other) in enumerate(dataloader):
            x, y, other = x.to(device), y.to(device), other.to(device)
            cr_pred = cr_module(x)
            id_cr, id_hf, id_ck = model(cr_pred), model(y), model(other)
            loss = loss_fn(id_cr, id_hf, id_ck)

            progress_bar.update(1)
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            acc_loss += loss.detach().item()

    acc_loss /= len(dataloader)
    wandb.log({"val_loss": loss.detach().item()})

    torch.cuda.empty_cache()
    gc.collect()


LEARNING_RATE = 5e-4
BATCH_SIZE = 56
EPOCHS = 24
CR_CHECKPOINT_PATH = "checkpoints/cr/03_crop/23.pt"

wandb.init(
    # Set the project where this run will be logged
    project="hifi_idc",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name="02_crop",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "HifiDiff",
        "dataset": "kface_crop",
        "epochs": EPOCHS,
    },
)

train_dataset = KfaceCropDataset_IDC(
    dataroot="../../datasets/kface_crop",
    use="train",
)
val_dataset = KfaceCropDataset_IDC(
    dataroot="../../datasets/kface_crop",
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

for epoch in range(EPOCHS):
    train_loop(
        dataloader=train_dataloader,
        cr_module=cr_module,
        model=model,
        loss_fn=idc_loss,
        optimizer=optimizer,
        current_epoch=epoch,
    )
    val_loop(
        dataloader=val_dataloader,
        cr_module=cr_module,
        model=model,
        loss_fn=idc_loss,
    )

    if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "./checkpoints/idc/%d.pt" % (epoch + 1),
        )

print("✅ Done!")
wandb.finish()
