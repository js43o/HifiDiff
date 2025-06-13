import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm
import wandb
import argparse
import gc

from dataset import KfaceCropDataset
from models.cr.model import CoarseRestoration
from models.cr.loss import cr_loss


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type=str,
    default="0",
    help="A number for checkpoints and output path names",
)
parser.add_argument("--gpu", type=str, default="0", help="Which GPU to use")
parser.add_argument("--num_epochs", type=int, default=24, help="A number of epoch")
parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-4,
    help="An intial learning rate for the optimizer",
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="A batch size of training dataset"
)
parser.add_argument(
    "--sample_size", type=int, default=8, help="The number of sampling images"
)
parser.add_argument(
    "--save_model_epoch",
    type=int,
    default=5,
    help="A number of epoch to save current model",
)
parser.add_argument(
    "--save_image_batch",
    type=int,
    default=100,
    help="A number of batch to save sample images",
)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./checkpoints/cr/%s" % args.name, exist_ok=True)
os.makedirs("./output/cr/%s/train" % args.name, exist_ok=True)
os.makedirs("./output/cr/%s/val" % args.name, exist_ok=True)


def train_loop(dataloader, model, loss_fn, optimizer, current_epoch):
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description(f"Epoch {current_epoch}")
    global_step = 0

    model.train()

    for batch, (x, y, y_patches) in enumerate(dataloader):
        x, y, y_patches = x.to(device), y.to(device), y_patches.to(device)

        pred = model(x)
        loss = loss_fn(pred, y, y_patches)

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

        # save images
        if (batch + 1) % args.save_image_batch == 0:
            result = torch.cat([x[0], pred[0], y[0]], dim=-1)
            save_image(
                result,
                os.path.join(
                    "output/cr/%s/train/%d_%d.png"
                    % (args.name, current_epoch, batch + 1),
                ),
            )


def val_loop(dataloader, model, loss_fn, current_epoch):
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description(f"Validating")
    global_step = 0
    acc_loss = 0

    model.eval()

    with torch.no_grad():
        for batch, (x, y, y_patches) in enumerate(dataloader):
            x, y, y_patches = x.to(device), y.to(device), y_patches.to(device)
            pred = model(x)
            loss = loss_fn(pred, y, y_patches)
            acc_loss += loss.detach().item()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            global_step += 1

            # save images
            if (batch + 1) % args.save_image_batch == 0:
                result = torch.cat([x[0], pred[0], y[0]], dim=-1)
                save_image(
                    result,
                    os.path.join(
                        "output/cr/%s/val/%d_%d.png"
                        % (args.name, current_epoch, batch + 1),
                    ),
                )

    acc_loss /= len(dataloader)

    print("avg_loss: %.4f" % (acc_loss))
    wandb.log({"val_acc": acc_loss})


wandb.init(
    # Set the project where this run will be logged
    project="hifi_cr",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=args.name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.learning_rate,
        "architecture": "HifiDiff",
        "dataset": "kface_crop",
        "epochs": args.num_epochs,
    },
)


train_dataset = KfaceCropDataset(
    dataroot="../../datasets/kface_crop", use="train", fixed_light=True
)
val_dataset = KfaceCropDataset(
    dataroot="../../datasets/kface_crop", use="val", fixed_light=True
)

train_dataloader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True
)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.sample_size)

model = CoarseRestoration().to(device=device)
loss_fn = cr_loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

for epoch in range(args.num_epochs):
    train_loop(
        dataloader=train_dataloader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        current_epoch=epoch,
    )
    val_loop(
        dataloader=val_dataloader, model=model, loss_fn=loss_fn, current_epoch=epoch
    )

    if epoch % args.save_model_epoch == 0 or epoch == args.num_epochs - 1:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "./checkpoints/cr/%s/%d.pt" % (args.name, epoch),
        )

print("✅ Done!")
wandb.finish()
