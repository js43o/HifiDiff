import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from torchvision.utils import save_image

from dataset import KfaceDataset_IDC
from models.cr.model import CoarseRestoration
from models.idc.model import ResNet50

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_loop(dataloader, cr_module, model):
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for batch, (x, y, other) in enumerate(dataloader):
            print("(%d/%d)" % (batch, len(dataloader)), end=" ")
            x, y, other = x.to(device), y.to(device), other.to(device)
            cr_pred = cr_module(x)
            id_cr, id_hf, id_ck = model(cr_pred), model(y), model(other)
            sim_hf, sim_ck = (
                cosine_similarity(id_cr, id_hf).mean().item(),
                cosine_similarity(id_cr, id_ck).mean().item(),
            )
            is_correct = sim_hf > sim_ck 
            print("CR-HF vs. CR-CK:", round(sim_hf, 4), round(sim_ck, 4), "✅" if is_correct else "❌")

            accuracy += 1 if is_correct else 0

    accuracy /= len(dataloader)

    print("test accuracy: %.4f" % (accuracy))


BATCH_SIZE = 8
CR_CHECKPOINT_PATH = "checkpoints/cr/23.pt"

test_dataset = KfaceDataset_IDC(
    dataroot="../../datasets/kface",
    use="test",
)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

cr_module = CoarseRestoration().to(device=device)
cr_checkpoint = torch.load(CR_CHECKPOINT_PATH)
cr_module.load_state_dict(cr_checkpoint["model_state_dict"])
cr_module.eval()

model = ResNet50().to(device=device)
# checkpoint = torch.load("./checkpoints/idc/10.pt")
# model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loop(dataloader=test_dataloader, cr_module=cr_module, model=model)

print("✅ Done!")
