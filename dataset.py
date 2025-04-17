import os
from PIL import Image
from typing import List
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.utils import save_image

LIGHT_CONDITION = ["L1", "L3", "L6"]
EXPRESSION_CONDITION = ["E01", "E02", "E03"]


def get_masked_patches(y, y_meta: List[str]):
    patches = []
    y_width, y_height = y.size
    y = y.resize((128, 128), resample=Image.Resampling.LANCZOS)
    y = np.array(y)
    
    head_left, head_top, _, _ = map(int, y_meta[7].split("\t"))
    for line in y_meta[8:12]:   # eye_r, eye_l, nose, mouth
        mask = np.zeros((y_height, y_width), dtype=np.uint8)
        left, top, width, height = map(int, line.split("\t"))
        mask[top-head_top:top+height-head_top, left-head_left:left+width-head_left] = 1
        mask = Image.fromarray(mask)
        mask = mask.resize((128, 128), resample=Image.Resampling.NEAREST)
        mask = np.array(mask)[..., np.newaxis]
        
        patch = y * mask
        patch = F.to_tensor(patch)
        
        patches.append(patch)

    return patches


class KfaceDataset(Dataset):
    def __init__(self, dataroot: str, use="train"):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.ids = os.listdir(self.dataroot)

        self.input_imgs = []
        self.input_metas = []
        self.gt_imgs = []
        self.gt_metas = []

        for id in self.ids:
            for light in LIGHT_CONDITION:
                for expression in EXPRESSION_CONDITION:
                    for angle in range(1, 21):
                        img = os.path.join(
                            self.dataroot,
                            id,
                            "S001",
                            light,
                            expression,
                            "C%s.jpg" % angle,
                        )
                        meta = os.path.join(
                            self.dataroot,
                            id,
                            "S001",
                            light,
                            expression,
                            "C%s.txt" % angle,
                        )

                        if angle == 7:  # frontal face
                            self.gt_imgs.extend([img] * 19)
                            self.gt_metas.extend([meta] * 19)
                        else:
                            self.input_imgs.append(img)
                            self.input_metas.append(meta)

    def __getitem__(self, index):
        # print("processing image # %d (total %d)" %(index, len(self.ids) * len(LIGHT_CONDITION) * len(EXPRESSION_CONDITION) * 19))
        input_img = Image.open(self.input_imgs[index]).convert("RGB")
        input_meta = open(self.input_metas[index], "r").readlines()
        gt_img = Image.open(self.gt_imgs[index]).convert("RGB")
        gt_meta = open(self.gt_metas[index], "r").readlines()

        left, top, width, height = map(int, input_meta[7].split("\t"))
        input_img = input_img.crop((left, top, left + width, top + height))
        input_img = input_img.resize((32, 32))  # make it low-resolution
        input_img = input_img.resize((128, 128))

        left, top, width, height = map(int, gt_meta[7].split("\t"))
        gt_img = gt_img.crop((left, top, left + width, top + height))
        gt_patches = get_masked_patches(gt_img, gt_meta)
        gt_img = gt_img.resize((128, 128))

        return F.to_tensor(input_img), F.to_tensor(gt_img), torch.stack(gt_patches)  # GT 바운딩 박스 정보 필요

    def __len__(self):
        return len(self.input_imgs)
