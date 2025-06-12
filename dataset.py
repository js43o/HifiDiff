import os
from PIL import Image
from typing import List
import torch
import numpy as np
from random import shuffle
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.utils import save_image

from models.face_parser.model import extract_masks

LIGHT_CONDITION = ["L1", "L3"]  # "L6" 임시 제외
EXPRESSION_CONDITION = ["E01", "E02", "E03"]


def get_masked_patches(y: Image.Image, y_meta: List[str] = None):
    patches = []
    y_width, y_height = y.size
    y = y.resize((128, 128), Image.Resampling.BICUBIC)

    if y_meta is not None:
        head_left, head_top, _, _ = map(int, y_meta[7].split("\t"))

        for line in y_meta[8:12]:  # eye_r, eye_l, nose, mouth
            mask = np.zeros((y_height, y_width), dtype=np.uint8)
            left, top, width, height = map(int, line.split("\t"))
            mask[
                top - head_top : top + height - head_top,
                left - head_left : left + width - head_left,
            ] = 1
            mask = Image.fromarray(mask)
            mask = mask.resize((128, 128), Image.Resampling.NEAREST)
            mask = np.array(mask)[..., np.newaxis]

            patch = np.array(y) * mask
            patch = F.to_tensor(patch)

            patches.append(patch)
    else:
        masks = extract_masks(y)
        for mask in masks:
            patch = np.array(y) * mask
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
        input_img = input_img.resize(
            (32, 32), Image.Resampling.BICUBIC
        )  # make it low-resolution
        input_img = input_img.resize((128, 128), Image.Resampling.BICUBIC)

        left, top, width, height = map(int, gt_meta[7].split("\t"))
        gt_img = gt_img.crop((left, top, left + width, top + height))
        gt_patches = get_masked_patches(gt_img, gt_meta)
        gt_img = gt_img.resize((128, 128), Image.Resampling.BICUBIC)

        return F.to_tensor(input_img), F.to_tensor(gt_img), torch.stack(gt_patches)

    def __len__(self):
        return len(self.input_imgs)


class KfaceCropDataset(Dataset):
    def __init__(self, dataroot: str, use="train"):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.ids = os.listdir(self.dataroot)

        self.input_imgs = []
        self.gt_imgs = []

        for id in self.ids:
            for light in range(1, 21):
                for expression in EXPRESSION_CONDITION:
                    gt_path = os.path.join(
                        self.dataroot, id, "S001", "L%d" % light, expression, "C7.jpg"
                    )
                    if not os.path.exists(gt_path):
                        continue

                    cropped_count = 0

                    for angle in range(1, 21):
                        if angle == 7:
                            continue

                        img_path = os.path.join(
                            self.dataroot,
                            id,
                            "S001",
                            "L%d" % light,
                            expression,
                            "C%s.jpg" % angle,
                        )
                        if os.path.exists(img_path):
                            self.input_imgs.append(img_path)
                            cropped_count += 1

                    self.gt_imgs.extend([gt_path] * cropped_count)

    def __getitem__(self, index):
        input_img = Image.open(self.input_imgs[index]).convert("RGB")
        gt_img = Image.open(self.gt_imgs[index]).convert("RGB")

        input_img = input_img.resize(
            (32, 32), Image.Resampling.BICUBIC
        )  # make it low-resolution
        input_img = input_img.resize((128, 128), Image.Resampling.BICUBIC)

        gt_patches = get_masked_patches(gt_img)
        gt_patches = (
            torch.stack(gt_patches) if len(gt_patches) > 0 else torch.tensor([])
        )

        gt_img = gt_img.resize((128, 128), Image.Resampling.BICUBIC)

        return F.to_tensor(input_img), F.to_tensor(gt_img), gt_patches

    def __len__(self):
        return len(self.input_imgs)


class KfaceDataset_IDC(Dataset):  # for pre-train the IDC module
    def __init__(self, dataroot: str, use="train"):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.ids = os.listdir(self.dataroot)
        shuffle(self.ids)
        self.ids.extend(self.ids[:19])  # loop for the last 19 persons

        self.input_imgs = []
        self.input_metas = []
        self.gt_imgs = []
        self.gt_metas = []
        self.other_imgs = []
        self.other_metas = []

        for idx in range(len(self.ids) - 19):
            for light in LIGHT_CONDITION:
                for expression in EXPRESSION_CONDITION:
                    for angle in range(1, 21):
                        img = os.path.join(
                            self.dataroot,
                            self.ids[idx],
                            "S001",
                            light,
                            expression,
                            "C%s.jpg" % angle,
                        )
                        meta = os.path.join(
                            self.dataroot,
                            self.ids[idx],
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

                    for idx_other in range(idx + 1, idx + 20):
                        img = os.path.join(
                            self.dataroot,
                            self.ids[idx_other],
                            "S001",
                            light,
                            expression,
                            "C7.jpg",
                        )
                        meta = os.path.join(
                            self.dataroot,
                            self.ids[idx_other],
                            "S001",
                            light,
                            expression,
                            "C7.txt",
                        )

                        self.other_imgs.append(img)
                        self.other_metas.append(meta)

    def __getitem__(self, index):
        input_img = Image.open(self.input_imgs[index]).convert("RGB")
        input_meta = open(self.input_metas[index], "r").readlines()
        gt_img = Image.open(self.gt_imgs[index]).convert("RGB")
        gt_meta = open(self.gt_metas[index], "r").readlines()
        other_img = Image.open(self.other_imgs[index]).convert("RGB")
        other_meta = open(self.other_metas[index], "r").readlines()

        left, top, width, height = map(int, input_meta[7].split("\t"))
        input_img = input_img.crop((left, top, left + width, top + height))
        input_img = input_img.resize(
            (32, 32), Image.Resampling.BICUBIC
        )  # make it low-resolution
        input_img = input_img.resize((128, 128), Image.Resampling.BICUBIC)

        left, top, width, height = map(int, gt_meta[7].split("\t"))
        gt_img = gt_img.crop((left, top, left + width, top + height))
        gt_img = gt_img.resize((128, 128), Image.Resampling.BICUBIC)

        left, top, width, height = map(int, other_meta[7].split("\t"))
        other_img = other_img.crop((left, top, left + width, top + height))
        other_img = other_img.resize((128, 128), Image.Resampling.BICUBIC)

        return F.to_tensor(input_img), F.to_tensor(gt_img), F.to_tensor(other_img)

    def __len__(self):
        return len(self.input_imgs)


class KfaceHRDataset(Dataset):  # for pre-train the denoiser
    def __init__(self, dataroot: str, res=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot, "train")
        self.ids = os.listdir(self.dataroot)
        self.res = res

        self.imgs = []
        self.metas = []

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
                        self.imgs.append(img)
                        self.metas.append(meta)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        meta = open(self.metas[index], "r").readlines()

        left, top, width, height = map(int, meta[7].split("\t"))
        img = img.crop((left, top, left + width, top + height))
        img = img.resize((self.res, self.res), Image.Resampling.BICUBIC)

        return F.to_tensor(img)

    def __len__(self):
        return len(self.imgs)


class KfaceCropHRDataset(Dataset):  # for pre-train the denoiser
    def __init__(self, dataroot: str, res=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot, "train")
        self.ids = os.listdir(self.dataroot)
        self.res = res

        self.imgs = []

        for id in self.ids:
            for light in range(1, 21):
                for expression in EXPRESSION_CONDITION:
                    for angle in range(1, 21):
                        img_path = os.path.join(
                            self.dataroot,
                            id,
                            "S001",
                            "L%s" % light,
                            expression,
                            "C%s.jpg" % angle,
                        )
                        if os.path.exists(img_path):
                            self.imgs.append(img_path)

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")

        img = img.resize((self.res, self.res), Image.Resampling.BICUBIC)

        return F.to_tensor(img)

    def __len__(self):
        return len(self.imgs)


class CelebAHQDataset(Dataset):  # for pre-train the denoiser
    def __init__(self, dataroot: str, res=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot)
        self.imgs = []
        self.res = res

        for filename in os.listdir(self.dataroot):
            self.imgs.append(os.path.join(self.dataroot, filename))

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert("RGB")
        img = img.resize((self.res, self.res), Image.Resampling.BICUBIC)

        return F.to_tensor(img)

    def __len__(self):
        return len(self.imgs)
