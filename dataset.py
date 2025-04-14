import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

LIGHT_CONDITION = ["L1", "L3", "L6"]
EXPRESSION_CONDITION = ["E01", "E02", "E03"]

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
                            self.dataroot, id, "S001", light, expression, "C%s.jpg" % angle
                        )
                        meta = os.path.join(
                            self.dataroot, id, "S001", light, expression, "C%s.txt" % angle
                        )

                        if angle == 7:  # frontal face
                            self.gt_imgs.extend([img] * 20)
                            self.gt_metas.extend([meta] * 20)
                        else:
                            self.input_imgs.append(img)
                            self.input_metas.append(meta)

    def __getitem__(self, index):
        print("processing image: #%d/%d" %(index, len(self.ids) * len(LIGHT_CONDITION) * len(EXPRESSION_CONDITION) * 19))
        input_img = Image.open(self.input_imgs[index]).convert("RGB")
        input_meta = open(self.input_metas[index], "r").readlines()
        gt_img = Image.open(self.gt_imgs[index]).convert("RGB")
        gt_meta = open(self.gt_metas[index], "r").readlines()
        
        left, top, width, height = map(int, input_meta[7].split("\t"))
        input_img = input_img.crop((left, top, left + width, top + height))
        input_img = input_img.resize((32, 32))
        
        left, top, width, height = map(int, gt_meta[7].split("\t"))
        gt_img = gt_img.crop((left, top, left + width, top + height))
        gt_img = gt_img.resize((128, 128))
        
        return input_img, gt_img

    def __len__(self):
        return len(self.input_imgs)


dataset = KfaceDataset("../datasets/kface", use="test")

print(list(dataset))
