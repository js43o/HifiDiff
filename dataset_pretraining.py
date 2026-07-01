import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import cv2
import numpy as np
from basicsr.utils import img2tensor

LIGHT_COND = ["%02d" % i for i in range(20)]

ANGLES_MODERATE = ["08_0", "13_0", "14_0", "05_0", "04_1", "19_0"]
ANGLE_FRONTAL = "05_1"


class MultiPIEHQDataset(Dataset):
    def __init__(self, dataroot: str, size=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot, "train")
        self.paths = []
        self.size = size

        angles = [*ANGLES_MODERATE, ANGLE_FRONTAL]

        for pid in sorted(os.listdir(self.dataroot)):
            for angle in angles:
                for light in LIGHT_COND:
                    path = os.path.join(self.dataroot, pid, angle, "%s.png" % light)
                    if os.path.exists(path):
                        self.paths.append(path)

    def __getitem__(self, index):
        image = cv2.imread(self.paths[index])
        image = image.astype(np.float32) / 255.0

        image = cv2.resize(
            image, dsize=(self.size, self.size), interpolation=cv2.INTER_CUBIC
        )

        # BGR to RGB, HWC to CHW, numpy to tensor
        image = img2tensor(image, bgr2rgb=True, float32=True)

        return image

    def __len__(self):
        return len(self.paths)


class CelebAHQDataset(Dataset):
    def __init__(self, dataroot: str, size=128):
        super().__init__()
        self.dataroot = dataroot
        self.paths = []
        self.size = size

        for filename in sorted(os.listdir(self.dataroot)):
            path = os.path.join(self.dataroot, filename)

            if os.path.isfile(path) and filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            ):
                self.paths.append(path)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        # Return range: [0, 1], same as MultiPIEHQDataset
        image = F.to_tensor(image)

        return image

    def __len__(self):
        return len(self.paths)
