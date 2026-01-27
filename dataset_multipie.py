import os
import random
import math
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor

LIGHT_COND = ["%02d" % i for i in range(20)]

ANGLES_EXTREME = ["11_0", "12_0", "09_0", "19_1", "08_1", "20_0", "01_0", "24_0"]
ANGLES_MODERATE = ["08_0", "13_0", "14_0", "05_0", "04_1", "19_0"]

GT_ANGLES_MODERATE = ["08_0", "19_0"]
GT_ANGLES_FRONTAL = ["05_1", "05_1"]


class MultiPIEDataset(Dataset):
    @staticmethod
    def color_jitter(img_input, img_gt, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)

        img_input = img_input + jitter_val
        img_gt = img_gt + jitter_val
        img_input = np.clip(img_input, 0, 1)
        img_gt = np.clip(img_gt, 0, 1)

        return img_input, img_gt

    def __init__(
        self,
        dataroot: str,
        model_type="uni",
        use="train",
        res=128,
        use_blind=False,  # using blind degradation model
        use_patch=False,  # using facial components patches
    ):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.res = res
        self.use_blind = use_blind
        self.use_patch = use_patch

        self.input_paths = []
        self.input_angles = []
        self.gt_paths = []
        self.gt_patch_paths = []

        if model_type == "e2m":
            angles = ANGLES_EXTREME
            gt_angles = GT_ANGLES_MODERATE
        elif model_type == "m2f":
            angles = ANGLES_MODERATE
            gt_angles = GT_ANGLES_FRONTAL
        elif model_type == "e2f":
            angles = ANGLES_EXTREME
            gt_angles = GT_ANGLES_FRONTAL
        elif model_type == "uni":
            angles = [*ANGLES_EXTREME, *ANGLES_MODERATE]
            gt_angles = GT_ANGLES_FRONTAL

        for pid in sorted(os.listdir(self.dataroot)):
            for idx, angle in enumerate(angles):
                for light in LIGHT_COND:
                    gt_angle = gt_angles[0] if idx < len(angles) // 2 else gt_angles[1]
                    gt_path = os.path.join(
                        self.dataroot, pid, gt_angle, "%s.png" % light
                    )
                    input_path = os.path.join(
                        self.dataroot, pid, angle, "%s.png" % light
                    )
                    if all(map(os.path.exists, [gt_path, input_path])):
                        self.input_paths.append(input_path)
                        self.gt_paths.append(gt_path)

                        if self.use_patch:
                            gt_patch_path = os.path.join(
                                self.dataroot, pid, gt_angle, "%s_patch.png" % light
                            )
                            self.gt_patch_paths.append(gt_patch_path)

    def __getitem__(self, index):
        input_image = cv2.imread(self.input_paths[index])
        gt_image = cv2.imread(self.gt_paths[index])

        input_image = cv2.resize(
            input_image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
        )
        gt_image = cv2.resize(
            gt_image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
        )

        # random horizontal flip
        input_image, _status = augment(
            input_image, hflip=True, rotation=False, return_status=True
        )
        gt_image, _status = augment(
            gt_image, hflip=True, rotation=False, return_status=True
        )

        input_image = input_image.astype(np.float32) / 255.0
        gt_image = gt_image.astype(np.float32) / 255.0

        if self.use_blind:
            # blur
            cur_kernel_size = random.randint(19, 20) * 2 + 1
            kernel = degradations.random_mixed_kernels(
                ["iso", "aniso"],
                [0.5, 0.5],
                cur_kernel_size,
                [0.1, 2.0],
                [0.1, 2.0],
                [-math.pi, math.pi],
                noise_range=None,
            )
            input_image = cv2.filter2D(input_image, -1, kernel)

            # downsample
            scale = np.random.uniform(0.8, 8.0)
            input_image = cv2.resize(
                input_image,
                (int(self.res // scale), int(self.res // scale)),
                interpolation=cv2.INTER_LINEAR,
            )

            # noise
            input_image = degradations.random_add_gaussian_noise(input_image, [0, 10])

            # jpeg compression
            input_image = degradations.random_add_jpg_compression(
                input_image, [80, 100]
            )

            # resize to original size
            input_image = cv2.resize(
                input_image, (self.res, self.res), interpolation=cv2.INTER_LINEAR
            )

            # random color jitter
            if np.random.uniform() < 0.5:
                input_image, gt_image = self.color_jitter(input_image, gt_image, 0.05)

            # random to gray (only for lq)
            if np.random.uniform() < 0.008:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                input_image = np.tile(input_image[:, :, None], [1, 1, 3])

        else:
            input_image = cv2.resize(
                input_image,
                dsize=(self.res // 4, self.res // 4),
                interpolation=cv2.INTER_CUBIC,
            )
            input_image = cv2.resize(
                input_image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
            )

        # BGR to RGB, HWC to CHW, numpy to tensor
        gt_image, input_image = img2tensor(
            [gt_image, input_image], bgr2rgb=True, float32=True
        )

        # round and clip
        input_image = torch.clamp((input_image * 255.0).round(), 0, 255) / 255.0

        if self.use_patch:
            gt_patch = to_tensor(
                Image.open(self.gt_patch_paths[index])
                .convert("RGB")
                .resize((self.res, self.res), Image.Resampling.BICUBIC)
            )

            return input_image, gt_image, gt_patch

        return input_image, gt_image

    def __len__(self):
        return len(self.gt_paths)


class MultiPIEDatasetWithSingleView(Dataset):
    def __init__(self, dataroot: str, angle: str, use="train", res=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.res = res
        self.angle = angle

        self.input_paths = []
        self.gt_paths = []
        self.gt_patches = []

        for pid in sorted(os.listdir(self.dataroot)):
            for light in LIGHT_COND:
                gt_angle = GT_ANGLES_FRONTAL[0]
                gt_path = os.path.join(
                    self.dataroot,
                    pid,
                    gt_angle,
                    "%s.png" % light,
                )
                gt_patch_path = os.path.join(
                    self.dataroot,
                    pid,
                    gt_angle,
                    "%s_patch.png" % light,
                )
                image_path = os.path.join(
                    self.dataroot,
                    pid,
                    angle,
                    "%s.png" % light,
                )
                if not all(map(os.path.exists, [gt_path, gt_patch_path, image_path])):
                    continue

                self.input_paths.append(image_path)
                self.gt_paths.append(gt_path)
                self.gt_patches.append(gt_patch_path)

    def __getitem__(self, index):
        input_image = Image.open(self.input_paths[index]).convert("RGB")
        gt_image = Image.open(self.gt_paths[index]).convert("RGB")

        input_image = input_image.resize(
            (32, 32), Image.Resampling.BICUBIC
        )  # make it low-resolution
        input_image = input_image.resize((self.res, self.res), Image.Resampling.BICUBIC)
        gt_image = gt_image.resize((self.res, self.res), Image.Resampling.BICUBIC)
        gt_patch = (
            Image.open(self.gt_patches[index])
            .convert("RGB")
            .resize((self.res, self.res), Image.Resampling.BICUBIC)
        )

        return (
            to_tensor(input_image),
            to_tensor(gt_image),
            to_tensor(gt_patch),
            self.angle,
        )

    def __len__(self):
        return len(self.input_paths)


class MultiPIEDatasetForInference(Dataset):
    def __init__(self, dataroot: str, model_type="uni", use="train", res=128):
        super().__init__()
        self.dataroot = os.path.join(dataroot, use)
        self.res = res

        self.images = []

        if model_type == "e2m":
            angles = ANGLES_EXTREME
        elif model_type == "m2f":
            angles = ANGLES_MODERATE
        elif model_type == "e2f":
            angles = ANGLES_EXTREME
        elif model_type == "uni":
            angles = [*ANGLES_EXTREME, *ANGLES_MODERATE]

        for pid in sorted(os.listdir(self.dataroot)):
            for angle in angles:
                for light in LIGHT_COND:
                    image_path = os.path.join(
                        self.dataroot,
                        pid,
                        angle,
                        "%s.png" % light,
                    )
                    if not os.path.exists(image_path):
                        continue

                    self.images.append(image_path)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = image.resize((32, 32), Image.Resampling.BICUBIC).resize(
            (self.res, self.res), Image.Resampling.BICUBIC
        )  # make it low-resolution

        return to_tensor(image), self.images[index]

    def __len__(self):
        return len(self.images)
