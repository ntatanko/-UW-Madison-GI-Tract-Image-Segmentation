# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import ast
import glob
import json
import os
import random

import albumentations
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch


# +

def rle_decode(mask_array, img_shape):
    h, w = img_shape
    img_mask = np.zeros(h * w)
    rle_mask = np.array(mask_array.split(), dtype="int")
    starts = rle_mask[0::2] - 1
    ends = rle_mask[1::2] + starts
    for start, end in zip(starts, ends):
        img_mask[start:end] = 1
    return img_mask.reshape([h, w])


def rle_encode(mask):
    mask = np.pad(
        array=mask.flatten(), pad_width=[1, 1], mode="constant", constant_values=0
    )
    rle_mask = np.where(mask[1:] != mask[:-1])[0] + 1
    rle_mask[1::2] -= rle_mask[::2]
    return " ".join(str(x) for x in rle_mask)


def create_transform(img_size):
    transform = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.3),
            albumentations.VerticalFlip(p=0.3),
            albumentations.augmentations.geometric.rotate.RandomRotate90(p=0.3),
            albumentations.Rotate(
                p=0.3,
                limit=(-180, 180),
                interpolation=cv2.INTER_LINEAR,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            albumentations.CoarseDropout(
                p=0.05,
                max_holes=8,
                max_height=8,
                max_width=8,
                min_holes=1,
                min_height=1,
                min_width=1,
            ),
            albumentations.OneOf(
                [
                    albumentations.augmentations.geometric.transforms.Perspective(
                        scale=(0.05, 0.05),
                        keep_size=True,
                        pad_mode=0,
                        pad_val=0,
                        mask_pad_val=0,
                        fit_output=False,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.3,
                    ),
                    albumentations.ElasticTransform(
                        alpha=1,
                        sigma=25,
                        alpha_affine=25,
                        border_mode=0,
                        interpolation=cv2.INTER_LINEAR,
                        value=0,
                        mask_value=0,
                        approximate=False,
                        same_dxdy=False,
                        p=0.7,
                    ),
                ],
                p=0.5,
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.3
            ),
            albumentations.augmentations.crops.transforms.RandomResizedCrop(
                height=img_size[0],
                width=img_size[1],
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05),
                interpolation=cv2.INTER_LINEAR,
                p=0.3,
            ),
            albumentations.OneOf(
                [
                    albumentations.GridDistortion(
                        num_steps=7,
                        distort_limit=0.2,
                        border_mode=0,
                        interpolation=cv2.INTER_LINEAR,
                        value=0,
                        mask_value=0,
                        p=0.5,
                    ),
                    albumentations.augmentations.geometric.transforms.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=180,
                        interpolation=cv2.INTER_LINEAR,
                        border_mode=0,
                        value=0,
                        mask_value=0,
                        p=0.3,
                    ),
                ],
                p=0.5,
            ),
            albumentations.augmentations.transforms.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.2, 0.1),
                brightness_by_max=True,
                p=0.2,
            ),
            albumentations.MotionBlur(blur_limit=(3, 5), p=0.1),
        ],
        p=0.8,
    )
    return transform


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        img_size,
        n_img_chanels,
        uint8,
        sequence,
        npy_col,
        norm,
        shuffle,
        seed,
        transform=None,
        pad=True
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.n_img_chanels = n_img_chanels
        self.uint8 = uint8
        self.sequence = sequence
        self.npy_col = npy_col
        self.norm = norm
        self.shuffle = shuffle
        self.seed = seed
        self.transform = transform
        self.pad = pad
        if self.shuffle:
            self.df = self.df.sample(frac=1, random_state=self.seed).reset_index(
                drop=True
            )

    def __len__(self):
        return self.df.shape[0]

    def pad_img(self, img):
        x_pad = self.img_size[1] - img.shape[1]
        y_pad = self.img_size[0] - img.shape[0]
        x_pad = x_pad if x_pad > 0 else 0
        y_pad = y_pad if y_pad > 0 else 0
        left_pad = x_pad // 2
        right_pad = x_pad - left_pad
        top_pad = y_pad // 2
        bottom_pad = y_pad - top_pad
        img = np.pad(
            img,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return img

    def __getitem__(self, ix):
        npy_path = self.df.loc[ix, self.npy_col]
        img = np.load(npy_path)
        if self.pad:
            if np.less(img.shape[:2], self.img_size).any():
                img = self.pad_img(img)
            if np.greater(img.shape[:2], self.img_size).any():
                img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        mask = img[:, :, -3:]
        if self.sequence:
            img = img[:, :, :-3]
        else:
            img = img[:, :, :1]
            if self.n_img_chanels == 3:
                img = np.repeat(img, 3, 2)
        img = img.astype("float32")
        if self.norm:
            img -= img.min()
            img /= img.max()
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        return img.transpose(2, 0, 1), mask.transpose(2, 0, 1)

# -
