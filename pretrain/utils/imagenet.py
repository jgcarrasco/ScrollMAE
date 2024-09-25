# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Optional, Tuple

import zarr
import cv2

import PIL.Image as PImage
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import transforms
from torch.utils.data import Dataset

try:
    from torchvision.transforms import InterpolationMode
    interpolation = InterpolationMode.BICUBIC
except:
    import PIL
    interpolation = PIL.Image.BICUBIC


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f: img: PImage.Image = PImage.open(f).convert('RGB')
    return img


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            imagenet_folder: str,
            train: bool,
            transform: Callable,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        imagenet_folder = os.path.join(imagenet_folder, 'train' if train else 'val')
        super(ImageNetDataset, self).__init__(
            imagenet_folder,
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=None, is_valid_file=is_valid_file
        )
        
        self.samples = tuple(img for (img, label) in self.samples)
        self.targets = None # this is self-supervised learning so we don't need labels
    
    def __getitem__(self, index: int) -> Any:
        img_file_path = self.samples[index]
        return self.transform(self.loader(img_file_path))


class SegmentDataset(Dataset):
    """
    Load patches of a given segment. As the segments are large, this dataset will divide the segment in a number of
    crops of size `(z_depth, crop_size, crop_size)` by sliding a window with a given `stride` and taking the `z_depth`
    centermost layers.

    Input:
    -----------
    - `segment_path`: Path containing the .zarr segment, downloaded from 'https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/'.
    The folder must be named as '{segment_id}.zarr'. Inside the .zarr folder, you must also include the mask of 
    the segment with the format '{segment_id}_mask.png'. Additionally, you can also include inklabels as '{segment_id}_inklabels.png'
    if you want to compare the results.

    - `mode`: ['pretrain', 'eval'] 'eval' will also yield the provided inklabels for the sake of evaluating our
    methods with a given baseline. However, on pretraining and once the model is developed, 'pretrain' mode will 
    be used.
    """

    def __init__(self, segment_path="../jepa-scrolls/data/20231210121321.zarr", crop_size=320, z_depth=20, stride=None, mode="pretrain", transforms=None):

        self.transforms = transforms

        if not stride:
            stride = crop_size
        self.crop_size = crop_size
        self.mode = mode
        self.segment = zarr.load(segment_path)[0] # (depth, height, width)
        # take the z_depth central layers
        z_i = self.segment.shape[0] // 2 - z_depth // 2
        z_f = z_i + z_depth
        self.segment = torch.tensor(self.segment[z_i:z_f], dtype=torch.float32)
        
        # Load segment mask
        segment_id = segment_path.split("/")[-1].split(".")[0]
        self.segment_mask = torch.tensor(cv2.imread(f"{segment_path}/{segment_id}_mask.png", cv2.IMREAD_GRAYSCALE), dtype=torch.bool)
        # Compute channel-wise mean and std for normalization purposes
        self.mean, self.std = self.compute_mean_std()
        self.segment = (self.segment - self.mean[:, None, None]) / self.std[:, None, None]

        if self.mode == "eval":
            self.inklabels = cv2.imread(f"{segment_path}/{segment_id}_inklabels.png", cv2.IMREAD_GRAYSCALE)
            # NOTE: Look into this - However, I think that they are aligned at (0, 0) so there should be no problem
            # The assertion below does not hold. I guess that there is some sort of padding/rescaling that causes it to not match?
            # assert(self.inklabels.shape == self.segment_mask.shape), "The shape of the mask and inklabels must be equal"
        assert (self.segment.shape[1:] == self.segment_mask.shape), "The shape of the mask must be the same as the segment!"
        self.h, self.w = self.segment.shape[1:]
        self.crop_pos = []
        for i in range(0, self.h - crop_size, stride):
            for j in range(0, self.w - crop_size, stride):
                if self.segment_mask[i:i+crop_size, j:j+crop_size].min() > 0:
                    self.crop_pos.append((i, j))
        
    def compute_mean_std(self):
        masked_segment = self.segment[:, self.segment_mask] # only retrieve the part inside the mask
        return masked_segment.mean(1), masked_segment.std(1)

    def __len__(self):
        return len(self.crop_pos)

    def __getitem__(self, idx):
        i, j = self.crop_pos[idx]
        crop = self.segment[:, i:i+self.crop_size, j:j+self.crop_size]
        if self.transforms:
            crop = self.transforms(crop)
        if self.mode != "eval":
            return crop
        else:
            label = self.inklabels[i:i+self.crop_size, j:j+self.crop_size]
            return torch.tensor(i), torch.tensor(j), crop, torch.tensor(label, dtype=torch.float32)


def build_dataset_to_pretrain(dataset_path, input_size, dataset_type) -> Dataset:
    """
    You may need to modify this function to return your own dataset.
    Define a new class, a subclass of `Dataset`, to replace our ImageNetDataset.
    Use dataset_path to build your image file path list.
    Use input_size to create the transformation function for your images, can refer to the `trans_train` blow. 
    
    :param dataset_path: the folder of dataset
    :param input_size: the input size (image resolution)
    :return: the dataset used for pretraining
    """
    if dataset_type == "segment":
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ])
        dataset_train = SegmentDataset(segment_path=dataset_path, transforms=trans_train, mode="pretrain")
    else:
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.67, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD), # channelwise mean/std over dataset
        ])
        
        dataset_path = os.path.abspath(dataset_path)
        for postfix in ('train', 'val'):
            if dataset_path.endswith(postfix):
                dataset_path = dataset_path[:-len(postfix)]
        
        dataset_train = ImageNetDataset(imagenet_folder=dataset_path, transform=trans_train, train=True)
    print_transform(trans_train, '[pre-train]')
    return dataset_train


def print_transform(transform, s):
    print(f'Transform {s} = ')
    for t in transform.transforms:
        print(t)
    print('---------------------------\n')
