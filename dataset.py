import copy
import os
import subprocess
from itertools import product
from typing import List, Optional

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from albumentations.pytorch import ToTensorV2
from skimage.color import rgb2gray
from skimage.io import imread
from torch.amp import autocast
from torch.utils.data import Dataset
from tqdm import tqdm


def load_img(fp):
        img = imread(fp)
        if len(img.shape) == 3:
            img = rgb2gray(img)
            if img.max() == 255: img /= 255.
        else:
            img = img / 255.
        return img


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

    def __init__(self, segment_id=20231210121321, crop_size=320, z_depth=20, stride=None, mode="pretrain", 
                 transforms=None, scale_factor=None, data_path="data", criteria="ink"):

        self.scale_factor = scale_factor
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = A.Compose([
                A.Normalize(
                    mean= [0] * z_depth,
                    std= [1] * z_depth
                ),
                ToTensorV2(transpose_mask=True),
            ])

        if not stride:
            stride = crop_size
        self.stride = stride
        self.crop_size = crop_size
        self.mode = mode
        print("Loading volume...")
        file_path = os.path.join(data_path, f"{segment_id}.zarr")
        if not os.path.exists(file_path):
            print("Pre-downloading segment...")
            subprocess.run(['./download_segment.sh', str(segment_id)])

        self.volume = zarr.load(file_path)[0] / 255. # (depth, height, width)
        self.inklabel = load_img(f"{file_path}/{segment_id}_inklabels.png")
        self.mask = load_img(f"{file_path}/{segment_id}_mask.png")
        
        # take the z_depth central layers
        z_i = self.volume.shape[0] // 2 - z_depth // 2
        z_f = z_i + z_depth
        self.z_i = z_i; self.z_f = z_f; self.z_depth = z_depth
        self.h, self.w = self.volume.shape[1:]

        print("Computing crops...")
        self.crop_pos = self.compute_crop_pos(criteria)
    
    def __len__(self):
        return len(self.crop_pos)

    def __getitem__(self, idx):
        i, j = self.crop_pos[idx]
        crop = self.volume[self.z_i:self.z_f, i:i+self.crop_size, j:j+self.crop_size]
        crop = np.transpose(crop, (1, 2, 0)).astype(np.float32)
        if self.mode == "pretrain":
            crop = self.transforms(image=crop)["image"]
            return crop
        else:
            label = self.inklabel[i:i+self.crop_size, j:j+self.crop_size][..., None].astype(np.float32)
            transformed = self.transforms(image=crop, mask=label)
            crop = transformed["image"]
            label = transformed["mask"][0]
            if self.scale_factor:
                label = F.interpolate(label[None, None], scale_factor=self.scale_factor)[0, 0]
            return torch.tensor(i), torch.tensor(j), crop, label
        
    def set_transforms(self, transforms: Optional[A.Compose]):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = A.Compose([
                A.Normalize(
                    mean= [0] * self.z_depth,
                    std= [1] * self.z_depth
                ),
                ToTensorV2(transpose_mask=True),
            ])

    def compute_crop_pos(self, criteria):
        crop_pos = []
        if criteria == "ink":
            for i, j in tqdm(list(product(range(0, self.h - self.crop_size, self.stride), range(0, self.w - self.crop_size, self.stride)))):
                if self.inklabel[i:i+self.crop_size, j:j+self.crop_size].mean() > 0.05: # at least 5% of ink
                    crop_pos.append((i, j))
        else:
           for i, j in tqdm(list(product(range(0, self.h - self.crop_size, self.stride), range(0, self.w - self.crop_size, self.stride)))):
                if self.mask[i:i+self.crop_size, j:j+self.crop_size].min() > 0.0: # the crop is fully inside the mask
                    crop_pos.append((i, j)) 
        return crop_pos
    
    def recompute_crop_pos(self, criteria):
        self.crop_pos = self.compute_crop_pos(criteria)

def train_val_split(dataset, p_train=0.9):
    crop_pos = dataset.crop_pos
    p_train = 0.8
    n_cut = int(p_train*len(dataset.crop_pos))
    train_dataset = copy.copy(dataset)
    train_dataset.crop_pos = crop_pos[:n_cut]

    val_dataset = copy.copy(dataset)
    val_dataset.crop_pos = crop_pos[n_cut:]
    val_dataset.set_transforms(transforms=None)

    return train_dataset, val_dataset


def inference_segment(checkpoint_name: str, dataset: SegmentDataset, dataloaders: List,
                      mask_dataloader=None, savefig=True, show=False, checkpoint_type="last"):

    model = torch.load(f"checkpoints/{checkpoint_name}/{checkpoint_type}_{checkpoint_name}.pth", weights_only=False)
    device = next(model.parameters()).device
    # Initialize predictions and counters with the same shape as the cropped ink label segment
    letter_predictions = np.zeros_like(dataset.inklabel, dtype=np.float32)
    counter_predictions = np.zeros_like(dataset.inklabel, dtype=np.float32)

    # Set the model to evaluation mode
    model.eval()
    for dataloader in dataloaders:
        with torch.no_grad():
            for i, j, crops, labels in tqdm(dataloader, total=len(dataloader)):
                # Move the data and labels to the GPU
                crops, labels = crops.cuda(), labels.float().cuda()
                
                # Forward pass to get model predictions
                with autocast(device_type=device.type):
                    outputs = model(crops)
                    if sf := dataloader.dataset.scale_factor:
                        outputs = F.interpolate(outputs[:, None], scale_factor=1./sf)[:, 0]

                # Apply sigmoid to get probabilities from logits
                predictions = torch.sigmoid(outputs)
                # Process each prediction and update the corresponding regions
                for ii, jj, prediction in zip(i, j, predictions):
                    ii, jj = ii.item(), jj.item()
                    #crop_size = dataset.crop_size
                    crop_size = prediction.shape[-1] # TODO: This is not the cleanest way to do this
                    prediction = prediction.cpu().numpy() # Convert to NumPy array
                    letter_predictions[ii:ii+crop_size, jj:jj+crop_size] += prediction
                    counter_predictions[ii:ii+crop_size, jj:jj+crop_size] += 1

    # Avoid division by zero by setting any zero counts to 1
    counter_predictions[counter_predictions == 0] = 1

    # Normalize the predictions by the counter values
    letter_predictions /= counter_predictions

    if mask_dataloader:
        with torch.no_grad():
            for i, j, crops, labels in tqdm(mask_dataloader, total=len(mask_dataloader)):
                # Move the data and labels to the GPU
                crops, labels = crops.cuda(), labels.float().cuda()
                
                # Forward pass to get model predictions
                with autocast(device_type=device.type):
                    outputs = model(crops)
                    if sf := dataloader.dataset.scale_factor:
                        outputs = F.interpolate(outputs[:, None], scale_factor=1./sf)[:, 0]

                # Apply sigmoid to get probabilities from logits
                predictions = torch.sigmoid(outputs)
                # Process each prediction and update the corresponding regions
                for ii, jj, prediction in zip(i, j, predictions):
                    ii, jj = ii.item(), jj.item()
                    #crop_size = dataset.crop_size
                    crop_size = prediction.shape[-1] # TODO: This is not the cleanest way to do this
                    prediction = prediction.cpu().numpy() # Convert to NumPy array
                    letter_predictions[ii:ii+crop_size, jj:jj+crop_size] = 0.

    # Plotting the Ground Truth and Model Predictions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Ground Truth Label
    ax = axes[0]
    ax.imshow(dataset.inklabel, cmap='gray')
    ax.set_title('Ground Truth Label')
    ax.axis('off')


    # Model Prediction
    ax = axes[1]
    ax.imshow(letter_predictions, cmap='gray')
    ax.set_title('Model Prediction')
    ax.axis('off')
    plt.tight_layout()
    # Display the plots
    if savefig:
        plt.savefig(f"checkpoints/{checkpoint_name}/{checkpoint_name}.jpg")
    if show:
        plt.show()