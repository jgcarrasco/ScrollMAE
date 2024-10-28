from typing import List, Optional
import copy
from itertools import product
import tempfile
import os 
import pickle
import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.amp import autocast
import vesuvius
from torch.utils.data import Dataset
from tqdm import tqdm
from vesuvius import Volume
from skimage.color import rgb2gray


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

    def __init__(self, segment_id=20231210121321, crop_size=320, z_depth=20, stride=None, mode="pretrain", transforms=None):

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
        self.crop_size = crop_size
        self.mode = mode
        print("Loading volume...")
        self.volume = Volume(segment_id, normalize=True, cache=True, cache_pool=1e11) # (depth, height, width)
        # take the z_depth central layers
        z_i = self.volume.shape(0)[0] // 2 - z_depth // 2
        z_f = z_i + z_depth
        self.z_i = z_i; self.z_f = z_f; self.z_depth = z_depth
        # It is faster to load all the segment at once than loading crop by crop
        # self.segment = self.volume[z_i:z_f, :, :]
        # NOTE: Temporary fix as the vesuvius library sometimes loads RGB tensors
        if len(self.volume.inklabel.shape) == 3:
            self.inklabel = rgb2gray(self.volume.inklabel)
            if self.inklabel.max() == 255: self.inklabel /= 255.
        else:
            self.inklabel = self.volume.inklabel / 255.

        self.h, self.w = self.volume.shape(0)[1:]

        # Check if the segment has already been downloaded
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{segment_id}_{z_depth}.pkl")
        if os.path.exists(file_path):
            print(f"Loading segment from local disk: {file_path}")
            with open(file_path, "rb") as f:
                self.volume = pickle.load(f)
        else:
            print("Pre-downloading segment...")
            self.volume = self.volume[z_i:z_f, :, :]
            with open(file_path, "wb") as f:
                pickle.dump(self.volume, f)

        self.crop_pos = []
        print("Computing crops...")
        for i, j in tqdm(list(product(range(0, self.h - crop_size, stride), range(0, self.w - crop_size, stride)))):
            # TODO: the criteria to select crops should be improved
            if self.inklabel[i:i+crop_size, j:j+crop_size].mean() > 0.05: # at least 5% of ink
                self.crop_pos.append((i, j))
        
    def __len__(self):
        return len(self.crop_pos)

    def __getitem__(self, idx):
        i, j = self.crop_pos[idx]
        crop = self.volume[:, i:i+self.crop_size, j:j+self.crop_size]
        crop = np.transpose(crop, (1, 2, 0)).astype(np.float32)
        if self.mode == "pretrain":
            crop = self.transforms(image=crop)["image"]
            return crop
        else:
            label = self.inklabel[i:i+self.crop_size, j:j+self.crop_size][..., None].astype(np.float32)
            transformed = self.transforms(image=crop, mask=label)
            crop = transformed["image"]
            label = transformed["mask"][0]
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
                      savefig=True, show=False, checkpoint_type="last"):

    model = torch.load(f"checkpoints/{checkpoint_name}/{checkpoint_type}_{checkpoint_name}.pth", weights_only=False)
    device = next(model.parameters()).device
    # Initialize predictions and counters with the same shape as the cropped ink label segment
    letter_predictions = np.zeros_like(dataset.inklabel, dtype=np.float32)
    counter_predictions = np.zeros_like(dataset.inklabel, dtype=np.float32)

    # Set the model to evaluation mode
    model.eval()
    for dataloader in dataloaders:
        with torch.no_grad():
            for i, j, crops, labels in dataloader:
                # Move the data and labels to the GPU
                crops, labels = crops.cuda(), labels.float().cuda()
                
                # Forward pass to get model predictions
                with autocast(device_type=device.type):
                    outputs = model(crops)

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