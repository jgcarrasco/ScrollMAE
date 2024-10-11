import copy
from itertools import product

import torch
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

        self.transforms = transforms

        if not stride:
            stride = crop_size
        self.crop_size = crop_size
        self.mode = mode
        print("Loading volume...")
        self.volume = Volume(segment_id, normalize=True, cache=True) # (depth, height, width)
        # take the z_depth central layers
        z_i = self.volume.shape(0)[0] // 2 - z_depth // 2
        z_f = z_i + z_depth
        self.z_i = z_i; self.z_f = z_f
        # It is faster to load all the segment at once than loading crop by crop
        # self.segment = self.volume[z_i:z_f, :, :]
        # NOTE: Temporary fix as the vesuvius library sometimes loads RGB tensors
        if len(self.volume.inklabel.shape) == 3:
            self.inklabel = rgb2gray(self.volume.inklabel)
            if self.inklabel.max() == 255: self.inklabel /= 255.
        else:
            self.inklabel = self.volume.inklabel / 255.

        self.h, self.w = self.volume.shape(0)[1:]
        self.crop_pos = []
        print("Computing crops...")
        for i, j in tqdm(
            list(product(range(0, self.h - crop_size, stride), 
                    range(0, self.w - crop_size, stride)))):
            # TODO: the criteria to select crops should be improved
            if self.mode == "supervised":
                if self.inklabel[i:i+crop_size, j:j+crop_size].mean() > 0.05: # at least 5% of ink
                    self.crop_pos.append((i, j))
            else:
                raise Exception
                if self.segment[:, i:i+crop_size, j:j+crop_size].mean() > 0:
                    self.crop_pos.append((i, j))
        
    def __len__(self):
        return len(self.crop_pos)

    def __getitem__(self, idx):
        i, j = self.crop_pos[idx]
        crop = self.volume[self.z_i:self.z_f, i:i+self.crop_size, j:j+self.crop_size]
        crop = torch.tensor(crop, dtype=torch.float32)
        if self.transforms:
            crop = self.transforms(crop)
        if self.mode == "pretrain":
            return crop
        else:
            label = self.inklabel[i:i+self.crop_size, j:j+self.crop_size]
            return torch.tensor(i), torch.tensor(j), \
                crop, torch.tensor(label, dtype=torch.float32)
        

def train_val_split(dataset, p_train=0.9):
    crop_pos = dataset.crop_pos
    p_train = 0.8
    n_cut = int(p_train*len(dataset.crop_pos))
    train_dataset = copy.copy(dataset)
    train_dataset.crop_pos = crop_pos[:n_cut]

    val_dataset = copy.copy(dataset)
    val_dataset.crop_pos = crop_pos[n_cut:]

    return train_dataset, val_dataset
