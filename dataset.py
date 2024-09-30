import zarr
import cv2

import torch
from torch.utils.data import Dataset

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