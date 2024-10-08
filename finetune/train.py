import copy
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vesuvius
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vesuvius import Volume


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
        self.segment = self.volume[z_i:z_f, :, :]
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
                if self.segment[:, i:i+crop_size, j:j+crop_size].mean() > 0:
                    self.crop_pos.append((i, j))
        
    def __len__(self):
        return len(self.crop_pos)

    def __getitem__(self, idx):
        i, j = self.crop_pos[idx]
        crop = self.segment[:, i:i+self.crop_size, j:j+self.crop_size]
        if self.transforms:
            crop = self.transforms(crop)
        if self.mode == "pretrain":
            return torch.tensor(crop, dtype=torch.float32)
        else:
            label = self.inklabel[i:i+self.crop_size, j:j+self.crop_size]
            return torch.tensor(i), torch.tensor(j), \
                torch.tensor(crop, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    

def train_val_split(dataset, p_train=0.9):
    crop_pos = dataset.crop_pos
    p_train = 0.8
    n_cut = int(p_train*len(dataset.crop_pos))
    train_dataset = copy.copy(dataset)
    train_dataset.crop_pos = crop_pos[:n_cut]

    val_dataset = copy.copy(dataset)
    val_dataset.crop_pos = crop_pos[n_cut:]

    return train_dataset, val_dataset




class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Z Convolution (Vertical Convolution along the depth dimension)
        #self.conv_z = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # Contracting path
        self.enc_conv1 = self.double_conv(20, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)
        self.enc_conv5 = self.double_conv(512, 1024)
        
        # Expansive path
        self.up_trans1 = self.up_conv(1024, 512)
        self.dec_conv1 = self.double_conv(1024, 512)
        self.up_trans2 = self.up_conv(512, 256)
        self.dec_conv2 = self.double_conv(512, 256)
        self.up_trans3 = self.up_conv(256, 128)
        self.dec_conv3 = self.double_conv(256, 128)
        self.up_trans4 = self.up_conv(128, 64)
        self.dec_conv4 = self.double_conv(128, 64)
        
        # Final output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # # Apply the Z convolution
        # x = self.conv_z(x)  # [batch_size, 64, depth, height, width]

        # # Aggregate the depth dimension using max pooling or average pooling
        # x, _ = torch.max(x, dim=2)  # Choose max or torch.mean(x, dim=2) for average pooling
        
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(F.max_pool2d(x1, kernel_size=2))
        x3 = self.enc_conv3(F.max_pool2d(x2, kernel_size=2))
        x4 = self.enc_conv4(F.max_pool2d(x3, kernel_size=2))
        x5 = self.enc_conv5(F.max_pool2d(x4, kernel_size=2))
        
        # Decoder
        x = self.up_trans1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec_conv1(x)
        
        x = self.up_trans2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_conv2(x)
        
        x = self.up_trans3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_conv3(x)
        
        x = self.up_trans4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv4(x)
        
        x = self.out_conv(x) # (batch_size, 1, h, w)
        
        return x[:, 0, :, :]
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


# This function initializes the weights in an intelligent way
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


dataset = SegmentDataset(segment_id=20230827161847, mode="supervised", crop_size=256, stride=128)
train_dataset, val_dataset = train_val_split(dataset)

# Create the DataLoader for batch processing
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Check if a GPU is available and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # Force using CPU (sometimes good for debugging)
model = UNet()
initialize_weights(model)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
NUM_EPOCHS = 1000
# Scheduler - Cosine Annealing LR
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Define gradient clipping value
clip_value = 10.0 

# Initialize GradScaler for mixed precision training
scaler = GradScaler()
model.train()
# Training loop
NUM_EPOCHS = 200
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    
    for i, j, crops, labels in train_dataloader:
        # Move the data to GPU
        crops, labels = crops.cuda(), labels.cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward and backward passes with mixed precision
        with autocast(device_type=device.type):
            # Forward pass
            outputs = model(crops)
            # Calculate loss
            loss = criterion(outputs, labels)
        # Scale the loss, compute gradients
        scaler.scale(loss).backward()
        # Unscale gradients and then clip
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), clip_value)

        # Step the optimizer
        scaler.step(optimizer)
        scaler.update()

        # Track running loss for the epoch
        running_loss += loss.item()

    # Scheduler step
    scheduler.step()
    if (epoch+1)%10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_dataloader):.4f}')

    # if running_loss/len(dataloader) < 0.08:
    #     print(f"Final loss {running_loss/len(dataloader)}")
    #     break
print("Training completed.")



