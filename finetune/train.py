import copy
import os
import sys
from datetime import datetime
from itertools import product
from typing import List

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import vesuvius
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vesuvius import Volume

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the root directory (where dataset.py is located)
sys.path.append(root_dir)
from dataset import SegmentDataset, inference_segment, train_val_split
from model import UNet, UNet3D

exp_name = "it"
BATCH_SIZE = 32
NUM_EPOCHS = 100
validate_every = -1
segment_id = 20230827161847 # 20230827161847 20231210121321
model_name = "unet3d"
n_layers = 20
crop_size = 224
freeze_encoder = False
pretrained_path = "pretrain_checkpoints/3d_20230827161847/resnet_3d_50_1kpretrained_timm_style.pth" #"pretrain_checkpoints/20231210121321/resnet50_1kpretrained_timm_style.pth"
scheme = "iterative" # "validation" | "iterative"
inklabel_path = "data/20230827161847.zarr/iterative_inklabels/1.png"

# Less commonly changed arguments
data_augmentation = True
scale_factor = 0.25
clip_value = 10.0
current_time = datetime.now().strftime("%b%d_%H-%M-%S")

backbone_kwargs = None

checkpoint_name = f"{exp_name}_{model_name}_{segment_id}"
if pretrained_path:
    backbone_kwargs = {}
    backbone_kwargs["checkpoint_path"] = pretrained_path 
    checkpoint_name += "_pretrained"
if freeze_encoder:
    checkpoint_name += "_frozen"
if data_augmentation: checkpoint_name += "_aug"
if not os.path.exists(f"checkpoints/{checkpoint_name}"):
    os.makedirs(f"checkpoints/{checkpoint_name}")

if data_augmentation:
    transforms_ = A.Compose([
        A.RandomResizedCrop(crop_size, crop_size, scale=(0.67, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.CoarseDropout(max_holes=2, max_width=int(crop_size * 0.2), max_height=int(crop_size * 0.2), 
                        mask_fill_value=0, p=0.5),
        A.Normalize(
            mean= [0] * n_layers,
            std= [1] * n_layers
        ),
        ToTensorV2(transpose_mask=True),
    ])
og_crop_size = crop_size / 0.7 # as we do a random resized crop, we start with larger crops
if scheme == "validation":
    dataset = SegmentDataset(segment_id=segment_id, mode="supervised", 
                            crop_size=og_crop_size, stride= og_crop_size // 3, transforms=transforms_, 
                            z_depth=n_layers, scale_factor=scale_factor)
    train_dataset, val_dataset = train_val_split(dataset, criteria="ink")
elif scheme == "iterative":
    dataset = SegmentDataset(segment_id=segment_id, mode="supervised", 
                            crop_size=og_crop_size, stride= og_crop_size // 3, transforms=transforms_, 
                            z_depth=n_layers, scale_factor=scale_factor, criteria="mask")
    val_dataset = copy.copy(dataset)
    train_dataset = copy.copy(val_dataset)
    train_dataset.set_inklabel(fp=inklabel_path)
    train_dataset.recompute_crop_pos(criteria="ink")
# Create the DataLoader for batch processing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check if a GPU is available and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model_name == "unet":
    model = UNet(in_chans=n_layers, backbone_kwargs=backbone_kwargs)
elif model_name == "unet3d":
    model = UNet3D(in_chans=n_layers, backbone_kwargs=backbone_kwargs)

model = model.to(device)
bce = SoftBCEWithLogitsLoss(smooth_factor=0.1)
dice = DiceLoss(mode="binary")
criterion = lambda y_pred, y: 0.5 * bce(y_pred, y) + 0.5 * dice(y_pred, y)

if freeze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False
    params = model.decoder.parameters()
else:
    params = model.parameters()

optimizer = optim.AdamW(params, lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler()

writer = SummaryWriter(log_dir=f"runs/{checkpoint_name}/{current_time}")

model.train()
best_loss = 1e9
# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    
    for i, j, crops, labels in tqdm(train_dataloader,
                                    desc="Training...",
                                    total=len(train_dataloader), 
                                    leave=False):
        crops, labels = crops.cuda(), labels.cuda()
        # Forward and backward passes with mixed precision
        with autocast(device_type=device.type):
            outputs = model(crops)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        # Scale the loss, compute gradients
        scaler.scale(loss).backward()
        # Unscale gradients and then clip
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), clip_value)
        scaler.step(optimizer)
        scaler.update()
        # Track running loss for the epoch
        running_loss += loss.item()

    scheduler.step()
    if (epoch+1) % validate_every == 0:
        val_loss = 0.
        with torch.no_grad():
            for _, _, crops, labels in tqdm(val_dataloader,
                                            desc="Validating...",
                                            total=len(val_dataloader),
                                            leave=False):
                crops, labels = crops.cuda(), labels.cuda()
                # Forward and backward passes with mixed precision
                with autocast(device_type=device.type):
                    outputs = model(crops)
                    val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_dataloader)
        train_loss = running_loss/len(train_dataloader)
        
        writer.add_scalar("Loss/train", train_loss, epoch+1)
        writer.add_scalar("Loss/val", val_loss, epoch+1)
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}')

        if val_loss <= best_loss:
            torch.save(model, f"checkpoints/{checkpoint_name}/best_{checkpoint_name}.pth")
            inference_segment(checkpoint_name, dataset, [val_dataloader], checkpoint_type="best")
        torch.save(model, f"checkpoints/{checkpoint_name}/last_{checkpoint_name}.pth")

print("Training completed.")

dataset = SegmentDataset(segment_id=segment_id, mode="supervised", 
                         crop_size=crop_size, stride=crop_size // 3, transforms=None, z_depth=n_layers)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
inference_segment(checkpoint_name, dataset, [dataloader], checkpoint_type="final")
