import copy
import os
import sys
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vesuvius import Volume

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the root directory (where dataset.py is located)
sys.path.append(root_dir)
from dataset import SegmentDataset, train_val_split
from models import UNet


segment_id = 20230827161847
BATCH_SIZE = 32
NUM_EPOCHS = 1000
clip_value = 10.0 

dataset = SegmentDataset(segment_id=segment_id, mode="supervised", crop_size=256, stride=128)
train_dataset, val_dataset = train_val_split(dataset)

# Create the DataLoader for batch processing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check if a GPU is available and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler()

writer = SummaryWriter()

model.train()
best_loss = 1e9
# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    
    for i, j, crops, labels in train_dataloader:
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
    if (epoch+1)%10 == 0:
        val_loss = 0.
        with torch.no_grad():
            for _, _, crops, labels in val_dataloader:
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
            torch.save(model, "checkpoints/best_model.pth")

print("Training completed.")



