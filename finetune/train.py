import copy
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.nn as nn
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
from models import UNet, VanillaUNet

exp_name = "random"
segment_id = 20230827161847 # 20231210121321
BATCH_SIZE = 32
NUM_EPOCHS = 50
clip_value = 10.0
model_name = "vanilla"

checkpoint_name = f"{exp_name}_{model_name}_{segment_id}"

dataset = SegmentDataset(segment_id=segment_id, mode="supervised", crop_size=256, stride=128)
train_dataset, val_dataset = train_val_split(dataset)

# Create the DataLoader for batch processing
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check if a GPU is available and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet() if model_name == "unet" else VanillaUNet()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler()

writer = SummaryWriter(log_dir=f"runs/{checkpoint_name}/")

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
    if (epoch+1)%1 == 0:
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
            torch.save(model, f"checkpoints/best_{checkpoint_name}.pth")
        torch.save(model, f"checkpoints/last_{checkpoint_name}.pth")

print("Training completed.")

model = torch.load(f"checkpoints/last_{checkpoint_name}.pth", weights_only=False)

# Visualize training
import matplotlib.pyplot as plt

# Initialize predictions and counters with the same shape as the cropped ink label segment
letter_predictions = np.zeros_like(dataset.volume.inklabel, dtype=np.float32)
counter_predictions = np.zeros_like(dataset.volume.inklabel, dtype=np.float32)

# Set the model to evaluation mode
model.eval()
# Disable gradient calculations for validation to save memory and computations
with torch.no_grad():
    for i, j, crops, labels in val_dataloader:
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
            crop_size = dataset.crop_size
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
ax.imshow(dataset.volume.inklabel, cmap='gray')
ax.set_title('Ground Truth Label')
ax.axis('off')


# Model Prediction
ax = axes[1]
ax.imshow(letter_predictions, cmap='gray')
ax.set_title('Model Prediction')
ax.axis('off')

# Display the plots
plt.savefig(f"checkpoints/last_{exp_name}.jpg")


