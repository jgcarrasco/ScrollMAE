import os
import sys
from datetime import datetime

import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Append the root directory (where dataset.py is located)
sys.path.append(root_dir)
from dataset import inference_segment, train_val_split
from model import UNet, UNet3D

exp_name = "scrollmae"
BATCH_SIZE = 16
NUM_EPOCHS = 5
validate_every = -1
segment_id = 20230827161847 # 20230827161847 (small) 20231210121321 (large) 20231210132040 (scroll 4)
pretrained_path = "pretrain_checkpoints/3d_30l_20230827161847/resnet_3d_50_1kpretrained_timm_style.pth"
scheme = "iterative" # "validation" | "iterative"
inklabel_path = "data/20230827161847.zarr/iterative_inklabels/1.png"
mask_path = None
save_path = "data/20230827161847.zarr/iterative_inklabels/1_output.png"

# Less commonly changed arguments
model_name = "unet3d"
n_layers = 30
crop_size = 256
stride_fraction = 8
freeze_encoder = False 
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
print("Loading segments...")
train_dataset, val_dataset = train_val_split(segment_id=segment_id, crop_size=crop_size, 
                                            stride_fraction=stride_fraction, z_depth=n_layers,
                                            scale_factor=scale_factor, transforms_=transforms_, 
                                            mode="supervised", schema=scheme, inklabel_path=inklabel_path,
                                            mask_path=mask_path)
print(f"Number of crops in train: {len(train_dataset)}")
print(f"Number of crops in val: {len(val_dataset)}")
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
bce = SoftBCEWithLogitsLoss(smooth_factor=0.15)
dice = DiceLoss(mode="binary")
criterion = lambda y_pred, y: 0.5 * bce(y_pred, y) + 0.5 * dice(y_pred, y)

if freeze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False
    params = model.decoder.parameters()
else:
    params = model.parameters()

optimizer = optim.AdamW(params, lr=1e-4, weight_decay=1e-6) # weight_decay=1e-6
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = GradScaler()

writer = SummaryWriter(log_dir=f"runs/{checkpoint_name}/{current_time}")
model.train()
best_loss = 1e9
# Training loop
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    
    for i, j, crops, labels in tqdm(train_dataloader, 
                                    total=len(train_dataloader),
                                    desc="Training...",
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
    # Logging
    train_loss = running_loss/len(train_dataloader)    
    writer.add_scalar("Loss/train", train_loss, epoch+1)
    if ((epoch) % validate_every == 0) and (validate_every != -1):
        val_loss = 0.
        with torch.no_grad():
            for _, _, crops, labels in tqdm(val_dataloader, 
                                            total=len(val_dataloader),
                                            desc="Validating...",
                                            leave=False):
                crops, labels = crops.cuda(), labels.cuda()
                # Forward and backward passes with mixed precision
                with autocast(device_type=device.type):
                    outputs = model(crops)
                    val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_dataloader)
        writer.add_scalar("Loss/val", val_loss, epoch+1)
        if val_loss <= best_loss:
            torch.save(model, f"checkpoints/{checkpoint_name}/best_{checkpoint_name}.pth")
            inference_segment(checkpoint_name, val_dataset, [val_dataloader], checkpoint_type="best")
        torch.save(model, f"checkpoints/{checkpoint_name}/last_{checkpoint_name}.pth")
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train loss: {train_loss:.4f} Val loss: {val_loss:.4f}')
    else:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train loss: {train_loss:.4f}')
print("Training completed.")
torch.save(model, f"checkpoints/{checkpoint_name}/last_{checkpoint_name}.pth")
inference_segment(checkpoint_name, val_dataset, [val_dataloader], checkpoint_type="last", save_path=save_path)
