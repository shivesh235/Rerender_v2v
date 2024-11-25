import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from glob import glob
from PIL import Image

# Set random seeds
random.seed(10)
torch.manual_seed(10)

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 64
MAX_TRAIN_IMAGES = 300
LEARNING_RATE = 1e-4

# Image preprocessing
def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    ])
    return transform(image)

# Dataset class
class LowLightDataset(Dataset):
    def __init__(self, low_images, high_images):
        self.low_images = low_images
        self.high_images = high_images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image = read_image(self.low_images[idx])
        high_image = read_image(self.high_images[idx])
        return low_image, high_image

# Data loaders
train_low_images = sorted(glob("lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
train_high_images = sorted(glob("lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]

#train_dataset = LowLightDataset(train_low_images, train_high_images)
#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Custom layers
class ChannelPooling(nn.Module):
    def __init__(self, axis=-1):
        super(ChannelPooling, self).__init__()
        self.axis = axis

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True).values
        return torch.cat([avg_pool, max_pool], dim=1)

class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        compressed = ChannelPooling()(x)
        feature_map = torch.sigmoid(self.conv(compressed))
        return x * feature_map

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(ChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        feature_map = F.relu(self.conv1(avg_pool))
        feature_map = torch.sigmoid(self.conv2(feature_map))
        return x * feature_map

class DualAttentionUnit(nn.Module):
    def __init__(self, channels):
        super(DualAttentionUnit, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.channel_attention = ChannelAttentionBlock(channels)
        self.spatial_attention = SpatialAttentionBlock()
        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        feature_map = F.relu(self.conv1(x))
        feature_map = self.conv2(feature_map)
        channel_attention = self.channel_attention(feature_map)
        spatial_attention = self.spatial_attention(feature_map)
        concatenation = torch.cat([channel_attention, spatial_attention], dim=1)
        return x + self.conv3(concatenation)

class MIRNet(nn.Module):
    def __init__(self, num_rrg, num_mrb, channels):
        super(MIRNet, self).__init__()
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.rrg_blocks = nn.ModuleList([
            self._build_rrg(channels, num_mrb) for _ in range(num_rrg)
        ])
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def _build_rrg(self, channels, num_mrb):
        return nn.Sequential(
            *[self._build_mrb(channels) for _ in range(num_mrb)]
        )

    def _build_mrb(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            DualAttentionUnit(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.conv_in(x)
        for rrg in self.rrg_blocks:
            x1 = rrg(x1)
        return x + self.conv_out(x1)

# Loss function
def charbonnier_loss(y_true, y_pred):
    epsilon = 1e-3
    diff = torch.sqrt((y_true - y_pred) ** 2 + epsilon ** 2)
    return torch.mean(diff)

# Initialize model, optimizer, and loss
model = MIRNet(num_rrg=3, num_mrb=2, channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
