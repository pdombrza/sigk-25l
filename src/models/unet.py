import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        return self.down_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, x_skip_conn):
        x = self.upsample(x)
        delta_height = x_skip_conn.size(2) - x.size(2)
        delta_width = x_skip_conn.size(3) - x.size(3)
        x = F.pad(x, (delta_width // 2, delta_width - delta_width // 2, delta_height // 2, delta_height - delta_height // 2))
        x = torch.cat([x, x_skip_conn], dim=1)
        return self.up_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_unet_blocks = 2):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.down_blocks = nn.ModuleList([DownBlock(32*2**i, 32*2**(i+1)) for i in range(num_unet_blocks)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32*2**num_unet_blocks, 32*2**num_unet_blocks, kernel_size=3, padding=1),
            nn.BatchNorm2d(32*2**num_unet_blocks),
            nn.Dropout2d(0.5, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32*2**num_unet_blocks, 32*2**num_unet_blocks, kernel_size=3, padding=1),
            nn.BatchNorm2d(32*2**num_unet_blocks),
            nn.Dropout2d(0.5, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.up_blocks = nn.ModuleList([UpBlock(32*2**(num_unet_blocks-i), 32*2**(num_unet_blocks-i-1)) for i in range(num_unet_blocks)])
        self.out_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu(x)
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x)
        x = self.bottleneck(x)
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections.pop())
        x = self.out_layers(x)
        return x