import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def align_tensor_image(x: torch.Tensor, y: torch.Tensor):
    delta_height = y.size(2) - x.size(2)
    delta_width = y.size(3) - x.size(3)
    x = F.pad(x, (delta_width // 2, delta_width - delta_width // 2, delta_height // 2, delta_height - delta_height // 2))
    return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim=128, timesteps=1000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.emb_size = timesteps
        if emb_dim % 2 != 0:
            raise ValueError("Positional embedding dimension must be divisible by 2.")
        position = torch.arange(timesteps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / emb_dim))
        embedding = torch.zeros(timesteps, emb_dim, requires_grad=False)
        embedding[:, 0::2] = torch.sin(position * div)
        embedding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("embedding", embedding)


    def forward(self, timestep, device):
        embed = self.embedding[timestep].to(device)
        return embed


class Attention(nn.Module):
    def __init__(self, channels: int, num_heads: int, dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(channels, channels*3)
        self.proj2 = nn.Linear(channels, channels)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')


class NormActConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=32):
        super(NormActConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32, t_emb_dim=None):
        super(ResnetBlock, self).__init__()
        self.feature_block1 = NormActConv(in_channels, out_channels, num_groups)
        self.feature_block2 = NormActConv(out_channels, out_channels, num_groups)
        self.time_emb_dim = t_emb_dim

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

        if t_emb_dim is not None:
            self.time_block = nn.Sequential(
                nn.Linear(t_emb_dim, out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.time_block = None

    def forward(self, x, time):
        residue = x
        x = self.feature_block1(x)
        time = self.time_block(time)
        time_feature = x + time.unsqueeze(-1).unsqueeze(-1)
        time_feature += self.residual_conv(residue)
        x = self.feature_block2(time_feature)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim=None, use_attention: bool = False, num_heads: int = 8, dropout_prob: float = 0.1):
        super(DownBlock, self).__init__()
        self.resnet_block1 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, t_emb_dim=t_emb_dim)
        self.resnet_block2 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, t_emb_dim=t_emb_dim)
        self.attention = Attention(channels=in_channels, num_heads=num_heads, dropout_prob=dropout_prob)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.use_attention = use_attention

    def forward(self, x, time):
        x = self.resnet_block1(x, time)
        if self.use_attention:
            x = self.attention(x)
        x = self.resnet_block2(x, time)
        return self.conv(x), x # downsampled output, residual connection


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim=None, use_attention: bool = False, num_heads: int = 8, dropout_prob: float = 0.1):
        super(UpBlock, self).__init__()
        self.resnet_block1 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, t_emb_dim=t_emb_dim)
        self.resnet_block2 = ResnetBlock(in_channels=in_channels, out_channels=in_channels, t_emb_dim=t_emb_dim)
        self.attention = Attention(channels=in_channels, num_heads=num_heads, dropout_prob=dropout_prob)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.use_attention = use_attention

    def forward(self, x, time):
        x = self.resnet_block1(x, time)
        if self.use_attention:
            x = self.attention(x)
        x = self.resnet_block2(x, time)
        x = self.conv_transpose(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int = 256, class_emb_dim: int = 8, base_channels: int = 32, timesteps=1000, num_classes=8):
        super(UNet, self).__init__()
        self.in_conv = NormActConv(in_channels + class_emb_dim, base_channels)
        self.time_embedding = SinusoidalPositionalEmbedding(emb_dim=t_emb_dim, timesteps=timesteps)
        self.class_embedding = nn.Embedding(num_classes, class_emb_dim)

        self.down_block1 = DownBlock(in_channels=base_channels, out_channels=2 * base_channels, t_emb_dim=t_emb_dim)
        self.down_block2 = DownBlock(in_channels=2 * base_channels, out_channels=4 * base_channels, t_emb_dim=t_emb_dim, use_attention=True)
        #self.down_block3 = DownBlock(in_channels=4 * base_channels, out_channels=8 * base_channels, t_emb_dim=t_emb_dim)
        self.down_blocks = nn.ModuleList([self.down_block1, self.down_block2])

        self.up_block1 = UpBlock(in_channels=4 * base_channels, out_channels=2 * base_channels, t_emb_dim=t_emb_dim)
        # self.up_block2 = UpBlock(in_channels=4 * base_channels + 4 * base_channels, out_channels=4 * base_channels, t_emb_dim=t_emb_dim, use_attention=True)
        self.up_block2 = UpBlock(in_channels=2 * base_channels + 2 * base_channels, out_channels=2 * base_channels, t_emb_dim=t_emb_dim, use_attention=True)
        #self.up_block3 = UpBlock(in_channels=4 * base_channels + 2 * base_channels, out_channels=2 * base_channels, t_emb_dim=t_emb_dim, use_attention=True)
        self.up_blocks = nn.ModuleList([self.up_block1, self.up_block2])

        self.out_conv = nn.Sequential(
            NormActConv(2 * base_channels + base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, t, c):
        bs, ch, w, h = x.shape
        time_embedding = self.time_embedding(t, x.device)
        class_embedding = self.class_embedding(c)
        class_embedding = class_embedding.view(bs, class_embedding.shape[1], 1, 1).expand(bs, class_embedding.shape[1], w, h)
        x = torch.cat([x, class_embedding], dim=1)
        x = self.in_conv(x)
        skip_connections = []
        for block in self.down_blocks:
            x, skip_connection = block(x, time_embedding)
            skip_connections.append(skip_connection)

        for block in self.up_blocks:
            skip_connection = skip_connections.pop()
            x = block(x, time_embedding)
            x = align_tensor_image(x, skip_connection)
            x = torch.cat([x, skip_connection], dim=1)

        return self.out_conv(x)
