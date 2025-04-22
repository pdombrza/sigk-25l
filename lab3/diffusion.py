import torch
import torch.nn as nn
import lightning as L
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
from diffusion_utils import DDPMSampler
from unet import UNet


class DiffusionModel(L.LightningModule):
    def __init__(self, scheduler: DDPMSampler | None = DDPMSampler, unet: UNet | None = UNet):
        super(DiffusionModel, self).__init__()
        self.scheduler = scheduler if scheduler else DDPMSampler()
        self.unet = unet if unet else UNet()
        self.loss_fn = nn.MSELoss()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        images, _ = batch
        batch_size = images.shape[0]
        optimizer = self.optimizers()
        optimizer.zero_grad()
        timesteps = torch.randint(0, self.scheduler.n_timesteps, (batch_size,), device=self.device)
        noisy_images, noise = self.scheduler.add_noise(images, timesteps)
        predicted_noise = self.unet(noisy_images, timesteps)
        loss = self.loss_fn(predicted_noise, noise)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        optimizer.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return

    def forward(self, x):
        for t in tqdm(range(self.scheduler.n_timesteps - 1, -1, -1)):
            ts = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            predicted_noise = self.unet(x, ts)
            x = self.scheduler.denoise_step(x, t, predicted_noise)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        return optimizer