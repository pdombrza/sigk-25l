from datetime import timedelta
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
from diffusion_utils import DDPMSampler
from unet import UNet

torch.set_float32_matmul_precision('medium')


class DiffusionModel(L.LightningModule):
    def __init__(self, scheduler: DDPMSampler | None = None, unet: UNet | None = None, num_classes: int = 8):
        super(DiffusionModel, self).__init__()
        self.scheduler = scheduler if scheduler else DDPMSampler()
        self.unet = unet if unet else UNet(in_channels=3, out_channels=3, num_classes=num_classes)
        self.loss_fn = nn.MSELoss()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze(-1)
        batch_size = images.shape[0]
        optimizer = self.optimizers()
        optimizer.zero_grad()
        timesteps = torch.randint(0, self.scheduler.n_timesteps, (batch_size,), device=self.device)
        noisy_images, noise = self.scheduler.add_noise(images, timesteps)
        predicted_noise = self.unet(noisy_images, timesteps, labels)
        loss = self.loss_fn(predicted_noise, noise)
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        optimizer.step()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.squeeze(-1)
        batch_size = images.shape[0]
        timesteps = torch.randint(0, self.scheduler.n_timesteps, (batch_size,), device=self.device)
        noisy_images, noise = self.scheduler.add_noise(images, timesteps)
        predicted_noise = self.unet(noisy_images, timesteps, labels)
        loss = self.loss_fn(predicted_noise, noise)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            #generated_images = [self.forward(torch.randn(2, 3, 64, 64, device=self.device), torch.randint(0, 8, (batch_size, ), device=self.device)) for _ in range(2)]
            #generated_images = torch.stack(generated_images)

            self.logger.experiment.add_images("generated_images", images, self.current_epoch)
            self.logger.experiment.add_text("val_labels", str(labels.tolist()), self.current_epoch)
        return

    def forward(self, x, label):
        for t in tqdm(range(self.scheduler.n_timesteps - 1, -1, -1)):
            ts = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            predicted_noise = self.unet(x, ts, label)
            x = self.scheduler.denoise_step(x, t, predicted_noise)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        return optimizer


def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = BloodMNIST(split="train", download=True, transform=transform, size=64)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=7)
    num_classes = 8
    val_dataset = BloodMNIST(split="val", download=True, transform=transform, size=64)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=7)

    model = DiffusionModel(num_classes=num_classes)
    val_every_n_epochs = 1
    ckpt_save_dir = "models/diffusion_v2"
    logger = TensorBoardLogger("models/diffusion_v2", "tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        filename="diffusion_v2_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )
    time_limit = timedelta(hours=5)

    trainer = L.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_time=time_limit,
        default_root_dir="./models/diffusion_v2",
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path="models/diffusion_v2/final_model.ckpt")
    return model, trainer


if __name__ == "__main__":
    model, trainer = train()
    trainer.save_checkpoint("models/diffusion_v2/final_model2.ckpt")