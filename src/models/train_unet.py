from datetime import timedelta

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint

from src.models.unet import UNet
from src.data.dataset import DeblurringDataset


class UNetModel(L.LightningModule):
    def __init__(self, model: UNet | None = None, lambda_lpips: float | None = None, lambda_l1: float | None = None):
        super(UNetModel, self).__init__()
        self.unet = UNet(in_channels=3) if model is None else model
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LPIPS(net_type='vgg')
        self.lambda_lpips = lambda_lpips if lambda_lpips is not None else 1.0
        self.lambda_l1 = lambda_l1 if lambda_l1 is not None else 1.0

    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        prediction = self.unet(input)
        prediction = self._normalize_output(prediction, min_val=torch.min(prediction), max_val=torch.max(prediction))
        l1_loss = self.l1_loss(prediction, target) * self.lambda_l1
        perceptual_loss = self.perceptual_loss(prediction, target) * self.lambda_lpips
        loss = l1_loss + perceptual_loss
        self.log_dict({
            'l1_loss': l1_loss,
            'perceptual_loss': perceptual_loss,
            'loss': loss,
        })

    def validation_step(self, batch, batch_idx):
        if not self.logger:
            return

        self.unet.eval()
        with torch.no_grad():
            input_img = batch['input']
            target_img = batch['target']
            deblurred_img = self.unet(input_img)
        self.unet.train()
        psnr = PSNR().to(self.device)
        ssim = SSIM().to(self.device)
        self.log_dict({
            'ssim': ssim(deblurred_img, target_img),
            'psnr': psnr(deblurred_img, target_img),
        })
        self.logger.experiment.add_images('input', input_img, self.current_epoch)
        self.logger.experiment.add_images('target', target_img[:4], self.current_epoch)
        self.logger.experiment.add_images('deblurred', deblurred_img[:4], self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.unet(x)

    def _normalize_output(self, output: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        output = (output - min_val) / (max_val - min_val)
        return output * 2.0 - 1.0



def train():
    dataset = DeblurringDataset(images_path='data/DIV2K_train_LR_bicubic/X4')
    n_valid_images = 16
    train_size = len(dataset) - n_valid_images
    train_set, valid_set = random_split(dataset, [train_size, n_valid_images])
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False)
    model = UNetModel(lambda_lpips=1.0, lambda_l1=1.0)
    val_every_n_epochs = 1
    ckpt_save_dir = "models/unet/"
    logger = TensorBoardLogger("models/unet/tb_logs", "unet")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        filename="unet_{epoch:02d}",
        every_n_epochs=val_every_n_epochs,
        save_top_k=-1,
    )
    exception_callback = OnExceptionCheckpoint(
        dirpath=ckpt_save_dir,
        filename="unet_{epoch}-{step}_ex",
    )
    time_limit = timedelta(hours=1)

    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_time=time_limit,
    )
    trainer.fit(model, train_loader, valid_loader)
    return model, trainer