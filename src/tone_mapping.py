import os
from datetime import timedelta
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.vgg import vgg19
from torch.utils.data import DataLoader, Dataset
import torchmetrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import lpips
import cv2
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from numpy import ndarray
from scipy.stats import norm

EPSILON = 0.0000000001

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

def read_exr(im_path: str) -> ndarray:
    return cv2.imread(filename=im_path, flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

def generate_exposures(hdr_img, target_val=1.2):
    max_val = np.max(hdr_img)
    median_val = np.median(hdr_img)
    
    c_start = np.log(target_val / max_val) / np.log(2.0)
    c_end   = np.log(target_val / median_val) / np.log(2.0)
    
    exp_values = [c_start, (c_start + c_end) / 2.0, c_end]
    
    exposures = []
    for exp in exp_values:
        scaling_factor = np.sqrt(2.0) ** exp
        scaled_img = hdr_img * scaling_factor
        exposure_img = np.clip(scaled_img, 0, 1)
        exposures.append(exposure_img)
    
    return exposures

def tone_map_mantiuk(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.2)
    result = tonemap_operator.process(src=image)
    return result

class VGG19Extractor(nn.Module):
    _LAYER_MAP = {
        0: "conv1_1", 2: "conv1_2",
        5: 'conv2_1', 7: 'conv2_2',
        10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3', 16: 'conv3_4',
        19: 'conv4_1', 21: 'conv4_2', 23: 'conv4_3', 25: 'conv4_4',
        28: 'conv5_1', 30: 'conv5_2', 32: 'conv5_3', 34: 'conv5_4',
    }

    def __init__(self, chosen_layers):
        super(VGG19Extractor, self).__init__()

        self.chosen_layers = chosen_layers
        self.vgg19_layers = nn.Sequential(*list(vgg19(weights='DEFAULT').features.children()))

        for param in self.vgg19_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.vgg19_layers):
            x = layer(x)
            name = self._LAYER_MAP.get(idx)

            if name in self.chosen_layers:
                features.append(x)

        return features


class NoisyDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.transform = transform

        self.images_low = []
        self.images_mid = []
        self.images_high = []
        self.images_ulaw = []
        self.images_original = []
        
        for fname in os.listdir(root_folder):
            image_src = cv2.cvtColor(read_exr(root_folder + '/' + fname), cv2.COLOR_BGR2RGB)
            image_hdr = 0.5 * image_src / np.mean(image_src)
            i_hdr = np.median(image_hdr)
            u = 8.759 * i_hdr**2.148 + 0.1494 * i_hdr**(-2.067)

            # plt.imshow(image_hdr)
            # plt.show()
            images = generate_exposures(image_hdr)
            self.images_low.append(images[0])
            # plt.imshow(self.images_low[-1])
            # plt.show()
            self.images_mid.append(images[1])
            # plt.imshow(self.images_mid[-1])
            # plt.show()
            self.images_high.append(images[2])
            # plt.imshow(self.images_high[-1])
            # plt.show()
            self.images_ulaw.append(np.log10(1.0 + u * image_hdr) / np.log10(1.0 + u))
            self.images_original.append(image_hdr)


    def __len__(self):
        return len(self.images_low)

    def __getitem__(self, idx):
        img_low = self.images_low[idx]
        img_mid = self.images_mid[idx]
        img_high = self.images_high[idx]
        img_ulaw = self.images_ulaw[idx]
        img_original = self.images_original[idx]
        if self.transform:
            img_low = self.transform(img_low)
            img_mid = self.transform(img_mid)
            img_high = self.transform(img_high)
            img_ulaw = self.transform(img_ulaw)
            img_original = self.transform(img_original)

        return img_low, img_mid, img_high, img_ulaw, img_original


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.channels = 16
        self.encoder_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channels, out_channels=2 * self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=2 * self.channels, out_channels=4 * self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder_block(x)

    
class Fusion(nn.Module):
    def __init__(self, in_channels=64):
        super(Fusion, self).__init__()
        self.channels = 3 * in_channels
        self.fusion_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fusion_block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=192):
        super(Decoder, self).__init__()
        self.channels = 32
        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.channels // 2, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        self.output_layer = nn.Sigmoid()

    def forward(self, fusion, img_low, img_mid, img_high):
        decoder_out = self.decoder_block(fusion)
        return self.output_layer(decoder_out + img_low + img_mid + img_high)


class ToneMappingModel(nn.Module): 
    def __init__(self):
        super(ToneMappingModel, self).__init__()
        self.encoder = Encoder()
        self.fusion = Fusion()
        self.decoder = Decoder()
        
    def forward(self, img_low, img_mid, img_high):
        low_encoded = self.encoder(img_low)
        mid_encoded = self.encoder(img_mid)
        high_encoded = self.encoder(img_high)
        fusion = self.fusion(torch.cat([low_encoded, mid_encoded, high_encoded], dim=1))
        decoded = self.decoder(fusion, img_low, img_mid, img_high)
        return decoded

class TMAP(L.LightningModule):
    def __init__(self, model=None):
        super(TMAP, self).__init__()
        self.model = ToneMappingModel() if model is None else model
        self.l1_loss = nn.L1Loss()
        self.feature_layers = ("conv1_1", "conv2_1", "conv3_1","conv4_1", "conv4_3", "conv5_3")
        self.vgg = VGG19Extractor(self.feature_layers)
        self.automatic_optimization = False
        self.i = 0

    def normalize_gaussian(self, feature_image, kernel_size, sigma):
        interval = (2 * sigma + 1.) / kernel_size
        space = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
        kern1d = np.diff(norm.cdf(space))

        kernel = np.sqrt(np.outer(kern1d, kern1d))
        kernel /= kernel.sum()
        kernel = kernel.astype(np.float32)
        kernel_tensor = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).repeat(feature_image.shape[1], 1, 1, 1).detach().to(self.device)
    
        blurred_image = torch.nn.functional.conv2d(feature_image, kernel_tensor, padding='same', groups=feature_image.shape[1]).detach()
        
        return (feature_image - blurred_image) / (torch.abs(blurred_image) + EPSILON)
    
    def calculate_local_mean(self, input_img, kernel_size):
        mean_filter = torch.ones(input_img.shape[1], 1, kernel_size, kernel_size).to(self.device)
        padding = kernel_size // 2
        # img_ms = torch.sign(input_img) * torch.abs(input_img)**(0.5 if u_law_processing else 1.0)
        mean = torch.nn.functional.conv2d(input_img, mean_filter / kernel_size ** 2, padding=padding, groups=input_img.shape[1]).detach()
        std = (torch.nn.functional.conv2d(input_img ** 2, mean_filter/ kernel_size ** 2, padding=padding, groups=input_img.shape[1]) - mean ** 2).detach()
        img_mn = torch.sqrt(torch.abs(std)) / (torch.abs(mean) + EPSILON)
        return img_mn

    def calculate_feature_contrast_masking(self, gauss_out, mean_out, u_law_processing):
        img_ms = torch.sign(gauss_out) * torch.abs(gauss_out)**(0.5 if u_law_processing else 1.0)
        return img_ms / (1.0 + mean_out)
    
    def compute_luminance(self, img):
        r, g, b = img[0], img[1], img[2]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b  # Standard luminance formula
        return lum
    
    def restore_color_from_luminance(self, gt_img, pred_lum, a=0.6):
        r, g, b = gt_img[0], gt_img[1], gt_img[2]
        y_gt_lum = self.compute_luminance(gt_img) + EPSILON

        restored = torch.zeros_like(gt_img)
        restored[0] = (r / y_gt_lum)**a * pred_lum
        restored[1] = (g / y_gt_lum)**a * pred_lum
        restored[2] = (b / y_gt_lum)**a * pred_lum
        return restored

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        img_low, img_mid, img_high, img_ulaw, img_original = batch
        prediction = self.model(img_low, img_mid, img_high)
        # print(torch.min(prediction), torch.max(prediction))
        # tensor_cpu = self._normalize_output(prediction[0], torch.min(prediction[0]), torch.max(prediction[0])).detach().cpu().permute(1, 2, 0)
        # with torch.no_grad():
        #     if self.i % 10 == 0:
        #             plt.imshow(tensor_cpu)
        #             plt.axis('off')
        #             plt.show()
        #     self.i += 1
        loss = 0
        for feature_prediction, feature_ulaw in zip(self.vgg(prediction), self.vgg(img_ulaw)):
            prediction_gauss = self.normalize_gaussian(feature_prediction, kernel_size=13, sigma=2)
            prediction_means = self.calculate_local_mean(feature_prediction, kernel_size=13)
            feature_prediction_mask = self.calculate_feature_contrast_masking(prediction_gauss, prediction_means, u_law_processing=False)
            ulaw_gauss = self.normalize_gaussian(feature_ulaw, kernel_size=13, sigma=2)
            ulaw_means = self.calculate_local_mean(feature_ulaw, kernel_size=13)
            feature_ulaw_mask = self.calculate_feature_contrast_masking(ulaw_gauss, ulaw_means, u_law_processing=True)
            loss += self.l1_loss(feature_prediction_mask, feature_ulaw_mask) / len(self.feature_layers)
        self.manual_backward(loss)
        optimizer.step()

        with torch.no_grad():
            pred = prediction[0]
            gt = img_original[0]

            pred_lum = self.compute_luminance(pred)
            restored_img = self.restore_color_from_luminance(gt, pred_lum)
            if self.i % 10 == 0:
                img = restored_img / 100.0 #torch.max(restored_img)
                print(restored_img.max(), restored_img.min())
                # img_min = img.min()
                # img_max = img.max()
                # img = (img - img_min) / (img_max - img_min + EPSILON)
                plt.imshow(img.cpu().permute(1, 2, 0).numpy())
                plt.show()
            self.i += 1

        self.log_dict({
            'l1_loss': loss,
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return {"optimizer": optimizer}

    def forward(self, x):
        return self.model(x)

    def _normalize_output(self, output: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        output = (output - min_val) / (max_val - min_val)
        return output


def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = NoisyDataset(root_folder="./data/output_dir", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TMAP()
    logger = TensorBoardLogger("models/tmap/tb_logs", "tmap")
    ckpt_save_dir = "models/tmap/"

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        filename="tmap_{epoch:02d}",
        every_n_epochs=5,
        save_top_k=-1,
    )
    # exception_callback = OnExceptionCheckpoint(
    #     dirpath=ckpt_save_dir,
    #     filename="tmap_{epoch}-{step}_ex",
    # )
    time_limit = timedelta(hours=1)


    trainer = L.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        max_time=time_limit,
        log_every_n_steps=1
    )
    trainer.fit(model, dataloader)
    return model, trainer

if __name__ == "__main__":
    train()
