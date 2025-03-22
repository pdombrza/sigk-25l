from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.io import read_image, ImageReadMode
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2
from skimage.restoration import richardson_lucy
from matplotlib import pyplot as plt

from src.models.train_unet import UNetModel


CHECKPOINT_PATH = "models/unet3/unet_epoch=84.ckpt"

def calculate_metrics(original_image, unet_image):
    original_image = original_image.astype(np.float32)
    unet_image = unet_image.astype(np.float32) / 255.0
    print(original_image.min(), original_image.max())
    print(unet_image.min(), unet_image.max())

    psnr_value = psnr(original_image, unet_image, data_range=1)
    ssim_value = ssim(original_image, unet_image, multichannel=True, data_range=1, channel_axis=2)
    sne_value = torch.sum((torch.tensor(unet_image).permute(2, 0, 1) - torch.tensor(original_image).permute(2, 0, 1)) ** 2)
    lpips_metric = LPIPS(net_type='vgg')
    original_tensor = torch.tensor(original_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)  # Convert to [B, C, H, W]
    unet_tensor = torch.tensor(unet_image * 2 - 1).permute(2, 0, 1).unsqueeze(0)  # Convert to [B, C, H, W]
    lpips_value = lpips_metric(original_tensor, unet_tensor).item()


    return psnr_value, ssim_value, lpips_value, sne_value


def gaussian_psf(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf


def gen_sk_deblur_img(image, blur_kernel, blur_sigma):
    blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX = blur_sigma)

    psf = gaussian_psf(size=blur_kernel, sigma=blur_sigma)

    deblurred_channels = []
    for c in range(image.shape[2]):
        deblurred_channel = richardson_lucy(blurred_image[:, :, c], psf, num_iter=30, clip=True)
        deblurred_channels.append(deblurred_channel)

    deblurred_image = np.stack(deblurred_channels, axis=-1)

    return deblurred_image


def get_original_image(image_path):
    image = np.array(Image.open(image_path)) / 255.0
    image = cv2.resize(image, (256, 256))
    return image


def get_blurry_image(image_path, blur_kernel, blur_sigma):
    image = np.array(Image.open(image_path)) / 255.0
    image = cv2.resize(image, (256, 256))
    blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX = blur_sigma)
    return blurred_image

def den_deblur_unet_image(image_path, blur_kernel, blur_sigma):
    model = UNetModel.load_from_checkpoint(CHECKPOINT_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Resize((256, 256)),
        v2.GaussianBlur((blur_kernel, blur_kernel), sigma=(blur_sigma, blur_sigma)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_image_tensor = transform(read_image(image_path, mode=ImageReadMode.RGB))

    with torch.no_grad():
        output = model(input_image_tensor.unsqueeze(0).to(device))

    output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    min_val, max_val = output.min(), output.max()
    new_image_normalized = 255 * (output - min_val) / (max_val - min_val)
    #new_image_normalized = output * 255
    new_image_normalized = new_image_normalized.astype('uint8')
    #sample_image = Image.fromarray(new_image_normalized)
    #sample_image.save("examples/out_restored.png")
    return new_image_normalized


def main():
    image_path = "data/DIV2K_train_LR_bicubic/X4/0785x4.png"
    plot_path = "examples/comparison5.png"
    blur_kernel = 3
    blur_sigma = 1.5

    original_image = get_original_image(image_path)
    blurry_image = get_blurry_image(image_path, blur_kernel, blur_sigma)
    sk_deblur_image = gen_sk_deblur_img(original_image, blur_kernel, blur_sigma)
    unet_deblur_image = den_deblur_unet_image(image_path, blur_kernel, blur_sigma)

    psnr_value, ssim_value, lpips_value, sne_value = calculate_metrics(original_image, unet_deblur_image)
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SSIM: {ssim_value:.4f}")
    print(f"LPIPS: {lpips_value:.4f}")
    print(f"SNE: {sne_value:.4f}")


    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title("Oryginalny obraz")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(blurry_image)
    plt.title(f"Rozmycie Gaussa {blur_kernel}x{blur_kernel}")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(sk_deblur_image)
    plt.title("Richardson-Lucy")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(unet_deblur_image)
    plt.title("UNet")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(plot_path)




if __name__ == "__main__":
    main()