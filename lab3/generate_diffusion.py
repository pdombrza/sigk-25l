import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from diffusion import DiffusionModel


def main():
    model = DiffusionModel.load_from_checkpoint("models/diffusion_v2/final_model2.ckpt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    channels = 3
    height, width = 64, 64
    gen_images = {}
    all_images = []
    all_labels = []
    for cl in range(2):
        print(f"Generating images for class {cl}")
        class_dir = f"lab3/examples/{cl}"
        os.makedirs(class_dir, exist_ok=True)
        for i in range(1):
            noise = torch.randn(1, channels, height, width, device=device)
            labels = torch.ones(1, device=device, dtype=torch.long) * cl
            with torch.no_grad():
                generated_images = model(noise, labels)
            generated_images = generated_images.squeeze(0)
            all_images.append(generated_images)
            all_labels.append(cl)
            # save_image(generated_images, class_dir, i)
    save_grid(all_images, all_labels, "lab3/generated_grid.png")




def save_image(img, output_dir, img_idx):
    img = (img + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype('uint8')
    img_path = os.path.join(output_dir, f"image_{img_idx}.png")
    Image.fromarray(img).save(img_path)


def save_grid(gen_images, labels, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 2))
    for i in range(2):
        axes[i].imshow(np.transpose((gen_images[i].cpu().numpy() + 1) / 2, (1, 2, 0)))
        axes[i].set_title(f"Pred: {labels[i]}")
        axes[i].axis('off')
    plt.suptitle("Diffusion generated images")
    plt.tight_layout()
    plt.savefig("lab3/generated_grid.png")


if __name__ == "__main__":
    main()