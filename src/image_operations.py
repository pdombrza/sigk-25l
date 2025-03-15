from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def resize_stretch(image, target_size=(256, 256)):
    """Resize image directly to target_size (may cause distortion)."""
    return image.resize(target_size, Image.LANCZOS)

def resize_and_center_crop(image, target_size=(256, 256)):
    """Resize and crop an image to fit exactly target_size while keeping aspect ratio."""
    aspect = image.width / image.height
    if aspect > 1:
        new_width = int(aspect * target_size[1])
        new_height = target_size[1]
    else:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect)

    image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    left = (image_resized.width - target_size[0]) / 2
    top = (image_resized.height - target_size[1]) / 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    return image_resized.crop((left, top, right, bottom))

def resize_with_padding(image, target_size=(256, 256), fill_color=(0, 0, 0)):
    """Resize an image to fit within target_size while keeping aspect ratio, adding padding if necessary."""
    image.thumbnail(target_size, Image.LANCZOS)  # Resize while maintaining aspect ratio
    delta_w = target_size[0] - image.size[0]
    delta_h = target_size[1] - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(image, padding, fill_color)

image = Image.open("0027x4m.png")
resized_image = resize_with_padding(image)
plt.imshow(resized_image)
plt.axis("off")  # Hide axes
plt.show()