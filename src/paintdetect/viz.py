"""Visualization helpers for predicted vs. true masks.

Consolidates ``plot_img_and_mask`` (from ``Pytorch-UNet/utils/utils.py``) and
``overlay_masks`` (from the repo-root ``utils.py``).
"""

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()


def overlay_masks(mask1, mask2, color1, color2):
    """Overlay two binary PIL masks onto a black canvas in the given colors.

    Typically used with color1=true-mask, color2=predicted-mask so overlap shows
    as the additive blend.
    """
    # Ensure both masks have the same size
    if mask1.size != mask2.size:
        raise ValueError("Masks must have the same size")

    # Create a new blank image
    width, height = mask1.size
    result = Image.new('RGB', (width, height), (0, 0, 0))

    # Convert binary masks to RGB format
    mask1_rgb = mask1.convert('RGB')
    mask2_rgb = mask2.convert('RGB')

    # Create a drawing object for the resulting image
    draw = ImageDraw.Draw(result)

    # Overlay the masks with assigned colors
    for x in range(width):
        for y in range(height):
            pixel1 = mask1_rgb.getpixel((x, y))
            pixel2 = mask2_rgb.getpixel((x, y))
            if pixel1 == (255, 255, 255):
                draw.point((x, y), color1)
            if pixel2 == (255, 255, 255):
                draw.point((x, y), color2)

    return result
