"""Data loading for the CVAT / LabelMe 3.0 export format.

Masks are NOT stored as single label images. Each image has an XML file that
maps object class names to the relative path of that class's binary PNG mask.
``assemble_mask_from_xml`` reads those binary masks and stacks them into a
multi-class ``(num_classes, H, W)`` array.

Ported from ``Pytorch-UNet/utils/data_loading.py``. The unused upstream
``BasicDataset`` / ``CarvanaDataset`` classes were dropped; the one piece of
``BasicDataset`` the inference path actually relied on — image preprocessing —
is now the standalone :func:`preprocess_image`.
"""

import logging
import os
from os import listdir
from os.path import splitext, isfile, join
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def get_xml_mask_dict(xml_file):
    """Return {class_name: relative_mask_path} for one LabelMe XML export."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_dict = {}
    for obj_elem in root.findall('.//object'):
        name = obj_elem.find('name').text
        mask = obj_elem.find('segm/mask').text
        xml_dict[name] = mask
    return xml_dict


def assemble_mask_from_xml(xml_file, mask_path):
    """Assemble a multi-class ``(num_classes, H, W)`` uint8 array by reading the
    per-class binary masks referenced in ``xml_file`` (paths relative to
    ``mask_path``). Each mask is grayscale-normalized and thresholded at 0.5.
    """
    xml_dict = get_xml_mask_dict(xml_file)
    mask_list = []
    i = 0
    for key in xml_dict:
        path = os.path.join(mask_path + xml_dict[key])
        img = Image.open(path).convert('L')
        normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # Convert the normalized image to a binary mask
        array = (normalized_img > 0.5).astype(np.uint8)
        mask_list.append(array)
        if i == 0:
            background_array = np.zeros_like(array)
            background_array = np.bitwise_or(background_array, array)
        else:
            background_array = np.bitwise_or(background_array, array)
        i += 1
    full_array = np.array(mask_list)
    return full_array


def preprocess_image(pil_img, scale):
    """Preprocess a PIL image into a ``(C, H, W)`` float array for the network.

    Reproduces exactly the ``is_mask=False`` branch of the original
    ``BasicDataset.preprocess`` (BICUBIC resize, CHW transpose, /255 scaling),
    which is what ``predict.py`` and the training eval loop relied on.
    """
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0

    return img


class LabelMeDataset(Dataset):
    """Dataset over CVAT / LabelMe 3.0 exports.

    Args:
        images_dir: folder of input ``.jpg`` images.
        mask_xml_dir: folder of per-image ``.xml`` files.
        mask_dir: root folder the XML mask paths are relative to.
        scale: image downscale factor in (0, 1].
    """

    def __init__(self, images_dir: str, mask_xml_dir: str, mask_dir: str,
                 scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_xml_dir = mask_xml_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir)
                    if isfile(join(images_dir, file)) and not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        xml_file = glob(os.path.join(self.mask_xml_dir, name + '.*'))
        img_file = glob(os.path.join(self.images_dir, name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(xml_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {xml_file}'

        # build array from multiple masks as specified in xml
        mask_array = assemble_mask_from_xml(xml_file[0], self.mask_dir)

        img = load_image(img_file[0])
        img = preprocess_image(img, self.scale)

        assert img[0].shape == mask_array[0].shape, \
            f'Image and mask {name} should be the same size, but are {img[0].shape} and {mask_array[0].shape}'

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_array.copy()).float().contiguous()
        }
