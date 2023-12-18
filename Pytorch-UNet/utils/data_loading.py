import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os
from glob import glob 
from matplotlib import pyplot as plt 

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def get_xml_mask_dict(xml_file):
    '''returns a dict of the paths to individual classwise masks for each class 
       of a multiclass array as contained in the xml file of the LabelMe format export'''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    xml_dict = {}
    for obj_elem in root.findall('.//object'):
    # Extract information about the object
        name = obj_elem.find('name').text
        # Extract information from the 'segm' element
        mask = obj_elem.find('segm/mask').text
        xml_dict[name]=mask
    return xml_dict

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')

class LabelMeDataset(Dataset):
    def __init__(self, images_dir: str, mask_xml_dir: str, mask_dir:str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_xml_dir = mask_xml_dir
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))

        # self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        # self.mask
        # logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess( pil_img, scale, is_mask,mask_values=8):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask: #generate class array? 
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
    
    def assemble_mask_from_xml(self,xml_file,mask_path):
        '''assembles a multiclass numpy array by reading multiple binary masks
        stored in the mask paths folder using filnames contained in the xml file'''
        xml_dict = get_xml_mask_dict(xml_file)
        mask_list = []
        i = 0 
        for key in xml_dict:
            path = os.path.join(mask_path+xml_dict[key])
            img = Image.open(path).convert('L')
            normalized_img = (img - np.min(img)) / (np.max(img) - np.min(img))
            # Convert the normalized image to a binary mask
            array = (normalized_img > 0.5).astype(np.uint8)
           # array =  np.array(array).transpose((1, 0,))
            mask_list.append(array)
            if i == 0:
                background_array = np.zeros_like(array)
                background_array = np.bitwise_or(background_array,array)
            else:
                background_array = np.bitwise_or(background_array,array)
            i+= 1 
        #mask_list.append(background_array)
        full_array = np.array(mask_list)
        return full_array


    def __getitem__(self, idx):
        name = self.ids[idx]
        path = os.path.join(self.mask_xml_dir,name + '.*')
        xml_file =  glob(path)
        img_file = glob(os.path.join(self.images_dir,name + '.*'))

        #mask_files = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(xml_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {xml_file}'
        
        #build array from multiple masks as specified in xml 
        mask_array = self.assemble_mask_from_xml(xml_file[0],self.mask_dir)      

        #mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        img = self.preprocess(img, self.scale, is_mask=False)
     
        assert img[0].shape == mask_array[0].shape, \
            f'Image and mask {name} should be the same size, but are {img[0].shape} and {mask_array[0].shape}'

        #mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_array.copy()).float().contiguous()
        }
