import logging
from os import listdir
from os.path import splitext
from pathlib import Path
from osgeo import gdal
from skimage.transform import resize
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, 
                 scale = 1.0, mask_suffix = ''):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, '[WARNING] Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        
        self.ids =[splitext(file)[0] for file in listdir(img_dir) if not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'[ERROR] No input file found in {img_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')\
            
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, '[WARNING] Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((new_w, new_h), 
                                 resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def preprocesstiff(cls, ds, scale, is_mask):
        count = ds.RasterCount
        band = ds.GetRasterBand(count)
        arr = band.ReadAsArray()
        w, h = arr.shape
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, '[WARNING] Scale is too small, resized images would have no pixel'
        img_ndarray = resize(arr, (new_w, new_h), preserve_range=True)
        
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
        
    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.tif','.tiff']:
            return gdal.Open(filename,gdal.GA_ReadOnly)
        else:
            return Image.open(filename)
    
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.img_dir.glob(name + '.*'))
        logging.info(mask_file)

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            '[WARNING] Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        # try:
        #     assert img.GetRasterBand(1).ReadAsArray().shape == mask.size, \
        #         '[WARNING] Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        #     img = self.preprocess(img, self.scale, is_mask=False)
        #     mask = self.preprocess(mask, self.scale, is_mask=True)
        # except Exception as e:
        #     assert img.GetRasterBand(1).ReadAsArray().shape == mask.GetRasterBand(1).ReadAsArray().shape, \
        #         '[WARNING] Image and mask {name} ' \
        #             + f'should be the same size, but are {img.GetRasterBand(1).ReadAsArray().shape} ' \
        #                 + f'and {mask.GetRasterBand(1).ReadAsArray().shape}'
        #     img = self.preprocesstiff(img, self.scale, is_mask=False)
        #     mask = self.preprocesstiff(mask, self.scale, is_mask=True)
        #     print(e)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        
class RemoteSensingDataset(BasicDataset):
    def __init__(self, img_dir, masks_dir, scale=1):
        super().__init__(img_dir, masks_dir, scale, mask_suffix='')