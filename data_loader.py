import cv2
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import random
import PIL.Image as Image


class dataset(torch.utils.data.Dataset):
    def __init__(self, ics, icsmask, transformer, only_icbubbles=False, use_num_data=-1, force_positive_loop=1):
        super(dataset, self).__init__()

        self.ics_icsmask = self.combin_all(ics, icsmask[0], icsmask[1], only_icbubbles)
        self.transformer = transformer
        self.use_num_data = use_num_data
        self.len = len(self.ics_icsmask) if self.use_num_data < 0 else self.use_num_data
        self.force_positive_loop = force_positive_loop
        self.random_indx = np.arange(self.len)
        np.random.shuffle(self.random_indx)

    
    def combin_all(selt, ics, icsMask, bubbleMask, only_icbubbles=False):
        ics = np.array(ics)[...,None]
        icsMask = np.array(icsMask)[...,None]
        bubbleMask = np.array(bubbleMask)[...,None]
        if only_icbubbles:
            bubbleMask = (bubbleMask.astype(float)*icsMask.astype(float)/255).astype('uint8')


        ics_icsmask = np.concatenate((ics, icsMask), axis=-1)
        ics_icsmask = np.concatenate((ics_icsmask, bubbleMask), axis=-1)
        return ics_icsmask


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = None
        y = None
        ic_mask = self.ics_icsmask[self.random_indx[idx]]
        if self.transformer is not None:
            for i in range(self.force_positive_loop):
                ic_mask_temp = self.transformer(ic_mask)
                x = ic_mask_temp[0][None,...]
                y = ic_mask_temp[1:]
                if torch.any(y):
                    break

        return (x, y)




class x_ray_image_dataLoader():
    def __init__(self, path, batch_size, shape=(256,256), crop_size=(256, 256), shuffle=True, only_icbubbles=False, use_num_data=-1, force_positive_loop=1):
        ICMaskPaths = glob.glob(path+'/*_ICMask.jpg')
        bubbleMaskPaths = glob.glob(path+'/*_BubbleMask.jpg')
        imagePaths = self.get_imgPaths_from_maskPaths(ICMaskPaths)
        self.ims = self.read_ims(imagePaths)
        self.ICMask = self.read_masks(ICMaskPaths)
        self.bubbleMask = self.read_masks(bubbleMaskPaths)
        transformer = transforms.Compose([transforms.ToPILImage(),
                                        transforms.RandomCrop(size=crop_size),
                                        transforms.RandomResizedCrop(shape, scale=(0.1111111111, 1.),ratio=(1,1) ,interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.RandomApply(torch.nn.ModuleList([Rotate90()])),                                        # transforms.Resize(shape),
                                        Random_contrast_image_only((0.9,1.1)),
                                        transforms.ToTensor(),
                                        ])
        self.dataset = dataset(self.ims, [self.ICMask, self.bubbleMask], transformer, only_icbubbles, use_num_data, force_positive_loop)

        self.loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size)

    def read_ims(self, im_paths):
        ims = []
        for impath in im_paths:
            im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
        return ims

    def read_masks(self, mask_paths):
        masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        return masks

    def get_imgPaths_from_maskPaths(self, maskPath):
        imagePaths = []
        for path in maskPath:
            img_path = path.replace('_ICMask.jpg', '.jpg')
            imagePaths.append(img_path)
        return imagePaths


class x_ray_image_dataLoader_testing_set():
    def __init__(self, path, batch_size, shape=(1536 ,1952), shuffle=True, only_icbubbles=False):
        ICMaskPaths = glob.glob(path+'/*_ICMask.jpg')
        bubbleMaskPaths = glob.glob(path+'/*_BubbleMask.jpg')
        imagePaths = self.get_imgPaths_from_maskPaths(ICMaskPaths)
        print(imagePaths)
        self.ims = self.read_ims(imagePaths)
        self.ICMask = self.read_masks(ICMaskPaths)
        self.bubbleMask = self.read_masks(bubbleMaskPaths)
        transformer = transforms.Compose([  transforms.ToPILImage(),
                                            # transforms.RandomCrop(size=shape, padding=8, padding_mode='edge'),
                                            transforms.CenterCrop(size=shape),
                                            # transforms.RandomApply(torch.nn.ModuleList([Rotate90()])), 
                                            transforms.ToTensor()])
        self.dataset = dataset(self.ims, [self.ICMask, self.bubbleMask], transformer, only_icbubbles)

        self.loader = DataLoader(self.dataset, shuffle=shuffle, batch_size=batch_size)

    def read_ims(self, im_paths):
        ims = []
        for impath in im_paths:
            im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
            ims.append(im)
        return ims

    def read_masks(self, mask_paths):
        masks = []
        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)
        return masks

    def get_imgPaths_from_maskPaths(self, maskPath):
        imagePaths = []
        for path in maskPath:
            img_path = path.replace('_ICMask.jpg', '.jpg')
            imagePaths.append(img_path)
        return imagePaths
    

class Rotate90(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.angles = 90

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angles)

class Random_contrast_image_only(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def __call__(self, x):
        x = np.array(x)
        img = x[:,:,0]
        temp = random.uniform(self.ratio[0], self.ratio[1])
        img = Image.fromarray(img)
        contrast_img = transforms.functional.adjust_contrast(img, temp)
        x[:,:,0] = np.array(contrast_img)
        return Image.fromarray(x)

class refine_label(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self,x):
        x = np.array(x)
        label = x[:,:,1:]
        label = (label>int(255/2)).astype('uint8')*255
        x[:,:,1:] = label
        return Image.fromarray(x)


def read_ims(im_paths):
    ims = []
    for impath in im_paths:
        im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        ims.append(im)
    return ims