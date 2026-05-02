import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2

class MVTecDataset(Dataset):
    """
    Dataset class for MVTec AD dataset.
    Args:
        root_dir (str): Root directory of the dataset (e.g., 'mvtec_ad').
        category (str): Category of the object (e.g., 'bottle').
        split (str): 'train' or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, category, split='train', transform=None, resolution=256):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.resolution = resolution
        
        self.image_paths = []
        self.labels = [] # 0 for good, 1 for anomaly
        self.mask_paths = []

        if split == 'train':
            # Train split only contains 'good' images
            img_dir = os.path.join(root_dir, category, 'train', 'good')
            self.image_paths = sorted(glob(os.path.join(img_dir, "*.png")))
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)
        else:
            # Check for MVTec AD 2 structure first
            test_public_dir = os.path.join(root_dir, category, 'test_public')
            if os.path.exists(test_public_dir):
                # AD 2 structure: test_public/good, test_public/bad
                for split_name, label in [('good', 0), ('bad', 1)]:
                    img_subdir = os.path.join(test_public_dir, split_name)
                    if os.path.exists(img_subdir):
                        imgs = sorted(glob(os.path.join(img_subdir, "*.png")))
                        self.image_paths.extend(imgs)
                        self.labels.extend([label] * len(imgs))
                        
                        if label == 1:
                            gt_dir = os.path.join(test_public_dir, 'ground_truth', 'bad')
                            for img_path in imgs:
                                base_name = os.path.basename(img_path).replace(".png", "_mask.png")
                                mask_path = os.path.join(gt_dir, base_name)
                                self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)
                        else:
                            self.mask_paths.extend([None] * len(imgs))
            else:
                # Original MVTec AD structure
                test_dir = os.path.join(root_dir, category, 'test')
                gt_dir = os.path.join(root_dir, category, 'ground_truth')
                
                defect_types = sorted(os.listdir(test_dir))
                for defect in defect_types:
                    img_subdir = os.path.join(test_dir, defect)
                    imgs = sorted(glob(os.path.join(img_subdir, "*.png")))
                    self.image_paths.extend(imgs)
                    
                    if defect == 'good':
                        self.labels.extend([0] * len(imgs))
                        self.mask_paths.extend([None] * len(imgs))
                    else:
                        self.labels.extend([1] * len(imgs))
                        mask_subdir = os.path.join(gt_dir, defect)
                        for img_path in imgs:
                            base_name = os.path.basename(img_path).replace(".png", "_mask.png")
                            mask_path = os.path.join(mask_subdir, base_name)
                            self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            # Use the specified resolution for masks
            mask_transform = transforms.Compose([
                PadToSquare(),
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
        else:
            # Return zero mask if none exists (all zeros for 'good')
            mask = torch.zeros((1, self.resolution, self.resolution))

        return image, label, mask, os.path.basename(img_path)

# Custom transform to pad to square before resizing
class PadToSquare(object):
    def __call__(self, img):
        w, h = img.size
        if w == h: return img
        max_size = max(w, h)
        padding_w = (max_size - w) // 2
        padding_h = (max_size - h) // 2
        # (left, top, right, bottom)
        padding = (padding_w, padding_h, max_size - w - padding_w, max_size - h - padding_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

def get_dataloader(root_dir, category, split='train', batch_size=16, shuffle=True, resolution=256, num_workers=4, pin_memory=None, get_dataset=False):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MVTecDataset(root_dir, category, split, transform, resolution=resolution)
    
    if get_dataset:
        return dataset
    
    # Use persistent_workers if num_workers > 0 to keep the same processes alive across epochs
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0) if split == 'train' else False
    )
    return dataloader
