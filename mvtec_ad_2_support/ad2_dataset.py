import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys

# Add parent directory to path to reuse PadToSquare
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dataset import PadToSquare
except ImportError:
    # Redefine if not found
    class PadToSquare(object):
        def __call__(self, img):
            w, h = img.size
            if w == h: return img
            max_size = max(w, h)
            padding_w = (max_size - w) // 2
            padding_h = (max_size - h) // 2
            padding = (padding_w, padding_h, max_size - w - padding_w, max_size - h - padding_h)
            return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

class MVTecAD2Dataset(Dataset):
    """
    Dedicated dataset class for MVTec AD 2.
    Handles the Public/Private split and grouped defects.
    """
    def __init__(self, root_dir, category, split='train', transform=None, resolution=256):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.resolution = resolution
        
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        
        if split == 'train':
            img_dir = os.path.join(root_dir, category, 'train', 'good')
            self.image_paths = sorted(glob(os.path.join(img_dir, "*.png")))
            self.labels = [0] * len(self.image_paths)
            self.mask_paths = [None] * len(self.image_paths)
        else:
            # We default to 'test_public' for evaluation
            test_dir = os.path.join(root_dir, category, 'test_public')
            
            # Good samples
            good_dir = os.path.join(test_dir, 'good')
            if os.path.exists(good_dir):
                imgs = sorted(glob(os.path.join(good_dir, "*.png")))
                self.image_paths.extend(imgs)
                self.labels.extend([0] * len(imgs))
                self.mask_paths.extend([None] * len(imgs))
            
            # Bad samples (grouped in AD 2)
            bad_dir = os.path.join(test_dir, 'bad')
            if os.path.exists(bad_dir):
                imgs = sorted(glob(os.path.join(bad_dir, "*.png")))
                self.image_paths.extend(imgs)
                self.labels.extend([1] * len(imgs))
                
                gt_dir = os.path.join(test_dir, 'ground_truth', 'bad')
                for img_path in imgs:
                    mask_name = os.path.basename(img_path).replace(".png", "_mask.png")
                    mask_path = os.path.join(gt_dir, mask_name)
                    self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if mask_path:
            mask = Image.open(mask_path).convert('L')
            mask = transforms.Resize((self.resolution, self.resolution))(mask)
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.zeros((1, self.resolution, self.resolution))
            
        return image, label, mask, os.path.basename(self.image_paths[idx])

def get_ad2_dataloader(root_dir, category, split='train', batch_size=16, resolution=256):
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MVTecAD2Dataset(root_dir, category, split, transform, resolution)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
