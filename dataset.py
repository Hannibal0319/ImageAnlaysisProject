import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    """
    Dataset class for MVTec AD dataset.
    Args:
        root_dir (str): Root directory of the dataset (e.g., 'mvtec_ad').
        category (str): Category of the object (e.g., 'bottle').
        split (str): 'train' or 'test'.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, root_dir, category, split='train', transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        
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
            # Test split contains both 'good' and various defect types
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
                    # Corresponding ground truth masks
                    mask_subdir = os.path.join(gt_dir, defect)
                    for img_path in imgs:
                        # Mask filenames usually append '_mask' before .png
                        base_name = os.path.basename(img_path).replace(".png", "_mask.png")
                        mask_path = os.path.join(mask_subdir, base_name)
                        if os.path.exists(mask_path):
                            self.mask_paths.append(mask_path)
                        else:
                            # In some cases, masks might be missing or named differently
                            self.mask_paths.append(None)

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
            # Use a slightly different transform for masks (Resize + ToTensor, no normalization)
            mask_transform = transforms.Compose([
                transforms.Resize((256, 256)), # Hardcoded for now to match default
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
        else:
            # Return zero mask if none exists (all zeros for 'good')
            mask = torch.zeros((1, 256, 256))

        return image, label, mask, os.path.basename(img_path)

def get_dataloader(root_dir, category, split='train', batch_size=16, shuffle=True, resolution=256, num_workers=4, pin_memory=True):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MVTecDataset(root_dir, category, split, transform)
    
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
