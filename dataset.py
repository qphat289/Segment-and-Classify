# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SegmentationClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=(256, 256)):
        # Giữ nguyên phần khởi tạo
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.num_classes = 4
        
        # Get paths
        self.images_dir = os.path.join(root_dir, split, 'image')
        self.masks_dir = os.path.join(root_dir, split, 'mask')
        
        # Create class to index mapping
        self.class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(self.images_dir))):
            self.class_to_idx[class_name] = idx
            
        # Create list of (image_path, mask_path, class_label) tuples
        self.image_mask_pairs = []
        
        for class_name in os.listdir(self.images_dir):
            class_img_dir = os.path.join(self.images_dir, class_name)
            class_mask_dir = os.path.join(self.masks_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_img_dir):
                mask_name = os.path.splitext(img_name)[0] + '.png'
                img_path = os.path.join(class_img_dir, img_name)
                mask_path = os.path.join(class_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((img_path, mask_path, class_idx))

    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path, class_label = self.image_mask_pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('L')  # Đảm bảo ảnh đầu vào là grayscale
        mask = Image.open(mask_path).convert('L')  # Chuyển mask về grayscale
        
        # Resize images
        image = image.resize(self.img_size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.img_size, Image.Resampling.NEAREST)
        
        # Convert to numpy arrays
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Normalize mask values to 0-3 (cho 4 lớp)
        if mask_np.max() > 0:  # Đảm bảo mask không toàn đen
            # Nếu mask có giá trị lớn (ví dụ: 255), chuẩn hóa về 0-3
            if mask_np.max() > self.num_classes - 1:
                # Chia tỷ lệ giá trị pixel để nằm trong khoảng 0 đến 3
                mask_np = (mask_np / 255.0 * (self.num_classes - 1)).astype(np.int64)
            
            # Đảm bảo mask chỉ chứa các giá trị từ 0 đến 3
            mask_np = np.clip(mask_np, 0, self.num_classes - 1)
        
        # Convert to tensors
        image = torch.from_numpy(image_np).float() / 255.0
        image = image.unsqueeze(0)  # Thêm kênh (1, H, W)
        image = (image - 0.5) / 0.5  # Chuẩn hóa về [-1, 1]
        
        mask = torch.from_numpy(mask_np).long()  # Đảm bảo mask là tensor Long
        class_label = torch.tensor(class_label, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask, class_label


def get_data_loaders(root_dir, batch_size=8, num_workers=4):
    """Create data loaders for train and validation sets"""
    # Create datasets
    train_dataset = SegmentationClassificationDataset(
        root_dir=root_dir,
        split='train',
        img_size=(256, 256)
    )
    
    val_dataset = SegmentationClassificationDataset(
        root_dir=root_dir,
        split='val',
        img_size=(256, 256)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
