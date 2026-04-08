import os
import torch
from torch.utils_data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairedDataset(Dataset):
    def __init__(self, root_dir, mode='separate', transform=None, direction='AtoB'):
        """
        Args:
            root_dir (string): Directory with images.
            mode (string): 'separate' for CUHK-style (A and B folders) 
                          or 'combined' for Anime-style (side-by-side).
            direction: 'AtoB' (e.g., Sketch to Photo) or 'BtoA'.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.direction = direction

        if mode == 'separate':
            # CUHK Face Sketch style
            self.dir_A = os.path.join(root_dir, 'sketches')
            self.dir_B = os.path.join(root_dir, 'photos')
            self.img_names = sorted(os.listdir(self.dir_A))
        else:
            # Combined / Side-by-side style
            self.img_names = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        if self.mode == 'separate':
            path_A = os.path.join(self.dir_A, img_name)
            path_B = os.path.join(self.dir_B, img_name)
            img_A = Image.open(path_A).convert('RGB')
            img_B = Image.open(path_B).convert('RGB')
        else:
            img_path = os.path.join(self.root_dir, img_name)
            combined_img = Image.open(img_path).convert('RGB')
            w, h = combined_img.size
            img_A = combined_img.crop((0, 0, w // 2, h))
            img_B = combined_img.crop((w // 2, 0, w, h))

        if self.direction == 'BtoA':
            img_A, img_B = img_B, img_A

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

def get_transforms(img_size=256):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
