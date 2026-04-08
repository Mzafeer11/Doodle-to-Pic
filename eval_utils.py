import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def tensor_to_numpy(tensor):
    # Convert tensor (-1 to 1) to numpy (0 to 255)
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 0.5 + 0.5) * 255
    return img.astype(np.uint8)

def calculate_metrics(real_B, fake_B):
    """
    Args:
        real_B (Tensor): Batch of real images [B, C, H, W]
        fake_B (Tensor): Batch of fake images [B, C, H, W]
    """
    batch_ssim = []
    batch_psnr = []
    
    for i in range(real_B.size(0)):
        r = tensor_to_numpy(real_B[i])
        f = tensor_to_numpy(fake_B[i])
        
        # SSIM requires multichannel=True for RGB
        batch_ssim.append(ssim(r, f, multichannel=True, channel_axis=2, data_range=255))
        batch_psnr.append(psnr(r, f, data_range=255))
        
    return np.mean(batch_ssim), np.mean(batch_psnr)
