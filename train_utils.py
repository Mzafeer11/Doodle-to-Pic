import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class Pix2PixTrainer:
    def __init__(self, generator, discriminator, gen_optimizer, disc_optimizer, device, lambda_l1=100.0):
        self.netG = generator
        self.netD = discriminator
        self.optimizer_G = gen_optimizer
        self.optimizer_D = disc_optimizer
        self.device = device
        self.lambda_l1 = lambda_l1
        
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        self.scaler = GradScaler()

    def train_step(self, real_A, real_B):
        real_A, real_B = real_A.to(self.device), real_B.to(self.device)
        
        # --- Update Discriminator ---
        self.optimizer_D.zero_grad()
        with autocast():
            # Fake
            fake_B = self.netG(real_A)
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = self.netD(fake_AB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            
            # Real
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = self.netD(real_AB)
            loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
            
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            
        self.scaler.scale(loss_D).backward()
        self.scaler.step(self.optimizer_D)
        
        # --- Update Generator ---
        self.optimizer_G.zero_grad()
        with autocast():
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = self.netD(fake_AB)
            
            loss_G_GAN = self.criterionGAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = self.criterionL1(fake_B, real_B) * self.lambda_l1
            
            loss_G = loss_G_GAN + loss_G_L1
            
        self.scaler.scale(loss_G).backward()
        self.scaler.step(self.optimizer_G)
        
        self.scaler.update()
        
        return {
            'loss_G': loss_G.item(),
            'loss_G_GAN': loss_G_GAN.item(),
            'loss_G_L1': loss_G_L1.item(),
            'loss_D': loss_D.item()
        }

def save_checkpoint(state, filename="pix2pix_checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_G, optimizer_D):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['state_dict_G'])
    discriminator.load_state_dict(checkpoint['state_dict_D'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    return checkpoint['epoch']
