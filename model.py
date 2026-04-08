import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, submodule=None, innermost=False, outermost=False, dropout=False):
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        
        downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(out_channels)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(in_channels)

        if outermost:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout:
                up += [nn.Dropout(0.5)]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64):
        super(UNetGenerator, self).__init__()
        # Build U-Net from the inside out
        unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, dropout=True)
        unet_block = UNetBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UNetBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UNetBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UNetBlock(input_nc, ngf, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=6, ndf=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        model = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)] # Output 1 channel prediction map
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def init_weights(net, init_gain=0.02):
    def init_func(m):
        if hasattr(m, 'weight') and (m.__class__.__name__.find('Conv') != -1 or m.__class__.__name__.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)
