# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 00:12:31 2022

@author: tripa
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU() if upsample else nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):

        return torch.add(self.residual_block(x), x)


class UpsampleBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__() 
        
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding='same'),
            nn.PixelShuffle(2),
            nn.PReLU()
            )
        
    def forward(self, x):
        
        return self.upsample(x)


class UNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.act = nn.LeakyReLU(0.2, True)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.act = nn.ReLU(True)
            
        self.norm = nn.BatchNorm2d(out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        
        if self.use_dropout:
            return self.dropout(self.act(self.norm(self.conv(x))))
        else:
            return self.act(self.norm(self.conv(x)))


class ResidualUNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        
        if downsample:
            self.unetblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
                )
            self.act = nn.LeakyReLU(0.2)
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.unetblock = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
                )
            self.act = nn.ReLU()
            self.shortcut_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.use_dropout = use_dropout 
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
    
        shortcut = self.shortcut_conv(x)
        if self.use_dropout:
            return self.dropout(self.act(self.norm(torch.add(self.unetblock(x), shortcut))))
        else:
            return self.act(self.norm(torch.add(self.unetblock(x), shortcut)))


class ResNet():
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU())
        
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 64)
        self.res4 = ResidualBlock(64, 64)
        self.res5 = ResidualBlock(64, 64)
        self.res6 = ResidualBlock(64, 64)
        self.res7 = ResidualBlock(64, 64)
        self.res8 = ResidualBlock(64, 64)

        self.conv2 = nn.Sequential(nn.Conv2d(64, out_channels, 3, 1, 1), nn.Tanh())
    
    def forward(self, x):
        
        temp = x

        x = self.conv1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        
        final = self.conv2(torch.add(x, temp))
        
        return final


class UNet(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=2, n_filters=64):
        super().__init__()
        
        self.downsample_initial = nn.Sequential(nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, True))
        
        self.down1 = UNetBlock(n_filters, n_filters*2, downsample=True, use_dropout=False)
        self.down2 = UNetBlock(n_filters*2, n_filters*4, downsample=True, use_dropout=False)
        self.down3 = UNetBlock(n_filters*4, n_filters*8, downsample=True, use_dropout=False)
        self.down4 = UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False)
        self.down5 = UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False)
        self.down6 = UNetBlock(n_filters*8, n_filters*8, downsample=True, use_dropout=False)
        
        self.downsample_inner = nn.Sequential(nn.Conv2d(n_filters*8, n_filters*8, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2, True))
        self.upsample_inner = UNetBlock(n_filters*8, n_filters*8, downsample=False, use_dropout=False)
        
        self.up1 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up2 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up3 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up4 = UNetBlock(n_filters*8*2, n_filters*4, downsample=False, use_dropout=False)
        self.up5 = UNetBlock(n_filters*4*2, n_filters*2, downsample=False, use_dropout=False)
        self.up6 = UNetBlock(n_filters*2*2, n_filters, downsample=False, use_dropout=False)
        
        self.upsample_final = nn.Sequential(nn.ConvTranspose2d(n_filters*2, out_channels, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        
    def forward(self, x):
        
        d_initial = self.downsample_initial(x)
        d1 = self.down1(d_initial)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        d_inner = self.downsample_inner(d6)
        u_inner = self.upsample_inner(d_inner)
        
        u1 = self.up1(torch.cat([u_inner, d6], dim=1))
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        
        u_final = self.upsample_final(torch.cat([u6, d_initial], dim=1))
        
        return u_final


class ResidualUNet():

    def __init__(self, in_channels=1, out_channels=3, n_filters=64):
        super().__init__()
        
        self.downsample_initial = nn.Sequential(nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2))
        
        self.down1 = UNetBlock(n_filters, n_filters*2, downsample=True,use_dropout=False)
        self.down2 = UNetBlock(n_filters*2, n_filters*4, downsample=True,use_dropout=False)
        self.down3 = UNetBlock(n_filters*4, n_filters*8, downsample=True,use_dropout=False)
        self.down4 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        self.down5 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        self.down6 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        
        self.downsample_inner = nn.Sequential(nn.Conv2d(n_filters*8, n_filters*8, kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(0.2))
        self.upsample_inner = nn.Sequential(nn.ConvTranspose2d(n_filters*8, n_filters*8, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(n_filters*8), nn.ReLU())
        
        self.up1 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up2 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up3 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up4 = UNetBlock(n_filters*8*2, n_filters*4, downsample=False, use_dropout=False)
        self.up5 = UNetBlock(n_filters*4*2, n_filters*2, downsample=False, use_dropout=False)
        self.up6 = UNetBlock(n_filters*2*2, n_filters, downsample=False, use_dropout=False)
        
        self.upsample_final = nn.Sequential(nn.ConvTranspose2d(n_filters*2, 2, kernel_size=4, stride=2, padding=1, bias=False), nn.Tanh())
        
    def forward(self, x):
        
        d_initial = self.downsample_initial(x)

        d1 = self.down1(d_initial)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        d_inner = self.downsample_inner(d6)
        u_inner = self.upsample_inner(d_inner)
        
        u1 = self.up1(torch.cat([u_inner, d6], dim=1))
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        
        u_final = self.upsample_final(torch.cat([u6, d_initial], dim=1))
        
        return u_final


class ResidualUNetUpsampled(nn.Module):
    
    def __init__(self, model, train_base_block=False, in_channels=1, out_channels=3, n_filters=64):
        super().__init__()
        
        # Initial model
        self.model = model
        
        for p in self.model.parameters():
            p.requires_grad = train_base_block
        
        # Residual network with upsampling blocks
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU())
        
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res3 = ResidualBlock(64, 64)
        self.res4 = ResidualBlock(64, 64)
        self.res5 = ResidualBlock(64, 64)
        self.res6 = ResidualBlock(64, 64)
        
        self.conv_add = nn.Sequential(nn.Conv2d(64, 3, 3, 1, 1), nn.BatchNorm2d(3))
        self.conv2 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.pixup1 = Upsample(64)
        self.final = nn.Sequential(nn.Conv2d(64, out_channels, 9, 1, 4))
        
    def forward(self, x):
        
        temp = x
        x = self.model(x)
        out_model = torch.cat([temp, x], dim = 1)
        conv1 = self.conv1(out_model)
        
        res1 = self.res1(conv1)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        res6 = self.res6(res5)
        
        conv_add = self.conv_add(res6)
        conv2 = self.conv2(torch.add(out_model, conv_add))
        pixup1 = self.pixup1(conv2)
        final = self.final(pixup1)
        
        return final


# %%

###############################################################################
#                       Test the generator model (U-Net)                      #
###############################################################################

if __name__ == "__main__":

    # Create a random noise vector of suitable shape
    noise_vector = torch.randn((1, 1, 256, 256))
    
    # Create U-Net model object with 3 input channels and 64 features
    model = UNet(in_channels=1, out_channels=2, n_filters=64)
    
    # Run model to get the output
    output = model(noise_vector)
    
    # Convert the output to numpy array
    output = output.cpu().detach().numpy()
    
    # Concatenate the two output channels from model output with the noise vector to get 3 channel image
    output = np.concatenate([noise_vector.cpu().detach().numpy(), output], axis=1)
    
    # Transpose the output array to make image-channels as the last dimensions
    output = np.transpose(output[0], (1, 2, 0))
    
    # Display the image
    plt.imshow(output)
    plt.show()