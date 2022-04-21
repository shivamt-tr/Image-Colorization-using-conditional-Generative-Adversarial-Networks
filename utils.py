# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:36:07 2022

@author: tripa
"""

import os
import config
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn
from torchvision import transforms

def lab_to_rgb(L, ab):
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    
    return np.stack(rgb_imgs, axis=0)

def init_weights(model):

    def init_func(m):
        
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and 'Conv' in classname:
            
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            
            if hasattr(m, 'bias') and m.bias is not None:
                
                nn.init.constant_(m.bias.data, 0.0)
        
        elif 'BatchNorm2d' in classname:
            
            nn.init.normal_(m.weight.data, 1., 0.02)
            nn.init.constant_(m.bias.data, 0.)
    
    model.apply(init_func)
    print(model.__class__.__name__, 'weights initialized')
    
    return model

def load_transformed_batch(data_dir, batch_files, data_transforms):

    L_channels = []
    a_and_b_channels = []
    
    # Enumerate over all the files in the batch_files list
    for i, x in enumerate(batch_files):
    
        # Open image as PIL and apply transformations
        image = Image.open(os.path.join(data_dir, x)).convert("RGB")
        image = data_transforms(image)
        
        # Convert the image from RGB to LAB format
        LAB_image = rgb2lab(np.array(image)).astype('float32')
        LAB_image = transforms.ToTensor()(LAB_image)
        
        L = LAB_image[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = LAB_image[[1, 2], ...] / 110.  # Between -1 and 1
        
        image_size = L.shape[1]  # L.shape[1] and L.shape[2] denote image height and width
        
        # For the first iteration create a tensor
        if i == 0:
            L_channels = L.reshape(1, 1, image_size, image_size)
            a_and_b_channels = ab.reshape(1, 2, image_size, image_size)
        # For subsequent iterations concatenate tensors to the data
        else:
            L_channels = torch.cat([L_channels, L.reshape(1, 1, image_size, image_size)], axis=0)
            a_and_b_channels = torch.cat([a_and_b_channels, ab.reshape(1, 2, image_size, image_size)], axis=0)

    return L_channels, a_and_b_channels


def load_rgb_batch(data_dir, batch_files, data_transforms):

    images = []
    
    # Enumerate over all the files in the batch_files list
    for i, x in enumerate(batch_files):
    
        # Open image as PIL and apply transformations
        image = Image.open(os.path.join(data_dir, x)).convert("RGB")
        image = data_transforms(image)
        
        image_size = image.shape[1]
        
        # For the first iteration create a tensor
        if i == 0:
            images = image.reshape(1, 3, image_size, image_size)
        # For subsequent iterations concatenate tensors to the data
        else:
            images = torch.cat([images, image.reshape(1, 3, image_size, image_size)], axis=0)
            
    return images

def load_generator(type: str):

    if config.GENERATOR_TYPE == 'UNet':

        from generator import UNet
        generator = UNet(in_channels=1, out_channels=2, n_filters=64)

    if config.GENERATOR_TYPE == 'ResNet':
        
        from generator import ResNet
        generator = ResNet(in_channels=1, out_channels=2)

    if config.GENERATOR_TYPE == 'ResidualUNet':
        
        from generator import ResidualUNet
        generator = ResidualUNet(in_channels=1, out_channels=2, n_filters=64)

        if config.ENHANCE_COLORIZED_IMAGE:
            generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'generator.pth')), map_location=config.DEVICE)
            generator = ResiduialUNetUpsampled(generator, train_base_block=False, in_channels=1, out_channels=3, n_filters=64)

    if config.GENERATOR_TYPE == 'PretrainedUNet':
        
        from fastai.vision.learner import create_body
        from torchvision.models.resnet import resnet18
        from fastai.vision.models.unet import DynamicUnet
        body = create_body(resnet18, pretrained=True, n_in=1, cut=-2)
        generator = DynamicUnet(body, n_out=2, img_size=(256, 256)).to(config.DEVICE)

    return generator