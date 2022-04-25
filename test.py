# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:30:40 2022

@author: tripa
"""

import os
import torch
import config
import argparse
import matplotlib.pyplot as plt

from utils import load_transformed_batch, load_rgb_batch, lab_to_rgb, load_generator

import warnings
warnings.filterwarnings("ignore")

# %%

# Configure command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='model type to train or test, specify a number between 1-7 or one of [\'1_generator_base_l1_loss\', \'2_generator_base_content_loss\', \'3_generator_base_l1_and_content_loss\', \'4_generator_resnet_l1_loss\', \'5_generator_residual_unet_l1_loss\', \'6_generator_residual_unet_upsampled_l1_loss\', \'7_generator_base_l1_loss_pretrained\']',
                    type=str,
                    default='1_generator_base_l1_loss')
args = parser.parse_args()

if args.model == '1_generator_base_l1_loss' or args.model == '1':
    config.MODEL_NAME = '1_generator_base_l1_loss'
    config.GENERATOR_TYPE = 'UNet'

if args.model == '2_generator_base_content_loss' or args.model == '2':
    config.MODEL_NAME = '2_generator_base_content_loss'
    config.GENERATOR_TYPE = 'UNet'

if args.model == '3_generator_base_l1_and_content_loss' or args.model == '3':
    config.MODEL_NAME = '3_generator_base_l1_and_content_loss'
    config.GENERATOR_TYPE = 'UNet'
    
if args.model == '4_generator_resnet_l1_loss' or args.model == '4':
    config.MODEL_NAME = '4_generator_resnet_l1_loss'
    config.GENERATOR_TYPE = 'ResNet'
    
if args.model == '5_generator_residual_unet_l1_loss' or args.model == '5':
    config.MODEL_NAME = '5_generator_residual_unet_l1_loss'
    config.GENERATOR_TYPE = 'ResidualUNet'
    
if args.model == '6_generator_residual_unet_upsampled_l1_loss' or args.model == '6':
    config.MODEL_NAME = '6_generator_residual_unet_upsampled_l1_loss'
    config.GENERATOR_TYPE = 'ResidualUNet'
    config.ENHANCE_COLORIZED_IMAGE = True
    
if args.model == '7_generator_base_l1_loss_pretrained' or args.model == '7':
    config.MODEL_NAME = '7_generator_base_l1_loss_pretrained'
    config.GENERATOR_TYPE = 'PretrainedUNet'
    config.LOAD_PRETRAINED_GENERATOR = True
    config.PRETRAIN_GENERATOR = False

# %%

# Root directory for test-data
test_dir = os.path.join(os.getcwd(), 'test-images')
test_files = os.listdir(test_dir)

# Set the location of test-results directory and create the directory if it does not exists
res_dir = os.path.join(os.getcwd(), 'test-results')
os.makedirs(res_dir, exist_ok=True)

# Create generator object and load pretrained weights
generator = load_generator(config.GENERATOR_TYPE)
generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, config.MODEL_NAME+'.pth'), map_location=config.DEVICE))

generator.eval()  # Since we are using only for testing
generator.requires_grad_(False)

# Load L and ab channels from the test-images
L, ab = load_transformed_batch(test_dir, test_files, config.VAL_TRANSFORMS)

if config.ENHANCE_COLORIZED_IMAGE:
    # When enhancing the image, we need the RGB ground-truth
    real_images = load_rgb_batch(test_dir, test_files, config.UPSAMPLE_TRANSFORMS)
    real_images = real_images.permute(0, 2, 3, 1).cpu().detach().numpy()
else:
    # In other cases, L channel + ground-truth ab channels make real images (LAB format)
    real_images = lab_to_rgb(L, ab)
   
if config.ENHANCE_COLORIZED_IMAGE:
    # Run the L channel through the generator to get 'RGB' results
    fake_images = generator(L).permute(0, 2, 3, 1).cpu().detach().numpy()
else:
    # Run the L channel through the generator to get 'ab' channels, which is then concatenated with L channel to construct LAB image
    # The LAB image is converted to RGB using lab_to_rgb function
    fake_images = lab_to_rgb(L, generator(L))

# Save results (black-&-white, ground-truth, and generated colored images)
for i in range(len(test_files)):
    
    fig = plt.figure(figsize=(50, 150))
    
    ax = plt.subplot(10, 5, 1)
    ax.imshow(L[i][0].cpu(), cmap='gray')
    ax.axis("off")
    
    ax = plt.subplot(10, 5, 2)
    ax.imshow(real_images[i])
    ax.axis("off")

    ax = plt.subplot(10, 5, 3)
    ax.imshow(fake_images[i])
    ax.axis("off")
    
    fig.subplots_adjust(left=0, bottom=0, right=(1 - 0), top=(1 - 0))
    plt.savefig(os.path.join(res_dir, test_files[i]), bbox_inches='tight')
    plt.close()