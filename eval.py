# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:00:37 2022

@author: tripa
"""

import os
import torch
import config
import argparse
from tqdm import tqdm

from utils import load_transformed_batch, lab_to_rgb, load_generator
from evaluation_metrics import mean_absolute_error, epsilon_accuracy, peak_signal_to_noise_ratio

import warnings
warnings.filterwarnings("ignore")

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='model type to train or test, specify one of [\'1_generator_base_l1_loss\', \'2_generator_base_content_loss\', \'3_generator_base_l1_and_content_loss\', \'4_generator_resnet_l1_loss\', \'5_generator_residual_unet_l1_loss\', \'6_generator_residual_unet_upsampled_l1_loss\', \'7_generator_base_l1_loss_pretrained\']',
                    type=str,
                    default='1_generator_base_l1_loss')
args = parser.parse_args()

if args.model in ['1_generator_base_l1_loss', '2_generator_base_content_loss', '3_generator_base_l1_and_content_loss']:
    config.GENERATOR_TYPE = 'UNet'
    
if args.model == '4_generator_resnet_l1_loss':
    config.GENERATOR_TYPE = 'ResNet'
    
if args.model == '5_generator_residual_unet_l1_loss':
    config.GENERATOR_TYPE = 'ResidualUNet'
    
if args.model == '6_generator_residual_unet_upsampled_l1_loss':
    config.GENERATOR_TYPE = 'ResidualUNet'
    config.ENHANCE_COLORIZED_IMAGE = True
    
if args.model == '7_generator_base_l1_loss_pretrained':
    config.GENERATOR_TYPE = 'PretrainedUNet'
    config.LOAD_PRETRAINED_GENERATOR = True
    config.PRETRAIN_GENERATOR = False

# %%

# Root directory for test-data
test_files = os.listdir(config.TEST_DIR)[:48]

# Create generator object and load pretrained weights
generator = load_generator(config.GENERATOR_TYPE)
generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, '1_generator_base_l1_loss.pth'), map_location=config.DEVICE))

generator.eval()  # Since we are using only for testing
generator.requires_grad_(False)

# Calculate the number of batches
n_batches = int(len(test_files)/config.BATCH_SIZE)

mae = 0.0
epsilon = 0.0
psnr = 0.0

# Iterate over all the batches
for i in tqdm(range(n_batches), desc='Batch'):
    
    # Get the test data for the current batch
    batch_files = test_files[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
    L, ab = load_transformed_batch(config.TEST_DIR, batch_files, config.VAL_TRANSFORMS)
    
    # L channel + Generator's ab channels make fake images
    if config.ENHANCE_COLORIZED_IMAGE:
        fake_images = generator(L).permute(0, 2, 3, 1).detach().numpy()
    else:
        fake_images = lab_to_rgb(L, generator(L))
    
    # L channel + ground-truth ab channels make real images
    real_images = lab_to_rgb(L, ab)

    mae += 255. * mean_absolute_error(torch.from_numpy(real_images), torch.from_numpy(fake_images)) * config.BATCH_SIZE
    epsilon += epsilon_accuracy(torch.from_numpy(real_images), torch.from_numpy(fake_images), epsilon=0.05) * config.BATCH_SIZE  # epsilon set at 5% of 255
    psnr += peak_signal_to_noise_ratio(torch.from_numpy(real_images), torch.from_numpy(fake_images), max_value=1.) * config.BATCH_SIZE

mae /= (n_batches*config.BATCH_SIZE)
epsilon /= (n_batches*config.BATCH_SIZE)
psnr /= (n_batches*config.BATCH_SIZE)

print('Mean Absolute Error: {:.4f}'.format(mae))
print('Epsilon Accuracy: {:.4f}'.format(epsilon))
print('Peak SNR: {:.4f}'.format(psnr))