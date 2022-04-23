# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:30:53 2022

@author: tripa
"""

import os
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import optim

import config
from discriminator import PatchGAN
from content_loss import ContentLoss
from utils import load_transformed_batch, load_rgb_batch, init_weights, lab_to_rgb, load_generator
from evaluation_metrics import mean_absolute_error, epsilon_accuracy, peak_signal_to_noise_ratio

import warnings
warnings.filterwarnings("ignore")

# %%

# Configure command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model',
                    help='model type to train or test, specify one of [\'1_generator_base_l1_loss\', \'2_generator_base_content_loss\', \'3_generator_base_l1_and_content_loss\', \'4_generator_resnet_l1_loss\', \'5_generator_residual_unet_l1_loss\', \'6_generator_residual_unet_upsampled_l1_loss\', \'7_generator_base_l1_loss_pretrained\']',
                    type=str,
                    default='1_generator_base_l1_loss')
parser.add_argument('--startfrom', help='set to 1 if you want to train from scratch', type=int, default=1)
args = parser.parse_args()

config.STARTING_EPOCH = args.startfrom

if args.model == '1_generator_base_l1_loss' or args.model == '1':
    config.MODEL_NAME = '1_generator_base_l1_loss'
    config.GENERATOR_TYPE = 'UNet'
    
if args.model == '2_generator_base_content_loss' or args.model == '2':
    config.MODEL_NAME = '2_generator_base_content_loss'
    config.GENERATOR_TYPE = 'UNet'
    config.LOSS_TYPE = 'content'
    
if args.model == '3_generator_base_l1_and_content_loss' or args.model == '3':
    config.MODEL_NAME = '3_generator_base_l1_and_content_loss'
    config.GENERATOR_TYPE = 'UNet'
    config.LOSS_TYPE = 'both'
    
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

# Create model and result directory if they do not exist
os.makedirs(config.RES_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

# Create a log file if the training starts from first epoch
if config.STARTING_EPOCH == 1:
    header = 'epoch,generator-adversarial-loss,perceptual-or-l1-loss,generator-loss-total,discriminator-loss,mae,epsilon,psnr'
    with open(os.path.join(config.RES_DIR, 'logs.csv'), 'w') as f:
        np.savetxt(f, [], delimiter=',', header=header, comments='')

# List of train, test, and visualization files
train_files = os.listdir(config.TRAIN_DIR)[:17]
test_files = os.listdir(config.TEST_DIR)
vis_files = os.listdir(config.VIS_DIR)

# %%

# Display 16 randomly chosen sample images from train data
random_files = np.random.choice(train_files, size=16)
random_samples = [os.path.join(config.TRAIN_DIR, x) for x in random_files]

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), random_samples):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

# %%

# Create generator object and initialize weights (normally)
generator = load_generator(config.GENERATOR_TYPE)
generator = init_weights(generator)
generator.to(config.DEVICE)

# Create discriminator object and initialize weights (normally)
discriminator = PatchGAN(in_channels=3)
discriminator = init_weights(discriminator)
discriminator.to(config.DEVICE)

# Load pre-trained weights to continue training
if config.STARTING_EPOCH != 1:
    generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'generator.pth'), map_location=config.DEVICE))
    discriminator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'discriminator.pth'), map_location=config.DEVICE))

# Set-up optimizer and scheduler
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Set-up loss functions
adversarial_criterion = torch.nn.BCEWithLogitsLoss()
additional_criterion = torch.nn.L1Loss() if config.LOSS_TYPE == 'l1' else ContentLoss(color='rgb')

# Set-up labels for real and fake predictions
real_label = torch.tensor(1.0)
fake_label = torch.tensor(0.0)

# Calculate the number of batches
n_batches = int(len(train_files)/config.BATCH_SIZE)

# %%

if config.LOAD_PRETRAINED_GENERATOR:
    # Load the saved weights of pre-trained generator (trained only on l1 loss)
    generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'generator_weights_pretrained_l1.pth'), map_location=config.DEVICE))

if config.PRETRAIN_GENERATOR:
    # Pre-training the generator for 20 epochs
    print('Pre-training the Generator')

    pretrain_generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    for epoch in range(1, 20+1):

        print('Epoch {}/{}'.format(epoch, 20))
        print('-' * 10)

        running_generator_loss_l1 = 0.0

        generator.train()  # Set the generator to training mode

        # Iterate over all the batches
        for j in tqdm(range(n_batches), desc='Batch'):

            # Get the train data and labels for the current batch
            batch_files = train_files[j*config.BATCH_SIZE:(j+1)*config.BATCH_SIZE]
            L, ab = load_transformed_batch(config.TRAIN_DIR, batch_files, config.TRAIN_TRANSFORMS)
            
            # Put the data to the device
            L, ab = L.to(config.DEVICE), ab.to(config.DEVICE)

            with torch.set_grad_enabled(True):

                # Run the batch through the generator
                output = generator(L)

                # Calculate the loss
                generator_loss_l1 = additional_criterion(output, ab)

                # Make gradients zero
                pretrain_generator_optimizer.zero_grad()

                # backward + optimize
                generator_loss_l1.backward()
                pretrain_generator_optimizer.step()

            running_generator_loss_l1 += generator_loss_l1.item() * config.BATCH_SIZE

        # Calculate average loss for current epoch
        epoch_generator_loss_l1 = running_generator_loss_l1 / (n_batches*config.BATCH_SIZE)

        print('Generator Loss: {:.4f}'.format(epoch_generator_loss_l1))

        # Save the generator and discriminator model
        torch.save(generator.state_dict(), os.path.join(config.MODEL_DIR, 'generator_pretrained.pth'))

# %%

# Adversarial Training
for epoch in range(config.STARTING_EPOCH, config.NUM_EPOCHS+1):
    
    # Variable to record time taken in each epoch
    since = time.time()
    
    print('Epoch {}/{}'.format(epoch, config.NUM_EPOCHS))
    print('-' * 10)

    running_generator_loss_adversarial = 0.0
    running_generator_loss_additional = 0.0
    running_generator_loss_total = 0.0
    running_discriminator_loss_total = 0.0
    running_mae = 0.0
    running_epsilon = 0.0
    running_psnr = 0.0
    
    # Iterate over all the batches
    for j in tqdm(range(n_batches), desc='Batch'):
            
        # Get the train data and labels for the current batch
        batch_files = train_files[j*config.BATCH_SIZE:(j+1)*config.BATCH_SIZE]
        L, ab = load_transformed_batch(config.TRAIN_DIR, batch_files, config.TRAIN_TRANSFORMS)
        
        # Put the data to the device
        L, ab = L.to(config.DEVICE), ab.to(config.DEVICE)
        
        if config.ENHANCE_COLORIZED_IMAGE:
            # When enhancing the image, we need the RGB ground-truth
            rgb_images = load_rgb_batch(config.TRAIN_DIR, batch_files, config.UPSAMPLE_TRANSFORMS)
            rgb_images = rgb_images.to(config.DEVICE)
        
        # Create a fake color image using the generator
        fake_color = generator(L)

        # Train the discriminator
        discriminator.train()

        # Enable grads
        for p in discriminator.parameters():
            p.requires_grad = True

        # Make gradients zero before forward pass
        discriminator_optimizer.zero_grad()

        # Run fake examples through the discriminator
        if config.ENHANCE_COLORIZED_IMAGE:
            # When enhancing the image, the output is of 3 channels
            fake_image = fake_color    
        else:
            # In other cases, the output is just ab channels so we concatenate the L and ab channels
            fake_image = torch.cat([L, fake_color], dim=1)  # Make dim=0 when passing only one sample
        fake_preds = discriminator(fake_image.detach())
        discriminator_loss_fake = adversarial_criterion(fake_preds, fake_label.expand_as(fake_preds).to(config.DEVICE))
        
        # Run real examples through the discriminator
        if config.ENHANCE_COLORIZED_IMAGE:
            # When enhancing the image, the ground-truth image is taken as the RGB image
            real_image = rgb_images  
        else:
            # In other cases, concatenate the ground-truth ab channels to the L channels to construct the ground-truth LAB image
            real_image = torch.cat([L, ab], dim=1)  # Make dim=0 when passing only one sample
        real_preds = discriminator(real_image)
        discriminator_loss_real = adversarial_criterion(real_preds, real_label.expand_as(real_preds).to(config.DEVICE))
        
        # Total loss is the sum of both the losses
        discriminator_loss_total = (discriminator_loss_fake + discriminator_loss_real) * 0.5
        
        # backward + optimize
        discriminator_loss_total.backward()
        discriminator_optimizer.step()
        
        # Train the generator while keeping the discriminator weights constant
        generator.train()
        
        # Enable grads
        for p in discriminator.parameters():
            p.requires_grad = False

        # Make gradients zero before forward pass
        generator_optimizer.zero_grad()

        # Calculate the prediction using discriminator
        fake_preds = discriminator(fake_image)
        
        # Calculate adversarial loss for the generator
        generator_loss_adversarial = adversarial_criterion(fake_preds, real_label.expand_as(real_preds).to(config.DEVICE))
        
        # Calculate L1 or content loss
        # Total loss is the sum of both the losses
        if config.LOSS_TYPE == 'l1':
            generator_loss_additional = additional_criterion(fake_color, ab) * config.L1_LAMBDA  # Calculates l1 loss
            generator_loss_total = generator_loss_adversarial + generator_loss_additional
        if config.LOSS_TYPE == 'content':
            generator_loss_additional = additional_criterion(fake_image, real_image)  # Calculates content loss
            generator_loss_total = 0.01 * generator_loss_adversarial + generator_loss_additional
        if config.LOSS_TYPE == 'both':
            generator_loss_additional = additional_criterion(fake_image, real_image) + torch.nn.L1Loss()(fake_color, ab) * config.L1_LAMBDA
            generator_loss_total = 0.01 * generator_loss_adversarial + generator_loss_additional
        
        # backward + optimize
        generator_loss_total.backward()
        generator_optimizer.step()
        
        # Add up the accuracy and losses for current batch
        running_generator_loss_adversarial += generator_loss_adversarial.item() * config.BATCH_SIZE
        running_generator_loss_additional += generator_loss_additional.item() * config.BATCH_SIZE
        running_generator_loss_total += generator_loss_total.item() * config.BATCH_SIZE
        running_discriminator_loss_total += discriminator_loss_total.item() * config.BATCH_SIZE
        
        if config.ENHANCE_COLORIZED_IMAGE:
            # When enhancing the image use the RGB ground-truth for the ground-truth image, and RGB output of the generator for fake image
            real_image_array = rgb_images
            fake_image_array = fake_image   
        else:
            # In other cases, use the lab_to_rgb function to convert the LAB image to RGB
            real_image_array = torch.from_numpy(lab_to_rgb(L.detach(), ab.detach()))
            fake_image_array = torch.from_numpy(lab_to_rgb(L.detach(), fake_color.detach()))
        
        running_mae += 255. * mean_absolute_error(real_image_array, fake_image_array) * config.BATCH_SIZE  # Multiplying by 255. because input rgb images have values between 0-1
        running_epsilon += epsilon_accuracy(real_image_array, fake_image_array, epsilon=0.05) * config.BATCH_SIZE  # epsilon set at 5% of 255
        running_psnr += peak_signal_to_noise_ratio(real_image_array, fake_image_array, max_value=1.) * config.BATCH_SIZE

    # Calculate the average accuracy and average loss for current epoch
    epoch_generator_loss_adversarial = running_generator_loss_adversarial / (n_batches*config.BATCH_SIZE)
    epoch_generator_loss_additional = running_generator_loss_additional / (n_batches*config.BATCH_SIZE)
    epoch_generator_loss_total = running_generator_loss_total / (n_batches*config.BATCH_SIZE)
    epoch_discriminator_loss_total = running_discriminator_loss_total / (n_batches*config.BATCH_SIZE)
    epoch_mae = running_mae / (n_batches*config.BATCH_SIZE)
    epoch_epsilon = running_epsilon / (n_batches*config.BATCH_SIZE)
    epoch_psnr = running_psnr / (n_batches*config.BATCH_SIZE)
    
    print('Generator Loss Adversarial: {:.4f}'.format(epoch_generator_loss_adversarial))
    print('Generator Loss L1/Perceptual: {:.4f}'.format(epoch_generator_loss_additional))
    print('Generator Loss Total: {:.4f}'.format(epoch_generator_loss_total))
    print('Discriminator Loss Total: {:.4f}'.format(epoch_discriminator_loss_total))
    print('Mean Absolute Error: {:.4f}'.format(epoch_mae))
    print('Epsilon Accuracy: {:.4f}'.format(epoch_epsilon))
    print('Peak SNR: {:.4f}'.format(epoch_psnr))
    
    time_elapsed = time.time() - since
    print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # Save the generator and discriminator model
    torch.save(generator.state_dict(), os.path.join(config.MODEL_DIR, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.MODEL_DIR, 'discriminator.pth'))
    
    # Save loss and accuracy to log file
    log = [[epoch,
            epoch_generator_loss_adversarial,
            epoch_generator_loss_additional,
            epoch_generator_loss_total,
            epoch_discriminator_loss_total,
            epoch_mae,
            epoch_epsilon,
            epoch_psnr]]
    with open(os.path.join(config.RES_DIR, 'logs.csv'), 'a') as f:
        np.savetxt(f, log, delimiter=',')
    
    # Test the model on 50 sample images to visualize the colorization
    generator.eval()

    with torch.no_grad():
    
        # Transform the images and get their L and ab channels
        L, ab = load_transformed_batch(config.VIS_DIR, vis_files, config.VAL_TRANSFORMS)
        L, ab = L.to(config.DEVICE), ab.to(config.DEVICE)
        
        if config.ENHANCE_COLORIZED_IMAGE:
            # Run the L channel through the generator to get 'RGB' results
            res_images = generator(L).permute(0, 2, 3, 1).detach().numpy()
        else:
            # Run the L channel through the generator to get 'ab' channels, which is then concatenated with L channel to construct LAB image
            # The LAB image is converted to RGB using lab_to_rgb function
            res_images = lab_to_rgb(L, generator(L))
            
        # Create directory for saving visualizations of images for the current epoch
        vis_result_dir = os.path.join(config.RES_DIR, 'epoch '+str(epoch))
        os.makedirs(vis_result_dir, exist_ok=True)
        
        # Save output images for this epoch
        for i in range(len(res_images)):
            image = res_images[i] * 255  # Make values between 0-255 (originally it is between 0-1)
            image = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 type
            image = image.resize((512, 512))  # Resize all images to (512, 512)
            image.save(os.path.join(vis_result_dir, vis_files[i]))
            
    generator.train()  # Set the generator back to training mode
