import os
import torch
import config
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import load_transformed_batch, load_rgb_batch, lab_to_rgb, load_generator
from evaluation_metrics import mean_absolute_error, epsilon_accuracy, peak_signal_to_noise_ratio

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
test_files = os.listdir(config.TEST_DIR)

# Create directory for saving visualizations of images for the current epoch
result_dir = os.path.join(os.getcwd(), 'data', 'eval-results')
os.makedirs(result_dir, exist_ok=True)

# Create generator object and load pretrained weights
generator = load_generator(config.GENERATOR_TYPE)
generator.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, config.MODEL_NAME+'.pth'), map_location=config.DEVICE))

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
    
    if config.ENHANCE_COLORIZED_IMAGE:
        # When enhancing the image, we need the RGB ground-truth
        real_images = load_rgb_batch(config.TEST_DIR, batch_files, config.UPSAMPLE_TRANSFORMS)
        real_images = real_images.to(config.DEVICE)
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

    mae += 255. * mean_absolute_error(torch.from_numpy(real_images), torch.from_numpy(fake_images)) * config.BATCH_SIZE  # Multiplying by 255. because input rgb images have values between 0-1
    epsilon += epsilon_accuracy(torch.from_numpy(real_images), torch.from_numpy(fake_images), epsilon=0.05) * config.BATCH_SIZE  # epsilon set at 5% of 255
    psnr += peak_signal_to_noise_ratio(torch.from_numpy(real_images), torch.from_numpy(fake_images), max_value=1.) * config.BATCH_SIZE

    # Save output images for this epoch
    for i in range(len(fake_images)):
        image = fake_images[i] * 255  # Make values between 0-255 (originally it is between 0-1)
        image = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 type
        image = image.resize((512, 512))  # Resize all images to (512, 512)
        image.save(os.path.join(result_dir, batch_files[i]))

mae /= (n_batches*config.BATCH_SIZE)
epsilon /= (n_batches*config.BATCH_SIZE)
psnr /= (n_batches*config.BATCH_SIZE)

print('Mean Absolute Error: {:.4f}'.format(mae))
print('Epsilon Accuracy: {:.4f}'.format(epsilon))
print('Peak SNR: {:.4f}'.format(psnr))