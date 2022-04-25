import torch
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENV = 'local'  # set to 'local', 'colab', or 'kaggle'
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
VIS_DIR = "data/visualize"
RES_DIR = "result-logs/"
MODEL_DIR = "models/"
MODEL_NAME = '1_generator_base_l1_loss'
PRETRAIN_GENERATOR = False
LOAD_PRETRAINED_GENERATOR = False
ENHANCE_COLORIZED_IMAGE = False
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
L1_LAMBDA = 100
CONTENT_LAMBDA = 1e-3
NUM_EPOCHS = 50
STARTING_EPOCH = 1
GENERATOR_TYPE = 'UNet'  # 'UNet', 'ResNet', 'ResidualUNet'
LOSS_TYPE = 'l1'  # 'l1', 'content', 'both'

# Transformations for the training data
TRAIN_TRANSFORMS = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.RandomHorizontalFlip()])  # for data augmentation

# Transformations for training upsampled architecture
UPSAMPLE_TRANSFORMS = transforms.Compose([transforms.Resize((512, 512), Image.BICUBIC),
                                          transforms.ToTensor()])

# Transformations for testing the model (resize input image to suitable size for model input)
VAL_TRANSFORMS = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC)])