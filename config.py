import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_MODEL = False
LOAD_MODEL = True
LOAD_PRETRAINED_GENERATOR = True
ENHANCE_COLORIZED_IMAGE = True