import torch
from torch import nn

class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class PatchGAN(nn.Module):
    
    def __init__(self, in_channels, n_filters=[64, 128, 256, 512]):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, n_filters[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True))
        self.layer2 = ConvolutionBlock(n_filters[0], n_filters[1], stride=2)
        self.layer3 = ConvolutionBlock(n_filters[1], n_filters[2], stride=2)
        self.layer4 = ConvolutionBlock(n_filters[2], n_filters[3], stride=1)    
        self.layer5 = nn.Conv2d(n_filters[3], 1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x

# %%

###############################################################################
#              Test the discriminator model (PatchGAN-Classifier)             #
###############################################################################

if __name__ == "__main__":

    # Create random vectors for input sample
    noise_x = torch.randn((1, 3, 256, 256))
    # noise_y = torch.randn((1, 3, 256, 256))
    
    # Create PatchGAN object with 3 input_channels
    model = PatchGAN(in_channels=3)
    
    # Run the PatchGAN model to get the output 
    output = model(noise_x)
    
    # Display model information
    print(model)
    print(output.shape)