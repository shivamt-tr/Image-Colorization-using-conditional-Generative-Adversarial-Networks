#%%

import torch
from torchvision import models

#%%
# Find available device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

#%%
class PerceptualLoss():
    
    def __init__(self, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225], color='lab'):
        
        if color == 'lab':
            self.model = models.vgg16(pretrained=False)
            self.model = torch.load('./models/LabVGG16_BN_epoch120_batchsize32.pth', map_location=device)
        else:
            self.model = models.vgg16(pretrained=True)
        
        self.model = self.model.features[:14 + 1]
        self.model.eval()

        for p in self.model.parameters():
            p.require_grad = False
        
        # self.normalize = transforms.Normalize(normalize_mean, normalize_std, True)
        
    def __call__(self, pred_img, gt_img):

        # pred_features= self.model(self.normalize(pred_img))
        # true_features = self.model(self.normalize(gt_img))

        return F.mse_loss(self.model(pred_img), self.model(gt_img))
    
#%%

###############################################################################
#                              Test Content-Loss                              #
###############################################################################

if __name__ == "__main__":

    # Create a randomly initialized predicted and true images of suitable shapes
    pred_img = torch.rand(2, 3, 256, 256).to(device)
    true_img = torch.rand(2, 3, 256, 256).to(device)

    # Initialize ContentLoss object and pass the ground-truth and predicted images
    content_loss = ContentLoss()(pred_img, true_img)
    print(content_loss)