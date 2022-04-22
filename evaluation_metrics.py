# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:26:49 2022

@author: Himanshu Lal
"""

import torch
from math import log10, sqrt

def mean_absolute_error(pred_img: torch.tensor, true_img: torch.tensor):
    
    assert pred_img.shape[0] > 0 and true_img.shape[0] > 0, "lists are empty"
    assert pred_img.shape[0] == true_img.shape[0], "num of images in pred_img is not matching with num of images in true_img"
    
    mae = torch.sum(torch.abs(pred_img - true_img)).item()
    
    return mae/true_img.numel()


def epsilon_accuracy(pred_img: torch.tensor, true_img: torch.tensor, epsilon=0.5):

    assert pred_img.shape[0] > 0 and true_img.shape[0] > 0, "lists are empty"
    assert pred_img.shape[0] == true_img.shape[0], "num of images in pred_img is not matching with num of images in true_img"

    temp = torch.abs(pred_img - true_img)
    ee = temp[temp < epsilon].shape[0]

    return ee/true_img.numel()


def peak_signal_to_noise_ratio(pred_img: torch.tensor, true_img: torch.tensor, max_value=1):

    assert pred_img.shape[0] > 0 and true_img.shape[0] > 0, "lists are empty"
    assert pred_img.shape[0] == true_img.shape[0], "num of images in pred_img is not matching with num of images in true_img"
    
    num_sample = true_img.shape[0]
    psnr = 0
    
    for p_img, t_img in zip(pred_img, true_img):      
        mse = torch.mean((p_img.type(torch.float32) - t_img.type(torch.float32)) ** 2)
        psnr += 100 if mse == 0 else 20 * log10(max_value / sqrt(mse)) 
    
    return psnr/num_sample


# %%

###############################################################################
#                        Test the Evaluation Metrics                          #
###############################################################################

if __name__ == "__main__":

    # Create predicated and true image tensors and calculate mean absolute error
    pred_img = torch.cat([torch.ones(3, 5, 5), torch.ones(3, 5, 5)*5]).view(2, 3, 5, 5)
    true_img =  torch.cat([torch.ones(3, 5, 5)*2, torch.ones(3, 5, 5)*7]).view(2, 3, 5, 5)
    print(mean_absolute_error(pred_img, true_img))  # output = 112.5
    
    # Create predicated and true image tensors and calculate epsilon-error
    pred_img = torch.rand(1, 3, 5, 5)
    true_img = torch.rand(1, 3, 5, 5)
    print(epsilon_accuracy(pred_img, true_img, epsilon=0.5))
    
    # Create predicated and true image tensors and calculate peak signal to noise ratio
    pred_img = torch.tensor([[[[1, 2], [2, 3]]], [[[3, 4], [5, 6]]]])
    true_img = torch.tensor([[[1, 2], [2, 3]]], [[[2, 4], [6, 6]]])
    print(peak_signal_to_noise_ratio(pred_img, true_img))  # output = 75.57055178265946