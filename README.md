# Image Colorization using conditional-Generative Adversarial Networks
 CS776A Course Project: Image Colorization using Conditional-Generative Adversarial Networks (c-GANs)

This work is based on Image-to-Image Translation with condition GANs (Pix2Pix).

## Installing the requirements
```
pip install -r requirements.txt
```

## Datasets and pre-trained models

The dataset and pre-trained models can be downloaded with the following script.
```bash
bash download_datasets_and_models.sh
```

## Experiments

The Pix2Pix model has been modified and trained with several settings. The experimental models are following:

| Model        | Generator           | Discriminator  | Loss Function
| ------------- |:-------------:| -----:|-----:|
| 1_generator_base_l1_loss | UNet | PatchGAN | Adversarial + L1 |
| 2_generator_base_content_loss | UNet | PatchGAN | Adversarial + Content-Loss |
| 3_generator_base_l1_and_content_loss | UNet | PatchGAN | Adversarial + L1 & Content-Loss |
| 4_generator_resnet_l1_loss | ResNet with no downsampling | PatchGAN | Adversarial + L1 |
| 5_generator_residual_unet_l1_loss | ResidualUNet | PatchGAN | Adversarial + L1 |
| 6_generator_residual_unet_upsampled_l1_loss | ResidualUNet with upsampling | PatchGAN | Adversarial + L1 |
| 7_generator_base_l1_loss_pretrained | UNet pre-trained with l1 loss | PatchGAN | Adversarial + L1 |

#### 1. 1_generator_base_l1_loss: Baseline
#### 2. 2_generator_base_content_loss: Baseline with content loss instead of l1
#### 3. 3_generator_base_l1_and_content_loss: Baseline with content loss and l1 loss
#### 4. 4_generator_resnet_l1_loss: With ResNet generator that does not perform any downsampling
#### 5. 5_generator_residual_unet_l1_loss: Experimental ResidualUNet generator and training on adversarial and l1 loss
#### 6. 6_generator_residual_unet_upsampled_l1_loss: Experimental ResidualUNet generator with upsampling block for enhancing output resolution
#### 7. 7_generator_base_l1_loss_pretrained: Pre-training the generator with l1 loss before starting training of baseline model (inspired from  https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

## Training
To train the model on one of the pre-defined tasks you can run the following command.
```bash
python train.py --model model_name --startfrom epoch_number
```

## Testing
- Run the following command to test the model on the images stored in test-images/ and generate results in test-results/.
- Note: Make sure that the download_datasets_and_models.sh has already been run.
```bash
python test.py --model model_name
```

## Pix2Pix paper
### Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros

#### Abstract
We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems. These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations. We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks. Indeed, since the release of the pix2pix software associated with this paper, a large number of internet users (many of them artists) have posted their own experiments with our system, further demonstrating its wide applicability and ease of adoption without the need for parameter tweaking. As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.
```
@misc{isola2018imagetoimage,
      title={Image-to-Image Translation with Conditional Adversarial Networks}, 
      author={Phillip Isola and Jun-Yan Zhu and Tinghui Zhou and Alexei A. Efros},
      year={2018},
      eprint={1611.07004},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
