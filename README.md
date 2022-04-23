# Image Colorization using conditional-Generative Adversarial Networks
 CS776A Course Project: Image Colorization using Conditional-Generative Adversarial Networks (c-GANs)

This work is based on Image-to-Image Translation with condition GANs (Pix2Pix).

## Installing the requirements
```
pip install -r requirements.txt
```

## Datasets and pre-trained models

The dataset can be downloaded from kaggle [mscoco7k](https://www.kaggle.com/datasets/shivam28/mscoco7k) and pre-trained models can be downloaded with the following link [model-weights](https://github.com/shivamt-tr/Image-Colorization-using-conditional-Generative-Adversarial-Networks/releases/tag/initial).

Save the model weights in models/ and data files in data/ directory.

## Experiments

The Pix2Pix model has been modified and trained with several settings. The experimental models are following:

| SNo | Model        | Generator           | Discriminator  | Loss Function |
| ---: | ------------- |:--------------:| -----:|-----:|
| 1 | 1_generator_base_l1_loss | UNet | PatchGAN | Adversarial + L1 |
| 2 | 2_generator_base_content_loss | UNet | PatchGAN | Adversarial + Content-Loss |
| 3 | 3_generator_base_l1_and_content_loss | UNet | PatchGAN | Adversarial + L1 & Content-Loss |
| 4 | 4_generator_resnet_l1_loss | ResNet with no downsampling | PatchGAN | Adversarial + L1 |
| 5 | 5_generator_residual_unet_l1_loss | ResidualUNet | PatchGAN | Adversarial + L1 |
| 6 | 6_generator_residual_unet_upsampled_l1_loss | ResidualUNet with upsampling | PatchGAN | Adversarial + L1 |
| 7 | 7_generator_base_l1_loss_pretrained | UNet pre-trained with l1 loss | PatchGAN | Adversarial + L1 |

Experiment 7 is inspired from [Colorizing black & white images with U-Net and conditional GAN — A Tutorial](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)

## Training
To train the model on one of the pre-defined tasks you can run the following command.
```bash
python train.py --model model_name --startfrom epoch_number
```

## Evaluation
To evaluate the model on the test dataset, run the following command.
```bash
python eval.py --model model_name
```

## Frechet Inception Distance (FID Score)
To calculate the FID score, first run eval.py on some model which will save the results of test dataset in ./data/eval-results directory. After eval.py has run successfully run the following command.
```bash
python -m pytorch_fid path/to/eval/images path/to/test/images --num-workers 0
```

For example:
```bash
python -m pytorch_fid ./data/test ./data/eval-results --num-workers 0
```


## Testing
- Run the following command to test the model on the images stored in test-images/ and generate results in test-results/.
- Note: Make sure that the model weights are present in models/ directory.
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

## FID Score in PyTorch

```
@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.2.1},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)