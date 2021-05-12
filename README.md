# Super-Resolution Networks for Pytorch

Super-resolution is a process that increases the resolution of an image, adding additional details.
Neural networks are the go-to method for accurate or realistic super-resolution.

A low-resolution image, magnified x4 by a neural network, and the high resolution image:

![Pixelated image of a butterfly](doc/example_small.png "Low resolution image")
![Smooth magnified image](doc/example_x4.png "Magnified x4")
![High resolution image](doc/example_hr.png "High resolution image")


In this repository, you will find:
* the popular super-resolution networks, pretrained
* common super-resolution datasets
* a unified training script for all models



## Usage

```python
from datasets import Div2K
from models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor

# Div2K dataset
dataset = Div2K(root="./data", scale=2, download=False)

# Get the first image in the dataset (High-Res and Low-Res)
hr, lr = dataset[0]

# Download a pretrained NinaSR model
model = ninasr_b0(scale=2, pretrained=True)

# Run the Super-Resolution model
lr_t = to_tensor(lr).unsqueeze(0)
sr_t = model(lr_t)
sr = to_pil_image(sr_t.squeeze(0))
sr.show()
```

<details>
<summary>More examples</summary>


```python
import datasets
import models
import transforms

# Div2K dataset, cropped to 256px, width color jitter
dataset = datasets.Div2K(
    root="./data", scale=2, download=False,
    transform=transforms.Compose([
        transforms.RandomCrop(256, scales=[1, 2]),
        transforms.ColorJitter(brightness=0.2)
    ]))

# Pretrained RCAN model, with tiling for large images
model = models.utils.ChoppedModel(
    models.rcan(scale=2, pretrained=True), scale=2,
    chop_size=400, chop_overlap=10)

# Pretrained EDSR model, with self-ensemble method for higher quality
model = models.utils.SelfEnsemble(models.edsr(scale=2, pretrained=True))
```
</details>

## Models

The following pretrained models are available:
* [EDSR](https://arxiv.org/abs/1707.02921) (x2 x3 x4)
* [CARN](https://arxiv.org/abs/1803.08664) (x2 x3 x4)
* [RDN](https://arxiv.org/abs/1802.08797) (x2 x3 x4)
* [RCAN](https://arxiv.org/abs/1807.02758) (x2 x3 x4 x8)
* [NinaSR](doc/NinaSR.md), my own model (x2 x3 x4 x8)

<details>
<summary>DIV2K validation results</summary>


|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) | 8x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 34.58 / 0.9373 | 30.91 / 0.8734 | 28.98 / 0.8188 | N/A            |
| carn\_m             | 0.41           | 34.29 / 0.9350 | 30.65 / 0.8689 | 28.73 / 0.8131 | N/A            |
| edsr\_baseline      | 1.37           | 34.66 / 0.9379 | 30.96 / 0.8743 | 28.99 / 0.8191 | N/A            |
| edsr                | 40.7           | 35.08 / 0.9413 | 31.30 / 0.8804 | 29.30 / 0.8274 | N/A            |
| ninasr\_b0          | 0.10           | 34.25 / 0.9346 | 30.56 / 0.8670 | 28.63 / 0.8102 | 25.12 / 0.6799 |
| ninasr\_b1          | 1.02           | 34.76 / 0.9388 | 31.04 / 0.8757 | 29.08 / 0.8216 | 25.48 / 0.6928 |
| ninasr\_b2          | 10.0           | 35.06 / 0.9411 | 31.29 / 0.8797 | 29.29 / 0.8267 | 25.62 / 0.6983 |
| rcan                | 15.4           | 35.13 / 0.9416 | 31.34 / 0.8807 | 29.30 / 0.8276 | 25.73 / 0.7036 |
| rdn                 | 22.1           | 34.85 / 0.9394 | 30.59 / 0.8678 | 29.17 / 0.8240 | N/A            |

</details>


## Datasets

Datasets return a list of images. The first image is the original one, and the next images are downscaled or degraded versions.

The following datasets are available:
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* [RealSR](https://github.com/csjcai/RealSR)
* [Flicr2K](https://github.com/limbee/NTIRE2017)
* [REDS](https://seungjunnah.github.io/Datasets/reds)
* [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://paperswithcode.com/dataset/set14), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [Urban100](https://paperswithcode.com/dataset/urban100)

They are downloaded automatically when using the `download=True` flag, or by running the corresponding script i.e. `./scripts/download_div2k.sh`.



## Transforms

Transforms are used for preprocessing and data augmentation. They are applied identically to the original and downscaled image.

This repository defines several transforms that follow torchvision conventions:
* ToTensor, ToPILImage
* Compose
* RandomCrop
* ColorJitter
* GaussianBlur
* RandomHorizontalFlip, RandomVerticalFlip, RandomFlipTurn



## Training

A script is available to train the models from scratch, evaluate them, and much more:
```bash
pip install pip tqdm tensorboard  # Additional dependencies
python main.py -h
python main.py --arch edsr_baseline --scale 2 --download-pretrained --images test/butterfly.png --destination results/
python main.py --arch edsr_baseline --scale 2 --download-pretrained --validation-only
python main.py --arch edsr_baseline --scale 2 --epochs 300 --loss l1 --dataset-train div2k_bicubic
```



# Contributions

All contributions are welcome! Usability improvements, training improvements, new models, new transforms, ...
Pretrained models are particularly welcome.



# Acknowledgements

Thanks to the people behind [torchvision](https://github.com/pytorch/vision) and [EDSR](https://github.com/zhouhuanxiang/EDSR-PyTorch), whose work inspired this repository.

Some of the model codes used here come from [EDSR-PyTorch](https://github.com/zhouhuanxiang/EDSR-PyTorch) and [CARN-PyTorch](https://github.com/nmhkahn/CARN-pytorch).
