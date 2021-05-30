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
<summary>Set5 results</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 37.88 / 0.9600 | 34.32 / 0.9265 | 32.14 / 0.8942 |
| carn\_m             | 0.41           | 37.68 / 0.9594 | 34.06 / 0.9247 | 31.88 / 0.8907 |
| edsr\_baseline      | 1.37           | 37.98 / 0.9604 | 34.37 / 0.9270 | 32.09 / 0.8936 |
| edsr                | 40.7           | 38.19 / 0.9609 | 34.68 / 0.9293 | 32.48 / 0.8985 |
| ninasr\_b0          | 0.10           | 37.69 / 0.9594 | 33.91 / 0.9229 | 31.65 / 0.8868 |
| ninasr\_b1          | 1.02           | 38.00 / 0.9604 | 34.42 / 0.9274 | 32.21 / 0.8947 |
| ninasr\_b2          | 10.0           | 38.22 / 0.9612 | 34.63 / 0.9288 | 32.48 / 0.8976 |
| rcan                | 15.4           | 38.27 / 0.9614 | 34.76 / 0.9299 | 32.64 / 0.9000 |
| rdn                 | 22.1           | 38.12 / 0.9609 | 33.98 / 0.9234 | 32.35 / 0.8968 |

</details>

<details>
<summary>Set14 results</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 33.57 / 0.9173 | 30.30 / 0.8412 | 28.61 / 0.7806 |
| carn\_m             | 0.41           | 33.30 / 0.9151 | 30.10 / 0.8374 | 28.42 / 0.7764 |
| edsr\_baseline      | 1.37           | 33.57 / 0.9174 | 30.28 / 0.8414 | 28.58 / 0.7804 |
| edsr                | 40.7           | 33.95 / 0.9201 | 30.53 / 0.8464 | 28.81 / 0.7872 |
| ninasr\_b0          | 0.10           | 33.23 / 0.9147 | 30.01 / 0.8352 | 28.26 / 0.7723 |
| ninasr\_b1          | 1.02           | 33.61 / 0.9176 | 30.37 / 0.8430 | 28.65 / 0.7824 |
| ninasr\_b2          | 10.0           | 33.99 / 0.9206 | 30.55 / 0.8461 | 28.81 / 0.7865 |
| rcan                | 15.4           | 34.13 / 0.9216 | 30.63 / 0.8475 | 28.85 / 0.7878 |
| rdn                 | 22.1           | 33.71 / 0.9182 | 30.07 / 0.8373 | 28.72 / 0.7846 |

</details>

<details>
<summary>DIV2K results (validation set)</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) | 8x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 36.08 / 0.9451 | 32.37 / 0.8871 | 30.43 / 0.8366 | N/A            |
| carn\_m             | 0.41           | 35.76 / 0.9429 | 32.09 / 0.8827 | 30.18 / 0.8313 | N/A            |
| edsr\_baseline      | 1.37           | 36.13 / 0.9455 | 32.41 / 0.8878 | 30.43 / 0.8370 | N/A            |
| edsr                | 40.7           | 36.56 / 0.9485 | 32.75 / 0.8933 | 30.73 / 0.8445 | N/A            |
| ninasr\_b0          | 0.10           | 35.72 / 0.9424 | 32.01 / 0.8811 | 30.08 / 0.8289 | 26.58 / 0.7076 |
| ninasr\_b1          | 1.02           | 36.23 / 0.9463 | 32.49 / 0.8891 | 30.53 / 0.8394 | 26.92 / 0.7195 |
| ninasr\_b2          | 10.0           | 36.54 / 0.9484 | 32.74 / 0.8927 | 30.74 / 0.8441 | 27.07 / 0.7247 |
| rcan                | 15.4           | 36.61 / 0.9489 | 32.78 / 0.8935 | 30.73 / 0.8447 | 27.17 / 0.7292 |
| rdn                 | 22.1           | 36.32 / 0.9468 | 32.04 / 0.8822 | 30.61 / 0.8414 | N/A            |

</details>

<details>
<summary>B100 results</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 32.12 / 0.8986 | 29.07 / 0.8042 | 27.58 / 0.7355 |
| carn\_m             | 0.41           | 31.97 / 0.8971 | 28.94 / 0.8010 | 27.45 / 0.7312 |
| edsr\_baseline      | 1.37           | 32.15 / 0.8993 | 29.08 / 0.8051 | 27.56 / 0.7354 |
| edsr                | 40.7           | 32.35 / 0.9019 | 29.26 / 0.8096 | 27.72 / 0.7419 |
| ninasr\_b0          | 0.10           | 31.94 / 0.8969 | 28.87 / 0.7996 | 27.35 / 0.7285 |
| ninasr\_b1          | 1.02           | 32.19 / 0.8999 | 29.11 / 0.8056 | 27.60 / 0.7369 |
| ninasr\_b2          | 10.0           | 32.34 / 0.9018 | 29.25 / 0.8090 | 27.71 / 0.7411 |
| rcan                | 15.4           | 32.39 / 0.9024 | 29.30 / 0.8106 | 27.74 / 0.7429 |
| rdn                 | 22.1           | 32.25 / 0.9006 | 28.90 / 0.8004 | 27.66 / 0.7388 |

</details>

<details>
<summary>Urban100 results</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 31.95 / 0.9263 | 28.07 / 0.849 | 26.07 / 0.78349 |
| carn\_m             | 0.41           | 31.30 / 0.9200 | 27.57 / 0.839 | 25.64 / 0.76961 |
| edsr\_baseline      | 1.37           | 31.98 / 0.9271 | 28.15 / 0.852 | 26.03 / 0.78424 |
| edsr                | 40.7           | 32.97 / 0.9358 | 28.81 / 0.865 | 26.65 / 0.80328 |
| ninasr\_b0          | 0.10           | 31.21 / 0.9190 | 27.37 / 0.834 | 25.40 / 0.76207 |
| ninasr\_b1          | 1.02           | 32.18 / 0.9288 | 28.23 / 0.854 | 26.11 / 0.78772 |
| ninasr\_b2          | 10.0           | 32.92 / 0.9356 | 28.69 / 0.863 | 26.55 / 0.80087 |
| rcan                | 15.4           | 33.19 / 0.9372 | 29.01 / 0.868 | 26.75 / 0.80624 |
| rdn                 | 22.1           | 32.41 / 0.9310 | 27.49 / 0.838 | 26.36 / 0.79460 |

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

A script is available to train the models from scratch, evaluate them, and much more. Install the dependencies first, as shown here:
```bash
pip install piq tqdm tensorboard  # Additional dependencies
python main.py -h
python main.py --arch edsr_baseline --scale 2 --download-pretrained --images test/butterfly.png --destination results/
python main.py --arch edsr_baseline --scale 2 --download-pretrained --validation-only
python main.py --arch edsr_baseline --scale 2 --epochs 300 --loss l1 --dataset-train div2k_bicubic
```



# Acknowledgements

Thanks to the people behind [torchvision](https://github.com/pytorch/vision) and [EDSR](https://github.com/zhouhuanxiang/EDSR-PyTorch), whose work inspired this repository.
Some of the models available here come from [EDSR-PyTorch](https://github.com/zhouhuanxiang/EDSR-PyTorch) and [CARN-PyTorch](https://github.com/nmhkahn/CARN-pytorch).

To cite this work, please use:

```
@misc{torchsr,
  author = {Gabriel Gouvine},
  title = {Super-Resolution Networks for Pytorch},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Coloquinte/torchSR}}
}
```
