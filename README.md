[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4868308.svg)](https://doi.org/10.5281/zenodo.4868308)
[![PyPI](https://img.shields.io/pypi/v/torchsr?color=orange)](https://pypi.org/project/torchsr/)
[![GitHub](https://img.shields.io/github/license/coloquinte/torchsr?color=lightgrey)](https://opensource.org/licenses/MIT)


# Super-Resolution Networks for Pytorch

[Super-resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) is a process that increases the resolution of an image, adding additional details.
Methods using neural networks give the most accurate results, much better than other interpolation methods.
With the right training, it is even possible to make photo-realistic images.

For example, here is a low-resolution image, magnified x4 by a neural network, and a high resolution image of the same object:

![Pixelated image of a butterfly](https://raw.githubusercontent.com/Coloquinte/torchSR/v1.0.2/doc/example_small.png "Low resolution image")
![Smooth magnified image](https://raw.githubusercontent.com/Coloquinte/torchSR/v1.0.2/doc/example_x4.png "Magnified x4")
![High resolution image](https://raw.githubusercontent.com/Coloquinte/torchSR/v1.0.2/doc/example_hr.png "High resolution image")


In this repository, you will find:
* the popular super-resolution networks, pretrained
* common super-resolution datasets
* a unified training script for all models



## Models

The following pretrained models are available. Click on the links for the paper:
* [EDSR](https://arxiv.org/abs/1707.02921)
* [CARN](https://arxiv.org/abs/1803.08664)
* [RDN](https://arxiv.org/abs/1802.08797)
* [RCAN](https://arxiv.org/abs/1807.02758)
* [NinaSR](doc/NinaSR.md)

Newer and larger models perform better: the most accurate models are EDSR (huge), RCAN and NinaSR-B2.
For practical applications, I recommend a smaller model, such as NinaSR-B1.


<details>
<summary>Expand benchmark results</summary>

<details>
<summary>Set5 results</summary>

|  Network            | Parameters (M) | 2x (PSNR/SSIM) | 3x (PSNR/SSIM) | 4x (PSNR/SSIM) |
| ------------------- | -------------- | -------------- | -------------- | -------------- |
| carn                | 1.59           | 37.88 / 0.9600 | 34.32 / 0.9265 | 32.14 / 0.8942 |
| carn\_m             | 0.41           | 37.68 / 0.9594 | 34.06 / 0.9247 | 31.88 / 0.8907 |
| edsr\_baseline      | 1.37           | 37.98 / 0.9604 | 34.37 / 0.9270 | 32.09 / 0.8936 |
| edsr                | 40.7           | 38.19 / 0.9609 | 34.68 / 0.9293 | 32.48 / 0.8985 |
| ninasr\_b0          | 0.10           | 37.72 / 0.9594 | 33.96 / 0.9234 | 31.77 / 0.8877 |
| ninasr\_b1          | 1.02           | 38.14 / 0.9609 | 34.48 / 0.9277 | 32.28 / 0.8955 |
| ninasr\_b2          | 10.0           | 38.21 / 0.9612 | 34.61 / 0.9288 | 32.45 / 0.8973 |
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
| ninasr\_b0          | 0.10           | 33.24 / 0.9144 | 30.02 / 0.8355 | 28.28 / 0.7727 |
| ninasr\_b1          | 1.02           | 33.71 / 0.9189 | 30.41 / 0.8437 | 28.71 / 0.7840 |
| ninasr\_b2          | 10.0           | 34.00 / 0.9206 | 30.53 / 0.8461 | 28.80 / 0.7863 |
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
| ninasr\_b0          | 0.10           | 35.77 / 0.9428 | 32.06 / 0.8818 | 30.09 / 0.8293 | 26.60 / 0.7084 |
| ninasr\_b1          | 1.02           | 36.35 / 0.9471 | 32.51 / 0.8892 | 30.56 / 0.8405 | 26.96 / 0.7207 |
| ninasr\_b2          | 10.0           | 36.52 / 0.9482 | 32.73 / 0.8926 | 30.73 / 0.8437 | 27.07 / 0.7246 |
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
| ninasr\_b0          | 0.10           | 31.97 / 0.8974 | 28.90 / 0.8000 | 27.36 / 0.7290 |
| ninasr\_b1          | 1.02           | 32.24 / 0.9004 | 29.13 / 0.8061 | 27.62 / 0.7377 |
| ninasr\_b2          | 10.0           | 32.32 / 0.9014 | 29.23 / 0.8087 | 27.71 / 0.7407 |
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
| ninasr\_b0          | 0.10           | 31.33 / 0.9204 | 27.48 / 0.8374 | 25.45 / 0.7645 |
| ninasr\_b1          | 1.02           | 32.48 / 0.9319 | 28.29 / 0.8555 | 26.25 / 0.7914 |
| ninasr\_b2          | 10.0           | 32.91 / 0.9354 | 28.70 / 0.8640 | 26.54 / 0.8008 |
| rcan                | 15.4           | 33.19 / 0.9372 | 29.01 / 0.868 | 26.75 / 0.80624 |
| rdn                 | 22.1           | 32.41 / 0.9310 | 27.49 / 0.838 | 26.36 / 0.79460 |

</details>

</details>

All models are defined in `torchsr.models`. Other useful tools to augment your models, such as self-ensemble methods and tiling, are present in `torchsr.models.utils`.



## Datasets

The following datasets are available. Click on the links for the project page:
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* [RealSR](https://github.com/csjcai/RealSR)
* [Flicr2K](https://github.com/limbee/NTIRE2017)
* [REDS](https://seungjunnah.github.io/Datasets/reds)
* [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://paperswithcode.com/dataset/set14), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [Urban100](https://paperswithcode.com/dataset/urban100)

All datasets are defined in `torchsr.datasets`. They return a list of images, with the high-resolution image followed by downscaled or degraded versions.
Data augmentation methods are provided in `torchsr.transforms`.

Datasets are downloaded automatically when using the `download=True` flag, or by running the corresponding script i.e. `./scripts/download_div2k.sh`.



## Usage


```python
from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0
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
<summary>Expand more examples</summary>


```python
from torchsr.datasets import Div2K
from torchsr.models import edsr, rcan
from torchsr.models.utils import ChoppedModel, SelfEnsembleModel
from torchsr.transforms import ColorJitter, Compose, RandomCrop

# Div2K dataset, cropped to 256px, width color jitter
dataset = Div2K(
    root="./data", scale=2, download=False,
    transform=Compose([
        RandomCrop(256, scales=[1, 2]),
        ColorJitter(brightness=0.2)
    ]))

# Pretrained RCAN model, with tiling for large images
model = ChoppedModel(
    rcan(scale=2, pretrained=True), scale=2,
    chop_size=400, chop_overlap=10)

# Pretrained EDSR model, with self-ensemble method for higher quality
model = SelfEnsembleModel(edsr(scale=2, pretrained=True))
```
</details>



## Training

A script is available to train the models from scratch, evaluate them, and much more. It is not part of the pip package, and requires additional dependencies. More examples are available in `scripts/`.

```bash
pip install piq tqdm tensorboard  # Additional dependencies
python -m torchsr.train -h
python -m torchsr.train --arch edsr_baseline --scale 2 --download-pretrained --images test/butterfly.png --destination results/
python -m torchsr.train --arch edsr_baseline --scale 2 --download-pretrained --validation-only
python -m torchsr.train --arch edsr_baseline --scale 2 --epochs 300 --loss l1 --dataset-train div2k_bicubic
```

You can evaluate models from the command line as well. For example, for EDSR with the paper's PSNR evaluation:
```
python -m torchsr.train --validation-only --arch edsr_baseline --scale 2 --dataset-val set5 --chop-size 400 --download-pretrained --shave-border 2 --eval-luminance
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
  howpublished = {\url{https://github.com/Coloquinte/torchSR}},
  doi = {10.5281/zenodo.4868308}
}

@misc{ninasr,
  author = {Gabriel Gouvine},
  title = {NinaSR: Efficient Small and Large ConvNets for Super-Resolution},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Coloquinte/torchSR/blob/main/doc/NinaSR.md}},
  doi = {10.5281/zenodo.4868308}
}
```
