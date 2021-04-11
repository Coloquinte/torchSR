# Super-Resolution datasets for Pytorch

This repository implements datasets and transforms to make Super-Resolution development easy.
It is heavily inspired by [torchvision](https://github.com/pytorch/vision) and [EDSR](https://github.com/zhouhuanxiang/EDSR-PyTorch).



## Usage

```python
from datasets import Div2K
from models import edsr
from transforms import Compose, RandomCrop, ColorJitter
from torchvision.transforms.functional import to_pil_image, to_tensor

# Div2K dataset, cropped to 256px with brightness jitter
dataset = Div2K(root="./data", scale=2, download=False,
                transform=Compose([
		    RandomCrop(256, scales=[1, 2]),
		    ColorJitter(brightness=0.2)
		]))
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



## Models

The following pretrained models are available:
* [EDSR](https://arxiv.org/abs/1707.02921) (x2 x3 x4)
* [RDN](https://arxiv.org/abs/1802.08797) (x2 x3 x4)
* [RCAN](https://arxiv.org/abs/1807.02758) (x2 x3 x4 x8)
* NinaSR, my own model for real-time super resolution (x2 x3 x4 x8)

The following models are implemented without pretrained weights:
* [VDSR](https://arxiv.org/abs/1511.04587)



## Datasets

Datasets return a list of images. The first image is the original one, and the next images are downscaled or degraded versions.

The following datasets are available:
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* [RealSR](https://github.com/csjcai/RealSR)
* [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](https://paperswithcode.com/dataset/set14), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), [Urban100](https://paperswithcode.com/dataset/urban100)

They are downloaded and extracted automatically when using the `download=True` flag. For a faster download, consider using the script first:
```bash
./scripts/download_div2k.sh
```



## Transforms

Transforms are used for preprocessing and data augmentation. They are applied identically to the original and downscaled image.

This repository defines several transforms that follow torchvision conventions:
* ToTensor/ToPILImage
* Compose
* RandomCrop
* ColorJitter
* GaussianBlur
* RandomHorizontalFlip/RandomVerticalFlip/RandomFlipTurn



## Training

A script is available to train the models from scratch, evaluate them, and much more:
```bash
pip install pip tqdm tensorboard  # Additional dependencies
python main.py -h
python main.py --arch edsr_baseline --download-pretrained --evaluate
python main.py --arch edsr_baseline --epochs 300 --loss l1 --dataset-train div2k_bicubic
python main.py --arch edsr_baseline --evaluate --download-pretrained
```



# Contributions

All contributions are welcome! Usability improvements, training improvements, new models, new transforms, ...
Don't hesitate to contribute your own pretrained models too.
