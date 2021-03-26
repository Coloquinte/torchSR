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

# Download a pretrained EDSR model
model = edsr(scale=2, pretrained=True)

# Run the Super-Resolution model
lr_t = to_tensor(lr).unsqueeze(0)
sr_t = model(lr_t)
sr = to_pil_image(sr_t.squeeze(0))
sr.show()
```



## Datasets

Datasets return a list of images. The first image is the original one, and the next images are downscaled or degraded versions.

The following datasets are available, and can be downloaded automatically:
* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
* [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)
* [Set14](https://paperswithcode.com/dataset/set14)
* [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
* [Urban100](https://paperswithcode.com/dataset/urban100)



## Transforms

Transforms are used for preprocessing and data augmentation. They are applied identically to the original and downscaled image.

This repository defines several transforms that follow torchvision conventions:
* ToTensor/ToPILImage
* Compose
* RandomCrop
* ColorJitter
* GaussianBlur
* RandomHorizontalFlip/RandomVerticalFlip



## Models

The following models are available:
* VDSR
* EDSR (pretrained x2 x3 x4)
* RDN
* RCAN (pretrained x2 x3 x4 x8)



