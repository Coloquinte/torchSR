from PIL import Image

import os

import torch
import torch.utils.data as data
import torchvision
from typing import Any, Callable, List, Optional, Tuple, Union


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Folder(data.Dataset):
    def __init__(
            self,
            root: str,
            scales: List[Union[int, float]],
            transforms: Optional[Callable] = None,
            loader = pil_loader
    ) -> None:
        super(Folder, self).__init__()
        self.root = os.path.expanduser(root)
        self.loader = loader
        self.transforms = transforms
        self.scales = scales
        self.samples = []

    def __getitem__(self, index: int) -> List[Any]:
        return [self.loader(path) for path in self.samples[index]]
        

    def __len__(self) -> int:
        return len(self.samples)


class Div2K(Folder):
    """`DIV2K <https://data.vision.ee.ethz.ch/cvl/DIV2K/>` Superresolution Dataset

    Args:
        root (string): Root directory of the DIV2K Dataset.
        scale (int, optional): The upsampling ratio: 2, 3, 4 or 8.
        track (str, optional): The downscaling method: bicubic, unknown, realistic_mild,
            realistic_difficult, realistic_wild.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.


    Attributes:
        scales (list): List of the downsampling scales
    """

    urls = [
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X2.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X2.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X3.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X3.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_unknown_X4.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_unknown_X4.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_x8.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_mild.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_difficult.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip"
      , "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_wild.zip"
       ]


    def __init__(
            self,
            root: str,
            scale: Union[int, List[int]] = 2,
            track: Union[str, List[str]] = 'bicubic',
            split: str = 'train',
            transforms: Optional[Callable] = None,
            loader = pil_loader,
            download: bool = False):
        if isinstance(scale, int):
            scale = [scale]
        if isinstance(track, str):
            track = [track] * len(scale)
        if len(track) != len(scale):
            raise ValueError("The number of scales and of tracks must be the same")
        self.split = split
        self.tracks = track
        super(Div2K, self).__init__(os.path.join(root, 'DIV2K'), scale, transforms, loader)
        if download:
            self.download()
        self.init_samples()

    def get_dir(self, track, split, scale):
        track_dirs = {
            ('hr', 'train', 1) : os.path.join('DIV2K_train_HR')
          , ('hr', 'val', 1) : os.path.join('DIV2K_valid_HR')
          , ('bicubic', 'train', 2) : os.path.join('DIV2K_train_LR_bicubic', 'X2')
          , ('bicubic', 'train', 3) : os.path.join('DIV2K_train_LR_bicubic', 'X3')
          , ('bicubic', 'train', 4) : os.path.join('DIV2K_train_LR_bicubic', 'X4')
          , ('bicubic', 'train', 8) : os.path.join('DIV2K_train_LR_X8')
          , ('bicubic', 'val', 2) : os.path.join('DIV2K_valid_LR_bicubic', 'X2')
          , ('bicubic', 'val', 3) : os.path.join('DIV2K_valid_LR_bicubic', 'X3')
          , ('bicubic', 'val', 4) : os.path.join('DIV2K_valid_LR_bicubic', 'X4')
          , ('bicubic', 'val', 8) : os.path.join('DIV2K_valid_LR_X8')
          , ('unknown', 'train', 2) : os.path.join('DIV2K_train_LR_unknown', 'X2')
          , ('unknown', 'train', 3) : os.path.join('DIV2K_train_LR_unknown', 'X3')
          , ('unknown', 'train', 4) : os.path.join('DIV2K_train_LR_unknown', 'X4')
          , ('unknown', 'val', 2) : os.path.join('DIV2K_valid_LR_unknown', 'X2')
          , ('unknown', 'val', 3) : os.path.join('DIV2K_valid_LR_unknown', 'X3')
          , ('unknown', 'val', 4) : os.path.join('DIV2K_valid_LR_unknown', 'X4')
        }

        if (track, split, scale) not in track_dirs:
            raise ValueError(f"Track {track}X{scale} is not part of DIV2K")
        return os.path.join(self.root, track_dirs[(track, split, scale)])

    def list_samples(self, track, split, scale):
        track_dir = self.get_dir(track, split, scale)
        all_samples = sorted(os.listdir(track_dir))
        return [os.path.join(track_dir, s) for s in all_samples]

    def init_samples(self):
        samples = []
        for track, scale in zip(['hr'] + self.tracks, [1] + self.scales):
            samples.append(self.list_samples(track, self.split, scale))
        for i, s in enumerate(samples[1:]):
            if len(s) != len(samples[0]):
                raise ValueError(f"Number of files for {self.tracks[i]}X{self.scales[i]} does not match HR")
        self.samples = []
        for i in range(len(samples[0])):
            self.samples.append([s[i] for s in samples])

    def download(self):
        for url in self.urls:
            torchvision.datasets.utils.download_and_extract_archive(url, self.root)

