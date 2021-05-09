import os
from typing import Callable, List, Optional, Tuple, Union

from .common import FolderByDir, pil_loader


class REDS(FolderByDir):
    """`REDS <https://seungjunnah.github.io/Datasets/reds>` Superresolution Dataset

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3, 4 or 8.
        track (str, optional): The downscaling method: bicubic, unknown, real_mild,
            real_difficult, real_wild.
        split (string, optional): The dataset split, supports ``train``, ``val`` or 'test'.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    urls = [
    ]

    track_dirs = {
        ('hr', 'train', 1) : os.path.join('train', 'train_sharp')
      , ('blur', 'train', 1) : os.path.join('train', 'train_blur')
      , ('blur_comp', 'train', 1) : os.path.join('train', 'train_blur_comp')
      , ('bicubic', 'train', 4) : os.path.join('train', 'train_sharp_bicubic')
      , ('blur_bicubic', 'train', 4) : os.path.join('train', 'train_blur_bicubic')
      , ('hr', 'val', 1) : os.path.join('val', 'val_sharp')
      , ('blur', 'val', 1) : os.path.join('val', 'val_blur')
      , ('blur_comp', 'val', 1) : os.path.join('val', 'val_blur_comp')
      , ('bicubic', 'val', 4) : os.path.join('val', 'val_sharp_bicubic')
      , ('blur_bicubic', 'val', 4) : os.path.join('val', 'val_blur_bicubic')
      , ('blur', 'test', 1) : os.path.join('test', 'test_blur')
      , ('blur_comp', 'test', 1) : os.path.join('test', 'test_blur_comp')
      , ('bicubic', 'test', 4) : os.path.join('test', 'test_sharp_bicubic')
      , ('blur_bicubic', 'test', 4) : os.path.join('test', 'test_blur_bicubic')
    }

    def __init__(
            self,
            root: str,
            scale: Optional[int] = None,
            track: Union[str, List[str]] = 'bicubic',
            split: str = 'train',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(REDS, self).__init__(os.path.join(root, 'REDS'),
                                    scale, track, split, transform,
                                    loader, download, predecode, preload)
