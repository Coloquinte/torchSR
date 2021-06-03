import os
from typing import Callable, List, Optional, Tuple, Union

from .common import FolderByDir, pil_loader


class Flickr2K(FolderByDir):
    """`Flickr2K <https://github.com/limbee/NTIRE2017>` Superresolution Dataset

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
        track (str, optional): The downscaling method: bicubic, unknown.
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
        ("https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar", "5d3f39443d5e9489bff8963f8f26cb03")
    ]


    track_dirs = {
        ('hr', 'train', 1) : os.path.join('Flickr2K', 'Flickr2K_HR')
      , ('bicubic', 'train', 2) : os.path.join('Flickr2K', 'Flickr2K_LR_bicubic', 'X2')
      , ('bicubic', 'train', 3) : os.path.join('Flickr2K', 'Flickr2K_LR_bicubic', 'X3')
      , ('bicubic', 'train', 4) : os.path.join('Flickr2K', 'Flickr2K_LR_bicubic', 'X4')
      , ('unknown', 'train', 2) : os.path.join('Flickr2K', 'Flickr2K_LR_unknown', 'X2')
      , ('unknown', 'train', 3) : os.path.join('Flickr2K', 'Flickr2K_LR_unknown', 'X3')
      , ('unknown', 'train', 4) : os.path.join('Flickr2K', 'Flickr2K_LR_unknown', 'X4')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            track: Union[str, List[str]] = 'bicubic',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(Flickr2K, self).__init__(os.path.join(root, 'Flickr2K'),
                                    scale, track, 'train', transform,
                                    loader, download, predecode, preload)
