import os
from typing import Callable, List, Optional, Tuple, Union

from .common import FolderByDir, pil_loader


class Set5(FolderByDir):
    """`Set5 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
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
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    track_dirs = {
        ('hr', 'val', 1) : os.path.join('benchmark', 'Set5', 'HR')
      , ('bicubic', 'val', 2) : os.path.join('benchmark', 'Set5', 'LR_bicubic', 'X2')
      , ('bicubic', 'val', 3) : os.path.join('benchmark', 'Set5', 'LR_bicubic', 'X3')
      , ('bicubic', 'val', 4) : os.path.join('benchmark', 'Set5', 'LR_bicubic', 'X4')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            split: str = 'val',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(Set5, self).__init__(os.path.join(root, 'SRBenchmarks'),
                                   scale, 'bicubic', split, transform,
                                   loader, download, predecode, preload)


class Set14(FolderByDir):
    """`Set14 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
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
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    track_dirs = {
        ('hr', 'val', 1) : os.path.join('benchmark', 'Set14', 'HR')
      , ('bicubic', 'val', 2) : os.path.join('benchmark', 'Set14', 'LR_bicubic', 'X2')
      , ('bicubic', 'val', 3) : os.path.join('benchmark', 'Set14', 'LR_bicubic', 'X3')
      , ('bicubic', 'val', 4) : os.path.join('benchmark', 'Set14', 'LR_bicubic', 'X4')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            split: str = 'val',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(Set14, self).__init__(os.path.join(root, 'SRBenchmarks'),
                                    scale, 'bicubic', split, transform,
                                    loader, download, predecode, preload)


class B100(FolderByDir):
    """`B100 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
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
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    track_dirs = {
        ('hr', 'val', 1) : os.path.join('benchmark', 'B100', 'HR')
      , ('bicubic', 'val', 2) : os.path.join('benchmark', 'B100', 'LR_bicubic', 'X2')
      , ('bicubic', 'val', 3) : os.path.join('benchmark', 'B100', 'LR_bicubic', 'X3')
      , ('bicubic', 'val', 4) : os.path.join('benchmark', 'B100', 'LR_bicubic', 'X4')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            split: str = 'val',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(B100, self).__init__(os.path.join(root, 'SRBenchmarks'),
                                   scale, 'bicubic', split, transform,
                                   loader, download, predecode, preload)

class Urban100(FolderByDir):
    """`Urban100 Superresolution Dataset, linked to by `EDSR <https://github.com/zhouhuanxiang/EDSR-PyTorch>`

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4.
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
        ("https://cv.snu.ac.kr/research/EDSR/benchmark.tar", "4ace41d33c2384b97e6b320cd0afd6ba")
    ]
    
    track_dirs = {
        ('hr', 'val', 1) : os.path.join('benchmark', 'Urban100', 'HR')
      , ('bicubic', 'val', 2) : os.path.join('benchmark', 'Urban100', 'LR_bicubic', 'X2')
      , ('bicubic', 'val', 3) : os.path.join('benchmark', 'Urban100', 'LR_bicubic', 'X3')
      , ('bicubic', 'val', 4) : os.path.join('benchmark', 'Urban100', 'LR_bicubic', 'X4')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            split: str = 'val',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(Urban100, self).__init__(os.path.join(root, 'SRBenchmarks'),
                                       scale, 'bicubic', split, transform,
                                       loader, download, predecode, preload)

