from .common import pil_loader, FolderByDir

import os
from typing import Callable, List, Optional, Tuple, Union


class RealSRv3(FolderByDir):
    """`RealSR v3 <https://github.com/csjcai/RealSR>` Superresolution Dataset

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3 or 4
        track (str, optional): The camera type: canon or nikon.
        split (string, optional): The dataset split, supports ``train`` or ``val``.
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
        ("https://drive.google.com/uc?export=download&id=17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM", '83e946bf4b5c22b30d1663231bb2e53e', 'RealSR(V3).tar.gz')
    ]

 
    track_dirs = {
        ('canon', 'train', 2) : os.path.join('RealSR(V3)', 'Canon', 'Train', '2')
      , ('canon', 'train', 3) : os.path.join('RealSR(V3)', 'Canon', 'Train', '3')
      , ('canon', 'train', 4) : os.path.join('RealSR(V3)', 'Canon', 'Train', '4')
      , ('nikon', 'train', 2) : os.path.join('RealSR(V3)', 'Nikon', 'Train', '2')
      , ('nikon', 'train', 3) : os.path.join('RealSR(V3)', 'Nikon', 'Train', '3')
      , ('nikon', 'train', 4) : os.path.join('RealSR(V3)', 'Nikon', 'Train', '4')
      , ('canon', 'val', 2) : os.path.join('RealSR(V3)', 'Canon', 'Test', '2')
      , ('canon', 'val', 3) : os.path.join('RealSR(V3)', 'Canon', 'Test', '3')
      , ('canon', 'val', 4) : os.path.join('RealSR(V3)', 'Canon', 'Test', '4')
      , ('nikon', 'val', 2) : os.path.join('RealSR(V3)', 'Nikon', 'Test', '2')
      , ('nikon', 'val', 3) : os.path.join('RealSR(V3)', 'Nikon', 'Test', '3')
      , ('nikon', 'val', 4) : os.path.join('RealSR(V3)', 'Nikon', 'Test', '4')
    }

    def __init__(
            self,
            root: str,
            scale: int = 2,
            track: Union[str, List[str]] = 'canon',
            split: str = 'train',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        if scale is None:
            raise ValueError("RealSR dataset does not support getting HR images only")
        super(RealSRv3, self).__init__(os.path.join(root, 'RealSR'),
                                     scale, track, split, transform,
                                     loader, download, predecode, preload)

    def list_samples(self, track, split, scale):
        raise NotImplementedError()

    def list_samples_realsr(self, track, split, scale):
        track_dir = self.get_dir(track, split, scale)
        all_samples = sorted(os.listdir(track_dir))
        all_samples = [s for s in all_samples if s.lower().endswith(self.extensions)]
        all_samples = [os.path.join(track_dir, s) for s in all_samples]
        hr_samples = [s for s in all_samples if os.path.splitext(s)[0].lower().endswith("_hr")]
        lr_samples = [s for s in all_samples if not os.path.splitext(s)[0].lower().endswith("_hr")]
        if len(lr_samples) != len(hr_samples):
            raise ValueError(f"Number of files for {track}X{scale} {split} does not match between HR and LR")
        return hr_samples, lr_samples

    def init_samples(self):
        if len(self.tracks) != 1:
            raise RuntimeError("Only one scale can be used at a time for RealSR dataset")
        hr_samples, lr_samples = self.list_samples_realsr(self.tracks[0], self.split, self.scales[0])
        self.samples = list(zip(hr_samples, lr_samples))


