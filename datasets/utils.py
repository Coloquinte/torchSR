from PIL import Image

import os

import numpy as np
import torch
import torchvision

from typing import Any, Callable, List, Optional, Tuple, Union


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Folder(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            scales: List[Union[int, float]],
            transform: Optional[Callable] = None,
            loader = pil_loader,
            predecode: bool = False,
            preload: bool = False
    ) -> None:
        super(Folder, self).__init__()
        self.root = os.path.expanduser(root)
        self.loader = loader
        self.transform = transform
        self.scales = scales
        self.predecode = predecode
        self.preload = preload
        self.samples = []
        self.cache = {}

    def __getitem__(self, index: int) -> List[Any]:
        images = []
        for path in self.samples[index]:
            if self.preload:
                if path in self.cache:
                    img = self.cache[path]
                else:
                    img = self.loader(path)
                    self.cache[path] = img
            else:
                if self.predecode:
                    img = self.get_or_create_predecode(path)
                else:
                    img = self.loader(path)
            images.append(img)
        if self.transform is not None:
            images = self.transform(images)
        return images

    def get_or_create_predecode(self, path):
        prepath = os.path.splitext(path)[0] + '.npy'
        try:
            arr = np.load(prepath, mmap_mode='r', allow_pickle=False)
        except IOError:
            img = self.loader(path)
            arr = np.array(img)
            np.save(prepath, arr)
        return arr

    def __len__(self) -> int:
        return len(self.samples)


class FolderByDir(Folder):
    urls = {}
    track_dirs = {}
    extensions = ( '.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp' )
    already_downloaded_urls = []

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int]],
            track: Union[str, List[str]],
            split: str = 'train',
            transform: Optional[Callable] = None,
            loader = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        if isinstance(scale, int):
            scale = [scale]
        if isinstance(track, str):
            track = [track] * len(scale)
        if len(track) != len(scale):
            raise ValueError("The number of scales and of tracks must be the same")
        self.split = split
        self.tracks = track
        super(FolderByDir, self).__init__(root, scale, transform, loader,
                                          predecode, preload)
        if download:
            self.download()
        self.init_samples()

    @classmethod
    def get_tracks(cls):
        return set(t for (t, sp, sc) in cls.track_dirs.keys())

    @classmethod
    def get_splits(cls):
        return set(sp for (t, sp, sc) in cls.track_dirs.keys())

    @classmethod
    def has_split(self, split):
        return split in cls.get_splits()

    def get_dir(self, track, split, scale):
        if (track, split, scale) not in self.track_dirs:
            if track not in self.get_tracks():
                raise ValueError(f"{self.__class__.__name__} does not include track {track}. "
                                 f"Use one of {list(self.get_tracks())}")
            if split not in self.get_splits():
                raise ValueError(f"{self.__class__.__name__} does not include split {split}. "
                                 f"Use one of {list(self.get_splits())}")
            raise ValueError(f"{self.__class__.__name__} track {track} does not include scale X{scale}")
        return os.path.join(self.root, self.track_dirs[(track, split, scale)])

    def list_samples(self, track, split, scale):
        track_dir = self.get_dir(track, split, scale)
        all_samples = sorted(os.listdir(track_dir))
        all_samples = [s for s in all_samples if s.lower().endswith(self.extensions)]
        return [os.path.join(track_dir, s) for s in all_samples]

    def init_samples(self):
        if ('hr', self.split, 1) in self.track_dirs:
            # Typical case: HR data is available
            all_tracks = zip(['hr'] + self.tracks, [1] + self.scales)
        else:
            # Testing set: no HR data
            all_tracks = zip(self.tracks, self.scales)
        samples = []
        for track, scale in all_tracks:
            samples.append(self.list_samples(track, self.split, scale))
        for i, s in enumerate(samples[1:]):
            if len(s) != len(samples[0]):
                raise ValueError(f"Number of files for {self.tracks[i]}X{self.scales[i]} {self.split} does not match HR")
        self.samples = []
        for i in range(len(samples[0])):
            self.samples.append([s[i] for s in samples])

    def download(self):
        # We just always download everything: the X4/X8 datasets are not big anyway
        for data in self.urls:
            filename = None
            md5sum = None
            if isinstance(data, str):
                url = data
            else:
                url = data[0]
                if len(data) > 1:
                    md5sum = data[1]
                if len(data) > 2:
                    filename = data[2]
            if (self.root, url) in self.already_downloaded_urls:
                continue
            torchvision.datasets.utils.download_and_extract_archive(url, self.root, filename=filename, md5=md5sum)
            self.already_downloaded_urls.append((self.root, url))


