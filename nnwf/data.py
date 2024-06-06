import gzip
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import numpy as np

from .autograd import Tensor


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img
        


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        
        img_pad = np.pad(img, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)], 'constant')
        H, W, _ = img_pad.shape
        return img_pad[self.padding + shift_x: H - self.padding + shift_x, self.padding + shift_y: W - self.padding + shift_y, :]
        


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration

        end_idx = min(self.idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[self.idx:end_idx]
        batch = [self.dataset[i] for i in batch_indices]
        batch_images = np.stack([x[0] for x in batch])
        batch_labels = np.array([x[1] for x in batch])
        self.idx += self.batch_size

        return Tensor(batch_images), Tensor(batch_labels)


class MNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        # Read and store the images and labels
        with gzip.open(image_filename, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols, 1).astype(np.float32)
            self.images /= 255.0  # Normalize images to [0, 1]

        with gzip.open(label_filename, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Apply transformations if any
        img = self.images[index]
        if self.transforms:
            img = self.apply_transforms(img)
        return img, self.labels[index]


class NDArrayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], None  # Return None or a placeholder if no labels

import numpy as np

class IrisDataset(Dataset):
    def __init__(self, file_path: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        # Load the dataset
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.features = data[:, :-1]
        self.labels = data[:, -1].astype(np.int32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        # Apply transformations if any
        feature = self.features[index]
        if self.transforms:
            feature = self.apply_transforms(feature)
        return feature, self.labels[index]
    
import gzip
import struct

class FashionMNISTDataset(Dataset):
    def __init__(self, image_filename: str, label_filename: str, transforms: Optional[List] = None):
        super().__init__(transforms)
        # Read and store the images and labels
        with gzip.open(image_filename, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols, 1).astype(np.float32)
            self.images /= 255.0  # Normalize images to [0, 1]

        with gzip.open(label_filename, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Apply transformations if any
        img = self.images[index]
        if self.transforms:
            img = self.apply_transforms(img)
        return img, self.labels[index]
