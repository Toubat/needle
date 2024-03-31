import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        import pickle
        if train:
            data_batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            data_batch_files = ['test_batch']
        X = []
        Y = []
        for data_batch_file in data_batch_files:
            with open(os.path.join(base_folder, data_batch_file), 'rb') as f:
                data_dict = pickle.load(f, encoding = 'bytes')
                X.append(data_dict[b'data'])
                Y.append(data_dict[b'labels'])
        X = np.concatenate(X, axis = 0)
        # preprocessing X.
        X = X / 255.
        X = X.reshape((-1, 3, 32, 32))
        Y = np.concatenate(Y, axis = None) # Y is just 1-dimensional.
        self.X = X
        self.Y = Y
        self.transforms=transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            image = np.array([self.apply_transforms(img) for img in self.X[index]])
        else:
            image = self.X[index]
        label = self.Y[index]
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.Y)
        ### END YOUR SOLUTION