import os

import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co


class FolderIterableDataset(IterableDataset):

    def __init__(self, directory: str, tokenize_fn: callable):
        self.dir = directory

        self.length = len([_ for _ in os.listdir(self.dir)])

        self.tokenize_fn = tokenize_fn

    def __iter__(self):
        for filename in os.listdir(self.dir):
            data = torch.load(os.path.join(self.dir, filename))

            yield data[0], self.tokenize_fn(data[1])

    def __getitem__(self, index) -> T_co:
        raise NotImplemented

    def __len__(self):
        return self.length
