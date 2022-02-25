import numpy
import torch
from torch.utils.data import Dataset


class TorchDatasetWrapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        ret = {
            "x": torch.tensor(self.x[index].astype(float)).type(torch.FloatTensor),
            "y": torch.tensor(self.y[index]).type(torch.LongTensor),
        }
        return ret

    def __len__(self):
        return numpy.size(self.x, axis=0) if self.x is not None else 0


class AbstractDataSet:
    def __init__(self, subtree):
        self.train = None
        self.test = None
        self.target_train = None
        self.target_test = None
        self.train_effective_length = None
        self.test_effective_length = None
        self.available_classes_count = 0
        self.node_name = ""
        self.future_range = None
        self.sequence_length = None
        self.quantile = None
        self.subtree = subtree

    def features_count(self):
        return numpy.size(self.train, axis=2)

    def sequence_length(self):
        return numpy.size(self.train, axis=1)

    def train_sequences_count(self):
        return numpy.size(self.train, axis=0)

    def test_sequences_count(self):
        return numpy.size(self.test, axis=0) if self.test is not None else 0

    def get_wrapped_train(self):
        return TorchDatasetWrapper(self.train, self.target_train)

    def get_wrapped_test(self):
        return TorchDatasetWrapper(self.test, self.target_test)

    def get_config(self):
        return {
            "quantile": self.quantile,
            "sequence_length": self.sequence_length,
            "future_range": self.future_range
        }
