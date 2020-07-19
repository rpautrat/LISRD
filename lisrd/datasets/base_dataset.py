from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class BaseDataset(metaclass=ABCMeta):
    """ Base dataset class.

    Arguments:
        config: A dictionary containing the configuration parameters.
        device: The device to train/test on.
    """
    required_baseconfig = ['batch_size', 'test_batch_size', 'sizes']

    def __init__(self, config, device):
        self._config = config
        self._device = device
        required = self.required_baseconfig + getattr(
            self, 'required_config_keys', [])
        for r in required:
            assert r in self._config, 'Required configuration entry: \'{}\''.format(r)
        seed = self._config.get('seed', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def get_dataset(self, split):
        """To be implemented by the child class."""
        raise NotImplementedError

    def get_data_loader(self, split):
        """Return a data loader for a given split."""
        assert split in ['train', 'val', 'test']
        batch_size = (self._config['test_batch_size'] if split == 'test'
                      else self._config['batch_size'])
        return DataLoader(self.get_dataset(split), batch_size=batch_size,
                          shuffle=split == 'train')
