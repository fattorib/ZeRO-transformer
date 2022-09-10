""" 
Class + helper methods for interfacing with PyTorch dataloaders
Taken/modified from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
"""

import jax.numpy as jnp
import numpy as np
from torch.utils import data


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
