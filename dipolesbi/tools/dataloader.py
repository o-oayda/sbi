from typing import Callable
from jax.random import PRNGKey, permutation
from torch.utils.data import Dataset
from torch import Tensor
from jax import numpy as jnp
from dataclasses import dataclass, fields


class DataHandler(Dataset):
    def __init__(self, theta: Tensor, x: Tensor) -> None:
        super().__init__()
        self.theta = theta
        self.x = x

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.theta[index,...], self.x[index,...]

def split_train_val(
        y: jnp.ndarray, 
        x: jnp.ndarray, 
        validation_fraction: float = 0.1, 
        key=PRNGKey(0)
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    '''
    :return: Tuple[ tuple[y_train, x_train], tuple[y_val, x_val] ].
    '''
    n_batches = y.shape[0]
    assert n_batches == x.shape[0], 'Batch sizes of x and y are inconsistent!'

    indexes = jnp.arange(n_batches)
    perm = permutation(key, indexes)
    n_validation = int(n_batches * validation_fraction)

    val_idx, train_idx = perm[:n_validation], perm[n_validation:]
    return (y[train_idx], y[val_idx]), (x[train_idx], x[val_idx])
