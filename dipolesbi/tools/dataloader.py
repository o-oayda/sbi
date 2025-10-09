from collections import namedtuple
import jax
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray
from dipolesbi.tools.np_rngkey import NPKey
from surjectors.util import _DataLoader, named_dataset


healpix_map_dataset_idx = namedtuple("healpix_map_dataset", "y x mask round_id")
healpix_map_dataset = namedtuple("healpix_map_dataset", "y x mask")

# class DataHandler(Dataset):
#     def __init__(self, theta: Tensor, x: Tensor) -> None:
#         super().__init__()
#         self.theta = theta
#         self.x = x
#
#     def __len__(self):
#         return self.theta.shape[0]
#
#     def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
#         return self.theta[index,...], self.x[index,...]

def split_train_val(
        key: NPKey,
        y: NDArray, 
        x: NDArray, 
        validation_fraction: float = 0.1
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    '''
    :return: Tuple[ tuple[y_train, y_val], tuple[x_train, x_val] ].
    '''
    n_batches = y.shape[0]
    assert n_batches == x.shape[0], 'Batch sizes of x and y are inconsistent!'

    indexes = np.arange(n_batches)
    perm = key.permutation(indexes)
    n_validation = int(n_batches * validation_fraction)

    val_idx, train_idx = perm[:n_validation], perm[n_validation:]
    return (y[train_idx], y[val_idx]), (x[train_idx], x[val_idx])

def split_train_val_dict(
        key: NPKey,
        y: NDArray, 
        x: dict[str, NDArray], 
        round_idx: NDArray,
        mask: NDArray[np.bool_],
        validation_fraction: float = 0.1, 
) -> tuple[
        tuple[NDArray, NDArray],
        tuple[dict[str, NDArray], dict[str, NDArray]],
        tuple[NDArray[np.bool_], NDArray[np.bool_]],
        NDArray
    ]:
    '''
    :return: Tuple[ tuple[y_train, x_train], tuple[y_val, x_val] ].
    '''
    n_batches = y.shape[0]

    indexes = jnp.arange(n_batches)
    perm = key.permutation(indexes)
    n_validation = int(n_batches * validation_fraction)

    val_idx, train_idx = perm[:n_validation], perm[n_validation:]

    y_tuple = (y[train_idx], y[val_idx])

    x_tr = {key: val[train_idx] for key, val in x.items()}
    x_val = {key: val[val_idx] for key, val in x.items()}
    x_tuple = (x_tr, x_val)

    training_set_round_idxs = round_idx[train_idx]
    mask_tuple = (mask[train_idx], mask[val_idx])
    return y_tuple, x_tuple, mask_tuple, training_set_round_idxs


def as_batch_iterator_cpu2gpu(
    rng_key: NPKey, 
    data: healpix_map_dataset_idx | healpix_map_dataset, 
    batch_size: int, 
    shuffle=True
):
    """Create a batch iterator for a data set, converting from CPU (numpy)
    to GPU (jax) as required.
    Returns:
        a data loader object
    """
    n = data.y.shape[0]
    if n < batch_size:
        num_batches = 1
        batch_size = n
    elif n % batch_size == 0:
        num_batches = int(n // batch_size)
    else:
        num_batches = int(n // batch_size) + 1

    idxs = jnp.arange(n)
    if shuffle:
        idxs = rng_key.permutation(idxs)

    def get_batch(idx, idxs=idxs):
        start_idx = idx * batch_size
        step_size = int(np.minimum(n - start_idx, batch_size))

        # NumPy slice to get the indices for this batch (CPU)
        ret_idx = idxs[start_idx:start_idx + step_size]

        # NumPy advanced indexing per field (CPU)
        batch = {}
        for name, array in zip(data._fields, data):
            if name == "x" and isinstance(array, dict):
                # Handle the case where 'x' is a dict of arrays (tree)
                batch["x"] = {k: v[ret_idx] for k, v in array.items()}
            else:
                batch[name] = array[ret_idx]

        # Move JUST this batch to GPU
        return jax.device_put(batch)

    return _DataLoader(num_batches, idxs, get_batch)

