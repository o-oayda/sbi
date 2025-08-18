from torch.utils.data import Dataset
from torch import Tensor


class DataHandler(Dataset):
    def __init__(self, theta: Tensor, x: Tensor) -> None:
        super().__init__()
        self.theta = theta
        self.x = x

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.theta[index,...], self.x[index,...]
