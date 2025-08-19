from typing import Literal, Optional
from healpy import nside2npix
from torch import nn
import torch
from torch.utils.data.dataloader import DataLoader
from dipolesbi.scripts.integer_flow_test import RoundStraightThrough, log_integer_probability
from torch import Tensor
import matplotlib.pyplot as plt
from dipolesbi.tools.maps import SimpleDipoleMap
from dipolesbi.tools.priors import DipolePrior
from dipolesbi.tools.simulator import Simulator
from dipolesbi.tools.transforms import MapDataset


class IntegerDiscreteFlow(nn.Module):
    def __init__(
            self,
            n_flows: int = 8,
            data_dimensionality: int = 2,
            n_neurons: int = 256
    ) -> None:
        super().__init__()

        coupling_network = [
            lambda: nn.Sequential(
                nn.Linear(data_dimensionality // 2, n_neurons), nn.LeakyReLU(),
                nn.Linear(n_neurons, n_neurons),
                nn.LeakyReLU(),
                nn.Linear(n_neurons, data_dimensionality // 2)
            )
        ]
        self.t = nn.ModuleList([coupling_network[0]() for _ in range(n_flows)])
        self.n_nets = 1

        self.n_flows = n_flows
        self.round = RoundStraightThrough.apply
        
        self.mean = nn.Parameter(torch.zeros(1, data_dimensionality))
        self.log_scale = nn.Parameter(torch.ones(1, data_dimensionality))

        self.data_dimensionality = data_dimensionality

    def coupling(self, x: Tensor, index: int, forward: bool = True) -> Tensor:
        x_a, x_b = torch.chunk(x, 2, 1) # possibly remove last arg and go along dim 0

        if forward:
            y_b = x_b + self.round(self.t[index](x_a)) # type: ignore
        else:
            y_b = x_b - self.round(self.t[index](x_a)) # type: ignore

        return torch.cat((x_a, y_b), dim=1)

    def permute(self, x: Tensor) -> Tensor:
        return x.flip(1)

    def f(self, x: Tensor) -> Tensor:
        z = x
        for i in range(self.n_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)
        return z

    def f_inv(self, z: Tensor) -> Tensor:
        x = z
        for i in reversed(range(self.n_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)
        return x

    def forward(self, x: Tensor, reduction: Literal['avg', 'sum'] = 'avg'):
        z = self.f(x)
        if reduction == 'sum':
            return -self.log_prior(z).sum()
        elif reduction == 'avg':
            return -self.log_prior(z).mean()
        else:
            raise Exception('Kwarg not recognised.')

    def log_prior(self, x: Tensor) -> Tensor:
        log_p = log_integer_probability(x, self.mean, self.log_scale)
        return log_p.sum(1)

    def prior_sample(self, batch_size: int) -> Tensor:
        # Sample from logistic
        y = torch.rand(batch_size, self.data_dimensionality)
        x = torch.exp(self.log_scale) * torch.log(y / (1. - y)) + self.mean
        # And then round it to an integer.
        return torch.round(x)

    def sample(self, batch_size: int) -> Tensor:
        # sample z:
        z = self.prior_sample(batch_size=batch_size)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batch_size, 1, self.data_dimensionality)

def evaluation(
        test_loader: DataLoader, 
        model: IntegerDiscreteFlow, 
        epoch: Optional[int] = None
) -> Tensor:
    model.eval()
    loss = torch.zeros(1)
    N = 0.

    for _, test_batch in enumerate(test_loader):
        loss_t = model.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss

def train_network(
        n_epochs: int,
        model: IntegerDiscreteFlow,
        training_loader: DataLoader,
        validation_loader: DataLoader,
        learning_rate: float = 0.001
) -> list[Tensor]:
    optimizer = torch.optim.Adamax(
        [p for p in model.parameters() if p.requires_grad == True],
        lr=learning_rate
    )

    nll = []
    for i in range(n_epochs):
        model.train()

        for _, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        validation_loss = evaluation(validation_loader, model, i)
        nll.append(validation_loss)

    return nll

if __name__ == '__main__':
    MEAN_DENSITY = 10_000
    NSIDE = 16
    D = nside2npix(NSIDE)

    mean_count_range = [0.95*MEAN_DENSITY, 1.05*MEAN_DENSITY]
    prior = DipolePrior(mean_count_range=mean_count_range)
    prior.change_kwarg('N', 'mean_density')

    dipole = SimpleDipoleMap(nside=NSIDE)
    simulator = Simulator(prior, dipole.generate_dipole)
    theta, x = simulator.make_batch_simulations(
        n_simulations=1000,
        n_workers=32,
        simulation_batch_size=100
    )

    x_data = MapDataset(x[:800])
    x_validation = MapDataset(x[800:])
    train_loader = DataLoader(x_data, batch_size=20, shuffle=True)
    validation_loader = DataLoader(x_validation, batch_size=20, shuffle=False)

    model = IntegerDiscreteFlow(data_dimensionality=D)
    losses = train_network(
        n_epochs=50,
        model=model, 
        training_loader=train_loader, 
        validation_loader=validation_loader
    )

    plt.plot(losses)
    plt.show()
