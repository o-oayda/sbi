import haiku as hk
from typing import Optional
from jax import numpy as jnp
from dipolesbi.tools.healsphere_conv import HealpixConv
import healpy as hp
import numpy as np


class hpCNNEmbedding(hk.Module):
    '''
    CNN emedding network for use on the healpix sphere.
    '''
    def __init__(self,
        nside: int,
        in_channels: int = 1,
        out_channels_per_layer: Optional[list[int]] = None,
        nest: bool = True,
        n_blocks: None | int = None,
        n_mlp_neurons: int = 48,
        n_mlp_layers: int = 2,
        output_dim: int = 20,
        dropout_rate: Optional[float] = None
    ) -> None:
        '''
        :param nside: Nside of input healpy map.
        :param in_channels: Number of input channels.
        :param out_channels_per_layer: Number of output channels for each layer.
            If None, defaults to [8, 16, 32, 64, 128, 256].
        :param n_blocks: Number of network building blocks to use, as defined
            in Krachmalnicoff & Tomasi (2019). A minimum of 1 block needs to be
            specified, up to a maximum of log2(nside), representing a maximally
            deep network (default).
        :param nest: Whether or not the healpy map uses nested ordering.
            This always needs to be true it seems (required for pooling).
        :param dropout_rate: Dropout rate to apply after CNN layers, before FC layers.
            If None, no dropout is applied.
        '''
        super().__init__()

        self.nside = nside
        self.in_channels = in_channels
        self.out_channels_per_layer = self.out_channels_per_layer
        self.nest = True
        self.n_blocks = n_blocks
        self.n_mlp_neurons = n_mlp_neurons
        self.n_mlp_layers = n_mlp_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self._neighbour_graphs = self._precompute_healpix_neighbours() # for jax support

        if n_blocks == None:
            self.n_blocks = jnp.log2(nside)
        else:
            self.n_blocks = n_blocks

        if out_channels_per_layer is None:
            out_channels_per_layer = [8, 16, 32, 64, 128, 256]
        
        # Ensure we don't have more layers than blocks
        max_blocks = int(self.n_blocks)
        out_channels_per_layer = out_channels_per_layer[:max_blocks]

        # repeat last element of out_channels_per_layer
        if len(out_channels_per_layer) < max_blocks:
            out_channels_per_layer += (
                  [out_channels_per_layer[-1]]
                * (max_blocks - len(out_channels_per_layer))
            )

        npix = 12 * nside * nside
        self.input_shape = (in_channels, npix)
        cur_nside = nside
        cnn_layers = []
        current_in_channels = in_channels
        current_out_channels = current_in_channels # for no blocks/linter


        for i in range(int(self.n_blocks)):
            current_out_channels = out_channels_per_layer[i]
                
            conv_layer = HealpixConv(
                nside=cur_nside,
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                nest=nest
            )
            pool = sp.sphericalDown(cur_nside)
            cnn_layers += [conv_layer, nn.ReLU(inplace=True), pool]
            cur_nside //= 2
            cnn_output_size = int(nside2npix(cur_nside))
            
            # Update input channels for next layer
            current_in_channels = current_out_channels
        
        self.cnn_subnet = nn.Sequential(*cnn_layers)

        # Add dropout layer if specified
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        # Construct linear post processing net
        self.linear_subnet = FCEmbedding(
            input_dim=current_out_channels * cnn_output_size,
            output_dim=output_dim,
            num_layers=n_mlp_layer,
            num_hiddens=n_mlp_neurons,
        )

    def __repr__(self) -> str:
        assert self.n_blocks is not None
        return (
            f"hpCNNEmbedding configuration("
            f"Input nside: {self.nside}, "
            f"Output nside {int( self.nside / (2**self.n_blocks)) }, "
            f"in_channels: {self.in_channels}, "
            f"out_channels_per_layer: {self.out_channels_per_layer}, "
            f"n_blocks: {self.n_blocks}, "
            f"dropout_rate: {self.dropout_rate}, "
            f"nest: {self.nest}, "
            f"n_mlp_neurons: {self.n_mlp_neurons}, "
            f"n_mlp_layers: {self.n_mlp_layers}, "
            f"output_dim: {self.output_dim}"
        )
        return super().__repr__()
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)

        # reshape to account for single channel data
        x = self.cnn_subnet(x.view(batch_size, *self.input_shape))

        # flatten for linear layers
        x = x.view(batch_size, -1)
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        x = self.linear_subnet(x)
        
        return x
