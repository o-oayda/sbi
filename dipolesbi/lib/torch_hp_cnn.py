from sbi.neural_nets.embedding_nets import FCEmbedding
from torch import nn
from typing import Optional
import torch
from healpy import nside2npix
from torch.types import Tensor
import healpy as hp
import numpy as np
import torch.nn.functional as F


class hpCNNEmbedding(nn.Module):
    '''
    CNN emedding network for use on the healpix sphere.
    '''
    def __init__(self,
        nside: int,
        in_channels: int = 1,
        out_channels_per_layer: Optional[list[int]] = None,
        nest: bool = True,
        n_blocks: None | int = None,
        num_fc_units: int = 48,
        num_fc_layers: int = 2,
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

        if n_blocks == None:
            self.n_blocks = torch.log2(torch.as_tensor(nside))
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

        npix = nside2npix(nside)
        self.input_shape = (in_channels, npix) # input map is always 1d?
        cur_nside = nside
        cnn_layers = []
        current_in_channels = in_channels
        current_out_channels = current_in_channels # for no blocks/linter

        print(f"hpCNNEmbedding configuration:")
        print(f"  Input nside: {nside}")
        print(f"  Output nside {int( nside / (2**self.n_blocks)) }")
        print(f"  in_channels: {in_channels}")
        print(f"  out_channels_per_layer: {out_channels_per_layer}")
        print(f"  n_blocks: {self.n_blocks}")
        print(f"  dropout_rate: {dropout_rate}")
        print(f"  nest: {nest}")
        print(f"  num_fc_units: {num_fc_units}")
        print(f"  num_fc_layers: {num_fc_layers}")
        print(f"  output_dim: {output_dim}")

        for i in range(int(self.n_blocks)):
            current_out_channels = out_channels_per_layer[i]
                
            conv_layer = sphericalConv(
                NSIDE=cur_nside,
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                nest=nest
            )
            pool = sphericalDown(cur_nside)
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
            num_layers=num_fc_layers,
            num_hiddens=num_fc_units,
        )
    
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


class sphericalConv(nn.Module):
    def __init__(self, NSIDE, in_channels, out_channels, bias=True, nest=True):
        """Convolutional layer as defined in Krachmalnicoff & Tomasi (A&A, 2019, 628, A129)

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        in_channels : int
            Number of channels of the input. The size is [B,C_in,N], with B batches, 
            C_in channels and N pixels in the HEALPix pixelization
        out_channels : int
            Number of channels of the output. The size is [B,C_out,N], with B batches, 
            C_out channels and N pixels in the HEALPix pixelization
        bias : bool, optional
            Add bias, by default True
        nest : bool, optional
            Used nested mapping, by default True
            Always use nested mapping if pooling layers are used.
        """
        super(sphericalConv, self).__init__()

        self.NSIDE = NSIDE
        self.npix = hp.nside2npix(self.NSIDE)
        self.nest = nest

        self.neighbours = torch.zeros(9 * self.npix, dtype=torch.long)
        self.weight = torch.ones(9 * self.npix, dtype=torch.float32)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=9, bias=bias)

        for i in range(self.npix):
            neighbours = hp.pixelfunc.get_all_neighbours(self.NSIDE, i, nest=nest)
            neighbours = np.insert(neighbours, 4, i)

            ind = np.where(neighbours == -1)[0]
            neighbours[ind] = self.npix            

            self.neighbours[9*i:9*i+9] = torch.tensor(neighbours)

        self.zeros = torch.zeros((1, 1, 1))

        nn.init.kaiming_normal_(self.conv.weight)        
        if (bias):
            nn.init.constant_(self.conv.bias, 0.0)
        
    def forward(self, x):

        x2 = F.pad(x, (0,1,0,0,0,0), mode='constant', value=0.0)
                
        vec = x2[:, :, self.neighbours]
        
        tmp = self.conv(vec)

        return tmp

class sphericalDown(nn.Module):    
    def __init__(self, NSIDE):
        """Average pooling layer

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        """
        super(sphericalDown, self).__init__()
        
        self.pool = nn.AvgPool1d(4)
                
    def forward(self, x):
                
        return self.pool(x)

class sphericalUp(nn.Module):
    def __init__(self, NSIDE):
        """Upsampling pooling layer

        Parameters
        ----------
        NSIDE : int
            HEALPix NSIDE
        """
        super(sphericalUp, self).__init__()
                
    def forward(self, x):
        
        return torch.repeat_interleave(x, 4, dim=-1)
