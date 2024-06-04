"""Temporal Convolution model implemented in PyTorch."""

import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import copy
import torch.nn.functional as F
from daart.models.base import BaseModel, get_activation_func_from_str

# to ignore imports for sphix-autoapidoc
__all__ = ['MultiStageModel']


class DilatedTCN(BaseModel):
    """Temporal Convolutional Model with dilated convolutions and no temporal downsampling.

    Code adapted from: https://www.kaggle.com/ceshine/pytorch-temporal-convolutional-networks
    """

    def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):
        super().__init__()
        self.hparams = hparams
        self.model = nn.Sequential()
        if type == 'encoder':
            in_size_ = hparams['input_size'] if in_size is None else in_size
            hid_size_ = hparams['n_hid_units'] if hid_size is None else hid_size
            out_size_ = hparams['n_hid_units'] if out_size is None else out_size
            self.build_encoder(in_size=in_size_, hid_size=hid_size_, out_size=out_size_)
        else:
            in_size_ = hparams['n_hid_units'] if in_size is None else in_size
            hid_size_ = hparams['n_hid_units'] if hid_size is None else hid_size
            out_size_ = hparams['input_size'] if out_size is None else out_size
            self.build_decoder(in_size=in_size_, hid_size=hid_size_, out_size=out_size_)

    def build_encoder(self, in_size, hid_size, out_size):
        """Construct encoder model using hparams."""

        global_layer_num = 0

        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** i_layer
            in_size_ = in_size if i_layer == 0 else hid_size
            hid_size_ = hid_size
            if i_layer == (self.hparams['n_hid_layers'] - 1):
                # final layer
                out_size_ = out_size
            else:
                # intermediate layer
                out_size_ = hid_size

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size_, int_size=hid_size_, output_size=out_size_,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], dropout=self.hparams.get('dropout', 0.2))
            name = 'tcn_block_%02i' % global_layer_num
            self.model.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        return global_layer_num

    def build_decoder(self, in_size, hid_size, out_size):
        """Construct the decoder using hparams."""

        global_layer_num = 0

        out_size_ = in_size  # set "output size" of the layer that feeds into this module
        for i_layer in range(self.hparams['n_hid_layers']):

            dilation = 2 ** (self.hparams['n_hid_layers'] - i_layer - 1)  # down by powers of 2
            in_size_ = out_size_  # input is output size of previous block
            hid_size_ = hid_size
            if i_layer == (self.hparams['n_hid_layers'] - 1):
                # final layer
                out_size_ = out_size
                final_activation = self.hparams['activation']
                predictor_block = False
            else:
                # intermediate layer
                out_size_ = hid_size
                final_activation = self.hparams['activation']
                predictor_block = False

            # conv -> activation -> dropout (+ residual)
            tcn_block = DilationBlock(
                input_size=in_size_, int_size=hid_size_, output_size=out_size_,
                kernel_size=self.hparams['n_lags'], stride=1, dilation=dilation,
                activation=self.hparams['activation'], final_activation=final_activation,
                dropout=self.hparams.get('dropout', 0.2), predictor_block=predictor_block)
            name = 'tcn_block_%02i' % global_layer_num
            self.model.add_module(name, tcn_block)

            # update layer info
            global_layer_num += 1

        # add final fully-connected layer
        dense = nn.Conv1d(
            in_channels=out_size,
            out_channels=out_size,
            kernel_size=1)  # kernel_size=1 <=> dense, fully connected layer
        self.model.add_module('final_dense_%02i' % global_layer_num, dense)

        return global_layer_num

    def forward(self, x, **kwargs):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor object
            input data of shape (n_sequences, sequence_length, n_markers)

        Returns
        -------
        torch.Tensor
            shape (n_sequences, sequence_length, n) where n is the embedding dimension if an
            encoder, or n_markers if a decoder/predictor

        """

        # push data through encoder to get latent embedding
        # x = B x T x N (e.g. B = 2, T = 500, N = 16)
        # x.transpose(1, 2) -> x = B x N x T
        # x = layer(x) -> x = B x M x T
        # x.transpose(1, 2) -> x = B x T x M
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class MultiStageModel(nn.Module):
    def __init__(self, hparams, type='encoder', in_size=None, hid_size=None, out_size=None):
        #num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        num_stages = 4
        num_layers = 10
        num_f_maps = 64
        
        if type == 'encoder':
            dim = hparams['input_size'] if in_size is None else in_size
            num_classes = hparams['n_hid_units']
        else:
            dim = hparams['n_hid_units'] if in_size is None else in_size
            num_classes = hparams['input_size']

        #dim = dd
        #num_classes
        
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

#     def forward(self, x, mask):
#         out = self.stage1(x, mask)
#         outputs = out.unsqueeze(0)
#         for s in self.stages:
#             out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
#             outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
#         return outputs
    
    def forward(self, x):
        if x.shape[1] == 2048:
            x = torch.transpose(x, 1, 2)
        out = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

#     def forward(self, x, mask):
#         out = self.conv_1x1(x)
#         for layer in self.layers:
#             out = layer(out, mask)
#         out = self.conv_out(out) * mask[:, 0:1, :]
#         return out
    
    def forward(self, x):
        #print(f"pre shape {x.shape}")
        if x.shape[1] == 2048:
            x = torch.transpose(x, 1, 2)#x.transpose(1,2)
        #print(f"post shape {x.shape}")
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

#     def forward(self, x, mask):
#         out = F.relu(self.conv_dilated(x))
#         out = self.conv_1x1(out)
#         out = self.dropout(out)
#         return (x + out) * mask[:, 0:1, :]
    
    def forward(self, x):
        if x.shape[1] == 2048:
            x = torch.transpose(x, 1, 2)
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)

