import math
import numpy as np
import os
import pickle
from scipy.special import softmax as scipy_softmax
from scipy.stats import entropy
import torch
from sklearn.metrics import accuracy_score, r2_score
from torch import nn, save

from daart import losses
from daart.models.base import BaseModel, BaseInference, BaseGenerative, reparameterize_gaussian, get_activation_func_from_str
from daart.transforms import MakeOneHot
    
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence





class GMDGMMGenerative(BaseModel):
    """
    Generative model for GMDGM
    """
    
    def __init__(self, hparams, model):
        """
        
        Parameters
        ----------
        hparams : dict
            
        """
        super().__init__()
        self.hparams = hparams
        self.model = model#hparams['model']
    
    def build_model(self):
        """Construct the generative netowork using hparams."""

        # select backbone network for geneative model
        if self.hparams['backbone_generative'].lower() == 'temporal-mlp':
            from daart.backbones.temporalmlp import TemporalMLP as Module
        elif self.hparams['backbone_generative'].lower() == 'tcn':
            raise NotImplementedError('deprecated; use dtcn instead')
        elif self.hparams['backbone_generative'].lower() == 'dtcn':
            from daart.backbones.tcn import DilatedTCN as Module
        elif self.hparams['backbone_generative'].lower() in ['lstm', 'gru']:
            from daart.backbones.rnn import RNN as Module
        elif self.hparams['backbone_generative'].lower() == 'tgm':
            raise NotImplementedError
            # from daart.models.tgm import TGM as Module
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_generative'])

        # build label prior: p(y)
        probs = np.ones((self.hparams['n_total_classes'],)) * (1/self.hparams['n_total_classes'])
        

        assert np.isclose([np.sum(np.array(probs))], [1])
        self.hparams['py_probs'] = probs
        
        # build latent_generator: p(z|y)
        # linear layer is essentially a lookup table of shape (n_hid_units, n_total_classes)
        self.model['pz_y_mean'] = self._build_linear(
            0, 'pz_y_mean', self.hparams['n_total_classes'], self.hparams['n_latents'])
        self.model['pz_y_logvar'] = self._build_linear(
            0, 'pz_y_logvar', self.hparams['n_total_classes'], self.hparams['n_latents'])
        
        # build decoder: p(x|z)
        self.model['decoder'] = self._build_mlp(
                0, self.hparams['n_latents'], self.hparams['n_latents'], self.hparams['input_size'])
#         self.model['decoder'] = Module(
#             self.hparams, 
#             type='decoder',
#             in_size=self.hparams['n_hid_units'],
#             hid_size=self.hparams['n_hid_units'],
#             out_size=self.hparams['input_size'])
        
    def __str__(self):
        """Pretty print generative model architecture."""

        format_str = 'Generative network architecture: %s\n' % self.hparams['backbone_generative'].upper()
        format_str += '------------------------\n'

        if 'decoder' in self.model:
            format_str += 'Decoder (p(x|z)):\n'
            for i, module in enumerate(self.model['decoder'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'py' in self.model:
            format_str += 'p(y):\n'
            for i, module in enumerate(self.model['py']):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'
                
        if 'pz_y_mean' in self.model:
            format_str += 'p(z|y) mean:\n'
            for i, module in enumerate(self.model['pz_y_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                
        if 'pz_y_logvar' in self.model:
            format_str += 'p(z|y) logvar:\n'
            for i, module in enumerate(self.model['pz_y_logvar']):
                format_str += str('    {}: {}\n'.format(i, module))

        return format_str
    
    def forward(self, **kwargs): 
        
        device = self.hparams['device']
        z_input = kwargs['z_sample']

        y_probs = kwargs['y_probs']
        idxs_labeled = kwargs['idxs_labeled']
        ignore_class = self.hparams['ignore_class']
        
        y_dim = self.hparams['n_total_classes']

        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x_t|z_t)
        x_hat = self.model['decoder'](z_input)
        
        # create z_(t-1) and z_(t+1) tensors
        z_t_skip_first = z_input[:, :, 1:, :]
        z_t_skip_last = z_input[:, :, :-1, :]

        
        # push y_t and z_(t-1) through generative model to get parameters of p(z_t| y_t)
        pz_mean = torch.zeros((y_dim, x_hat.shape[1], x_hat.shape[2], self.hparams['n_latents']))
        pz_logvar = torch.zeros((y_dim, x_hat.shape[1], x_hat.shape[2], self.hparams['n_latents']))
        
        for i in range(y_dim):
            
            pz_input = torch.zeros((pz_mean.shape[1], pz_mean.shape[2], y_dim)).to(device)
            pz_input[:, :, i] = torch.tensor(1).to(device)
            
            pz_mean[i, :, :, :] = self.model['pz_y_mean'](pz_input) 
            pz_logvar[i, :, :, :] = self.model['pz_y_logvar'](pz_input)
        
     
        return {
            
            'pz_mean': pz_mean,  # (n_total_classes, n_sequences, sequence_length, embedding_dim)
            'pz_logvar': pz_logvar,  # (n_total_classes, n_sequences, sequence_length, embedding_dim)
            'reconstruction': x_hat,  # (n_total_classes, n_sequences, sequence_length, n_markers)
        }
    
    
    
class GMDGMMInference(BaseModel):
    """
    Approximate posterior setup shared among models
    """
    
    def __init__(self, hparams, model):
        """
        
        Parameters
        ----------
        hparams : dict
            
        """
        super().__init__()
        self.hparams = hparams
        self.model = model

    
    def build_model(self):
        """Construct the inference network using hparams."""

        # select backbone network for inference model
        if self.hparams['backbone_inference'].lower() == 'temporal-mlp':
            from daart.backbones.temporalmlp import TemporalMLP as Module
        elif self.hparams['backbone_inference'].lower() == 'tcn':
            raise NotImplementedError('deprecated; use dtcn instead')
        elif self.hparams['backbone_inference'].lower() == 'dtcn':
            from daart.backbones.tcn import DilatedTCN as Module
        elif self.hparams['backbone_inference'].lower() in ['lstm', 'gru']:
            from daart.backbones.rnn import RNN as Module
        elif self.hparams['backbone_inference'].lower() == 'tgm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_inference'])

        if self.hparams['temporal_inference']:
            # build classifier: q(y|x)
            self.model['qy_x'] = Module(
                self.hparams, 
                type='decoder',
                in_size=self.hparams['input_size'], 
                hid_size=self.hparams['n_hid_units'], 
                out_size=self.hparams['n_total_classes'])


            # build encoder: q(z|x,y)

            self.model['encoder'] = Module(
                self.hparams, 
                in_size=self.hparams['n_total_classes'] + self.hparams['input_size'],
                hid_size=self.hparams['n_hid_units'],
                out_size=self.hparams['n_latents'])

            self.model['qz_xy_mean'] = self._build_linear(
                        global_layer_num=0, name='qz_xy_mean',
                        in_size=self.hparams['n_latents'], out_size=self.hparams['n_latents'])

            self.model['qz_xy_logvar'] = self._build_linear(
                global_layer_num=0, name='qz_xy_logvar',
                        in_size=self.hparams['n_latents'], out_size=self.hparams['n_latents']) 
            
        else:
            # build classifier: q(y|x)
            self.model['qy_x'] = self._build_mlp(
                0, self.hparams['input_size'], self.hparams['n_hid_units'], self.hparams['n_total_classes'])

            # build encoder: q(z|x,y)

            self.model['encoder'] = self._build_mlp(
                0, self.hparams['n_total_classes'] + self.hparams['input_size'],
                self.hparams['n_hid_units'], self.hparams['n_latents'])
            
            

            self.model['qz_xy_mean'] = self._build_linear(
                        global_layer_num=0, name='qz_xy_mean',
                        in_size=self.hparams['n_latents'], out_size=self.hparams['n_latents'])

            self.model['qz_xy_logvar'] = self._build_linear(
                global_layer_num=0, name='qz_xy_logvar',
                        in_size=self.hparams['n_latents'], out_size=self.hparams['n_latents']) 
        
    def __str__(self):
        format_str = 'Inference network architecture: %s\n' % self.hparams['backbone_inference'].upper()
        format_str += '------------------------\n'

        format_str += 'Encoder (q(z|x,y)):\n'
        for i, module in enumerate(self.model['encoder'].model):
            format_str += str('    {}: {}\n'.format(i, module))
        format_str += '\n'

        if 'qy_x' in self.model:
            format_str += 'q(y|x):\n'
            for i, module in enumerate(self.model['qy_x'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'qz_xy_mean' in self.model:
            format_str += 'q(z|xy) mean:\n'
            for i, module in enumerate(self.model['qz_xy_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                
        if 'qz_xy_logvar' in self.model:
            format_str += 'q(z|xy) logvar:\n'
            for i, module in enumerate(self.model['qz_xy_logvar']):
                format_str += str('    {}: {}\n'.format(i, module))
                
        return format_str
    
    def forward(self, x, y):
        
        device = self.hparams['device']
        y = y.to(device=device)
        x = x.to(device=device)
        # push inputs through classifier to get q(y|x)
        y_dim = self.hparams['n_total_classes']
        device = self.hparams['device']
        qy_x_logits = self.model['qy_x'](x).to(device=device)
        #print('yl', y_logits.shape)
        
        # initialize and sample q(y|x) (should be a one-hot vector)
        qy_x_probs = nn.Softmax(dim=2)(qy_x_logits).to(device=device)
        #qy_x_logits = y_logits
        
        # qy vector to use for taking expectations in loss terms
        qy_e_probs = qy_x_probs.clone().detach().to(device=device) # (n_sequences, sequence_length, n_total_classes)
       
        # loop over sequences in batch
        idxs_labeled = torch.zeros_like(y).to(device=device)
        
        for s in range(y.shape[0]):
            idxs_labeled[s] = y[s] != self.hparams.get('ignore_class', 0)
            if idxs_labeled[s].sum() != 0:
                qy_e_probs[s][idxs_labeled[s] == 1] = MakeOneHot()(y[s][idxs_labeled[s] == 1], n_classes=y_dim).to(device=device)

        
        # concatenate all poosible y's with input x
        # (n_sequences, sequence_length, n_total_classes, input_dim + n_total_classes))
        
        x_with_y_dim = x.unsqueeze(2).repeat(1, 1, y_dim, 1).to(device=device) # (n_sequences, sequence_length, n_total_classes, input_dim))

        
        ys = torch.eye(y_dim).expand((x.shape[0], x.shape[1], y_dim, y_dim)).to(device=device)
        xy = torch.cat([x_with_y_dim, ys], dim=3).to(device=device)
        
        # push [y, x] through encoder to get parameters of q(z|x,y)
        w = torch.zeros((y_dim, x.shape[0], x.shape[1], self.hparams['n_latents'])).to(device=device)
        
        for k in range(y_dim):
            w_temp = self.model['encoder'](xy[:, :, k, :]).to(device=device) # (n_sequences, sequence_length, embedding_dim)
            w[k] = w_temp
        
        
        qz_xy_mean = self.model['qz_xy_mean'](w)

        
        qz_xy_logvar = self.model['qz_xy_logvar'](w)
        
        
        # sample with reparam trick
        z_xy_sample = qz_xy_mean + torch.randn(qz_xy_mean.shape, device=qy_x_logits.device) * qz_xy_logvar.exp().pow(0.5)
        #print('z samp', z_xy_sample.shape, z_xy_sample)
        
        return {
            'qy_x_probs': qy_x_probs,  # (n_sequences, sequence_length, n_classes)
            'qy_e_probs': qy_e_probs, # (n_sequences, sequence_length, n_classes)
            'qy_x_logits': qy_x_logits,  # (n_sequences, sequence_length, n_classes)
            'qz_xy_mean': qz_xy_mean,  # (n_classes, n_sequences, sequence_length, embedding_dim)
            'qz_xy_logvar': qz_xy_logvar,  # (n_classes, n_sequences, sequence_length, embedding_dim)
            'z_xy_sample': z_xy_sample, # (n_classes, n_sequences, sequence_length, embedding_dim)
            'idxs_labeled': idxs_labeled,  # (n_sequences, sequence_length)
        }
    
