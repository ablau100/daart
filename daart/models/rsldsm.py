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


class RSLDSMGenerative(BaseModel):
    """
    Generative model for rSLDS
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
        """Construct the generative netowork using hparams."""
        
        """
        Notation guide:
        py_1 := p(y_1)
        pz_1 := p(z_1)
        
        px_t := p(x_t|z_t)
        py_t := p(y_t|y_(t-1), z_(t-1)), t > 1
        pz_t := p(z_t|z_(t-1), y_t), t > 1
        
        """

        # select backbone network for geneative model
        if self.hparams['backbone_generative'].lower() == 'temporal-mlp':
            from daart.backbones.temporalmlp import TemporalMLP as Module
        elif self.hparams['backbone_generative'].lower() == 'tcn':
            raise NotImplementedError('deprecated; use dtcn instead')
        elif self.hparams['backbone_generative'].lower() == 'dtcn':
            from daart.backbones.tcn import DilatedTCN as Module
        elif self.hparams['backbone_generative'].lower() in ['lstm', 'gru']:
            from daart.backbones.rnn import RNN as Module
        elif self.hparams['backbone_generative'].lower() == 'msm':
            from daart.backbones.ms_tcn import MultiStageModel as Module
        elif self.hparams['backbone_generative'].lower() == 'tgm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_generative'])

        # build label prior: p(y_1)
        logits = []
        
        prob = (1/self.hparams['n_total_classes'])
        val = float(np.log((prob/(1-prob))))
        for k in range(self.hparams['n_total_classes']):
            logits.append(val)
        
        self.hparams['py_1_logits'] = logits
        
        # build updated label prior/transition matrices: p(y_t|y_(t-1), z_(t-1)), t > 1
        if self.hparams['backbone_transition'] == 'linear':
            self.model['py_t_probs'] = self._build_linear(
                0, 'py_t_probs', self.hparams['n_latents'], self.hparams['n_total_classes'])           
        elif self.hparams['backbone_transition'] == 'mlp':
            self.model['py_t_probs'] = self._build_mlp(
                0, self.hparams['n_latents'], self.hparams['n_latents'], self.hparams['n_total_classes'])
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_transition'])
            
        
        # build latent prior: p(z_1)
        self.hparams['pz_1_mean'] = [0 for i in range(self.hparams['n_latents'])]
        self.hparams['pz_1_logvar'] = -9.21
        
        # build latent_generator/dynamic matrices: p(z_t|z_(t-1), y_t)
        for i in range(self.hparams['n_total_classes']):
            
            if self.hparams['backbone_dynamic'] == 'linear':
                self.model['pz_t_mean_{}'.format(i)] = self._build_linear(
                    0, 'pz_t_mean', self.hparams['n_latents'], self.hparams['n_latents'])            
            elif self.hparams['backbone_dynamic'] == 'mlp':
                self.model['pz_t_mean_{}'.format(i)] = self._build_mlp(
                    0, self.hparams['n_latents'],  self.hparams['n_latents'], self.hparams['n_latents'])         
            else:
                raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_dynamic'])
                
        
        self.model['pz_t_logvar'] = self._build_linear(
            0, 'pz_t_logvar', self.hparams['n_total_classes'], self.hparams['n_latents'])
        
        # build decoder: p(x_t|z_t)
        if self.hparams['backbone_decoder'] == 'linear':
            self.model['decoder'] = self._build_linear(
                0, 'decoder', self.hparams['n_latents'], self.hparams['input_size'])        
        elif self.hparams['backbone_decoder'] == 'mlp':
            self.model['decoder'] = self._build_mlp(
                0, self.hparams['n_latents'], self.hparams['n_latents'], self.hparams['input_size'])
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_decoder'])
        
        
        
    def __str__(self):
        """Pretty print generative model architecture."""
        
        format_str = 'Generative network architecture: %s\n' % self.hparams['backbone_generative'].upper()
        format_str += '------------------------\n'

#         if 'decoder' in self.model:
#             format_str += 'Decoder (p(x_t|z_t)):\n'
#             for i, module in enumerate(self.model['decoder'].model):
#                 format_str += str('    {}: {}\n'.format(i, module))
#             format_str += '\n'

        if 'py_t_probs' in self.model:
            format_str += 'p(y_t|y_(t-1), z_(t-1)):\n'
            for i, module in enumerate(self.model['py_t_probs']):
                format_str += str('    {}: {}\n'.format(i, module))
                format_str += str(' Weights py: {}\n'.format(module.weight))
                format_str += str(' bias py: {}\n'.format(module.bias))
            format_str += '\n'
                
        if 'pz_t_mean' in self.model:
            format_str += 'p(z_t|z_(t-1), y_t) mean:\n'
            for i, module in enumerate(self.model['pz_t_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                format_str += str(' Weights pz: {}\n'.format(module.weight))
                format_str += str(' bias pz: {}\n'.format(module.bias))
            
                
#         if 'pz_t_logvar' in self.model:
#             format_str += 'p(z_t|z_(t-1), y_t) logvar:\n'
#             for i, module in enumerate(self.model['pz_t_logvar']):
#                 format_str += str('    {}: {}\n'.format(i, module))

        return ''#format_str
    
    def forward(self, **kwargs): 
        
        """
        forward function for rSLDS generative model
        kwargs['z_sample']: (n_classes, n_sequences, sequence_length, embedding_dim)
        kwargs['y_gt']: (n_sequences, sequence_length)
        kwargs['y_probs']: (n_sequences, sequence_length, n_classes)
        
        
        gen_inputs = {
           'y_gt': inf_outputs['y'],
           'y_probs': inf_outputs['qy_x_probs'],
           'z_sample': inf_outputs['z_xy_sample'], 
           'idxs_labeled': inf_outputs['idxs_labeled']
        }
        
        """
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

        # push z_(t-1) through generative model to get parameters of p(y_t|y_(t-1), z_(t-1))
        py_t_logits = self.model['py_t_probs'](z_t_skip_last) #(n_classes, n_sequences, sequence_length, embedding_dim)
        
        # concat py_1 with py_t, t > 1
        py_1_logits = torch.tensor(self.hparams['py_1_logits'])
        
        py_logits = torch.zeros(py_t_logits.shape[0], py_t_logits.shape[1], 
                                py_t_logits.shape[2]+1,py_t_logits.shape[3]).to(device=self.hparams['device'])
        
        py_logits[:, :, 1:, :] = py_t_logits 
        py_logits[:, :, 0, :] = py_1_logits 

        
        # push y_t and z_(t-1) through generative model to get parameters of p(z_t|z_(t-1), y_t)
        pz_t_mean = torch.zeros((y_dim, x_hat.shape[1], x_hat.shape[2]-1, y_dim, self.hparams['n_latents']))
        for i in range(y_dim):
            pz_t_mean[:, :, :, i, :] = self.model['pz_t_mean_{}'.format(i)](z_t_skip_last)       
        
        # concat pz_1 mean with pz_t mean , t > 1
        pz_1_mean = torch.tensor(self.hparams['pz_1_mean'])
        pz_mean = torch.zeros(pz_t_mean.shape[0], pz_t_mean.shape[1], pz_t_mean.shape[2]+1, pz_t_mean.shape[3], pz_t_mean.shape[4]).to(device=self.hparams['device'])
        
        pz_mean[:, :, 1:, :,  :] = pz_t_mean
        pz_mean[:, :, 0, :, :] = pz_1_mean
        
        logvar_input = torch.zeros((z_input.shape[1], z_input.shape[2], y_dim, y_dim)).to(device=device)
        logvar_input[:] = torch.eye(y_dim).to(device=device)
        pz_logvar = self.model['pz_t_logvar'](logvar_input)

        return {
            'py_logits': py_logits, # (n_total_classes, n_sequences, sequence_length, n_total_classes)
            'pz_mean': pz_mean,  # (n_total_classes, n_sequences, sequence_length, n_total_classes, embedding_dim)
            'pz_logvar': pz_logvar,  # (n_sequences, sequence_length, n_total_classes, embedding_dim)
            'reconstruction': x_hat,  # (n_total_classes, n_sequences, sequence_length, n_markers)
        }
    
    
 
    
class RSLDSMInference(BaseModel):
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
        elif self.hparams['backbone_inference'].lower() == 'msm':
            from daart.backbones.ms_tcn import MultiStageModel as Module
        elif self.hparams['backbone_inference'].lower() == 'tgm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_inference'])

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
                
        return ''#format_str
    
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
        
        return {
            'qy_x_probs': qy_x_probs,  # (n_sequences, sequence_length, n_classes)
            'qy_e_probs': qy_e_probs, # (n_sequences, sequence_length, n_classes)
            'qy_x_logits': qy_x_logits,  # (n_sequences, sequence_length, n_classes)
            'qz_xy_mean': qz_xy_mean,  # (n_classes, n_sequences, sequence_length, embedding_dim)
            'qz_xy_logvar': qz_xy_logvar,  # (n_classes, n_sequences, sequence_length, embedding_dim)
            'z_xy_sample': z_xy_sample, # (n_classes, n_sequences, sequence_length, embedding_dim)
            'idxs_labeled': idxs_labeled,  # (n_sequences, sequence_length)
        }
    