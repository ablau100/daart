"""rSLDS models/modules in PyTorch."""

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


class RSLDSGenerative(BaseModel):
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
        elif self.hparams['backbone_generative'].lower() == 'tgm':
            raise NotImplementedError
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone_generative'])

        # build label prior: p(y_1)
        #probs = [.25,.25,.25,.25]#[0.00001, .1, .4, .4, .1]#np.ones((self.hparams['n_total_classes'],))
        
        logits = [-1.09861, -1.09861, -1.09861, -1.09861]
        
        # this code is for specifiying logits/probs for observed and aug classes
        
#         background_prob = 0.01
#         probs.append(background_prob)
        
#         new_classes_indexes = [1,2]
        
#         for i in range(self.hparams['n_total_classes']):
#             if i in new_classes_indexes:
#                 probs.append((1-background_prob) * .7 * (1/len(new_classes_indexes)))
#             elif i > 0:
#                 probs.append((1-background_prob) * .3 * (1/(self.hparams['n_observed_classes']-1)))

#         probs[:self.hparams['output_size']] /= (self.hparams['output_size'] * 2)
        #probs[:] /= (self.hparams['n_aug_classes']+self.hparams['n_observed_classes'])
#)

        
        #self.hparams['py_1_probs'] = probs
        self.hparams['py_1_logits'] = logits
        
#         # build updated label prior: p(y_t|y_(t-1), z_(t-1)), t > 1

#         self.model['py_t_probs'] = self._build_linear(
#             0, 'py_t_probs', self.hparams['n_hid_units'], (self.hparams['n_total_classes']*self.hparams['n_total_classes']))
           
        self.model['py_t_probs'] = self._build_linear(
            0, 'py_t_probs', self.hparams['n_hid_units'], self.hparams['n_total_classes'])
        
#         self.model['py_t_probs'] = self._build_mlp(
#             0, self.hparams['n_hid_units'], 16, self.hparams['n_total_classes'])
        
        
        
        # build latent prior: p(z_1)
        self.hparams['pz_1_mean'] = [0,1]
        self.hparams['pz_1_logvar'] = -9.21
        
        # build latent_generator: p(z_t|z_(t-1), y_t)
        self.model['pz_t_mean'] = self._build_linear(
            0, 'pz_t_mean', self.hparams['n_hid_units'], (self.hparams['n_hid_units']*self.hparams['n_total_classes']))
        
#         self.model['pz_t_mean'] = self._build_mlp(
#             0, self.hparams['n_hid_units'], 16, (self.hparams['n_hid_units']*self.hparams['n_total_classes']))
        
        self.model['pz_t_logvar'] = self._build_linear(
            0, 'pz_t_logvar', self.hparams['n_total_classes'], self.hparams['n_hid_units'])
        
        # build decoder: p(x_t|z_t)
        self.model['decoder'] = self._build_linear(
            0, 'decoder', self.hparams['n_hid_units'], self.hparams['input_size'])
        
#         self.model['decoder'] = self._build_mlp(
#             0, self.hparams['n_hid_units'], 16, self.hparams['input_size'])
        
        
#         self.model['decoder'] = Module(
#             self.hparams, 
#             type='decoder',
#             in_size=self.hparams['n_hid_units'],
#             hid_size=self.hparams['n_hid_units'],
#             out_size=self.hparams['input_size'])

        
        As = torch.tensor(
             [[ 9.80436447e-01, -1.30991746e-01],
             [ 1.27755862e-01,  9.88280774e-01],
             [ 9.87511608e-01, -6.56728202e-02],
             [ 6.21772981e-02,  9.93775794e-01],
             [ 9.93601840e-01,  1.04869992e-03],
             [ 6.04993083e-04,  9.87245556e-01],
             [ 9.94231254e-01, -2.44135927e-04],
             [-1.31332387e-03,  9.86318472e-01]])

        bs = torch.tensor([ 0.16618703, -0.30929626, -0.01083673,  0.08616756,  0.10652665, -0.01280473,
     -0.24733377,  0.01513071])
        
        
#         R = torch.tensor([[ 20.5312,   2.1907],
#         [-19.4506,  -2.1242],
#         [ -0.3354,  -2.3760],
#         [ -0.7452,   2.3095]])
        
#         r = torch.tensor([-34.4430,  -5.5914,  21.5151,  18.5194])
        
#         for i, module in enumerate(self.model['pz_t_mean']):
#             module.weight = nn.Parameter(As)
#             module.bias = nn.Parameter(bs)
        
#         for i, module in enumerate(self.model['py_t_probs']):
#             module.weight = nn.Parameter(R)
#             module.bias = nn.Parameter(r)
            
            
            
        
        
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

        return format_str
    
    def forward(self, **kwargs): 
        
        """
        forward function for rSLDS generative model
        kwargs['z_sample']: (n_sequences, sequence_length, embedding_dim)
        kwargs['y_sample']: (n_sequences, sequence_length, n_total_classes)
        
        """
        
        z_input = kwargs['z_sample']
        y_input = kwargs['y_sample']
        y_dim = self.hparams['n_total_classes']
        
        # reshape z to have an extra dim of length y_dim
        z_with_y_dim = z_input.unsqueeze(2).repeat(1, 1, y_dim, 1)
        
        # create z_(t-1) and z_(t+1) tensors
        z_t_skip_first = z_input[:, 1:, :]
        z_t_skip_first_with_y_dim = z_t_skip_first.unsqueeze(2).repeat(1, 1, y_dim, 1)
        
        z_t_skip_last = z_input[:, :-1, :]
        z_t_skip_last_with_y_dim = z_t_skip_last.unsqueeze(2).repeat(1, 1, y_dim, 1)
        
        # create y_(t-1) and y_(t+1) tensors
        y_t_skip_first = y_input[:, 1:, :]
        y_t_skip_last = y_input[:, :-1, :]
        
        # push z_(t-1) through generative model to get parameters of p(y_t|y_(t-1), z_(t-1))
        py_t_logits = self.model['py_t_probs'](z_t_skip_last)
#         py_t_logits = torch.reshape(py_t_logits, (py_t_logits.shape[0], py_t_logits.shape[1], y_dim,-1))

#         # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from t=0
#         py_indexer = y_t_skip_last.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_last.ndim, py_t_logits.shape[3])
#         py_t_logits = torch.gather(py_t_logits, 2, py_indexer).squeeze(2)
        
        # concat py_1 with py_t, t > 1
        py_1_logits = torch.tensor(self.hparams['py_1_logits'])
        
        py_logits = torch.zeros(py_t_logits.shape[0], py_t_logits.shape[1]+1, py_t_logits.shape[2]).to(device=self.hparams['device'])
        py_logits[:, 1:, :] = py_t_logits 
        py_logits[:, 0, :] = py_1_logits
        
        # push y_t and z_(t-1) through generative model to get parameters of p(z_t|z_(t-1), y_t)
        pz_t_mean = self.model['pz_t_mean'](z_t_skip_last)
        pz_t_mean = torch.reshape(pz_t_mean, (pz_t_mean.shape[0], pz_t_mean.shape[1], y_dim,-1))

        
        pz_t_logvar = self.model['pz_t_logvar'](y_t_skip_first)
        
        
        # index mean by y_t - z_t starts from t=1, so our indexer y starts from y=1
        pz_indexer = y_t_skip_first.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_first.ndim, pz_t_mean.shape[3])
        pz_t_mean = torch.gather(pz_t_mean, 2, pz_indexer).squeeze(2)
        
        # concat pz_1 mean with pz_t mean , t > 1
        pz_1_mean = torch.tensor(self.hparams['pz_1_mean'])
        pz_mean = torch.zeros(pz_t_mean.shape[0], pz_t_mean.shape[1]+1, pz_t_mean.shape[2]).to(device=self.hparams['device'])
        
        pz_mean[:, 1:, :] = pz_t_mean
        pz_mean[:, 0, :] = pz_1_mean
        
        # concat pz_1 logvar with pz_t logvar , t > 1
        pz_1_logvar = torch.tensor(self.hparams['pz_1_logvar'])
        pz_logvar = torch.zeros(pz_t_mean.shape[0], pz_t_mean.shape[1]+1, pz_t_mean.shape[2]).to(device=self.hparams['device'])
        pz_logvar[:, 1:, :] = pz_t_logvar
        pz_logvar[:, 0, :] = pz_1_logvar
       
        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x_t|z_t)
        x_hat = self.model['decoder'](z_input)

        return {
            'py_logits': py_logits, # (n_sequences, sequence_length, n_total_classes)
            'pz_mean': pz_mean,  # (n_sequences, sequence_length, embedding_dim)
            'pz_logvar': pz_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'reconstruction': x_hat,  # (n_sequences, sequence_length, n_markers)
        }


class RSLDS(BaseModel):
    """rSLDS Model.
    
    [insert arxiv link here]
    """

    def __init__(self, hparams):
        """
        
        Parameters
        ----------
        hparams : dict
            - backbone (str): 'temporal-mlp' | 'dtcn' | 'lstm' | 'gru'
            - rng_seed_model (int): random seed to control weight initialization
            - input_size (int): number of input channels
            - output_size (int): number of observed classes
            - n_aug_classes (int): number of additional classes without labels
            - sequence_pad (int): padding needed to account for convolutions
            - n_hid_layers (int): hidden layers of network architecture
            - n_hid_units (int): hidden units per layer
            - n_lags (int): number of lags in input data to use for temporal convolution
            - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
            - lambda_strong (float): hyperparam on strong label classification 
              (alpha in original paper)
            
        """
        super().__init__()
        self.hparams = hparams
        #self.keys = ['py_probs','qy_x_probs','y_sample','qz_xy_mean','qz_xy_logvar'
         #       ,'pz_mean','pz_logvar','reconstruction']
        self.keys = ['qy_x_probs','y_sample','qz_xy_mean','qz_xy_logvar'
                ,'pz_mean','pz_logvar','reconstruction']

        # model dict will contain some or all of the following components:
        # - classifier: q(y|x) [weighted by hparams['lambda_strong'] on labeled data]
        # - encoder: q(z|x,y)
        # - decoder: p(x|z)
        # - latent_generator: p(z|y)

        self.model = nn.ModuleDict()
        #hparams['model'] = self.model
        
        self.inference = BaseInference(hparams, self.model)
        self.generative = RSLDSGenerative(hparams, self.model)
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        ignore_index = hparams.get('ignore_class', 0)
        self.class_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        
        # MSE loss for reconstruction
        self.recon_loss = nn.MSELoss(reduction='mean')
        
    def __str__(self):
        """Pretty print model architecture."""

        format_str = '\n\nInference model\n'
        format_str += self.inference.__str__()

        format_str += '------------------------\n'

        format_str += '\n\nGenerative model\n'
        format_str += self.generative.__str__()

        return format_str
    
    def sampler(self, num_samples=1000):
        
        y_samples = []
        z_samples = []
        x_samples = []
        #self.hparams['py_1_probs'] = [.25,.25,.25,.25]
        self.hparams['py_1_probs'] = [-1.09861, -1.09861, -1.09861, -1.09861]
        y_dim = 4#self.hparams['n_total_classes']
        self.hparams['n_total_classes'] = 4
        
        ##################
        # for t = 1
        ##################
        
        # sample y_1
        py_1_probs = torch.tensor(self.hparams['py_1_probs'])
        
        py_1 = Categorical(logits=py_1_probs)
        y_1 = py_1.sample()#.type(torch.FloatTensor)
        #print('un', y_1.unsqueeze(-1).shape)
        y_1 = MakeOneHot()(y_1.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
        y_samples.append(torch.squeeze(y_1))
        
        # sample z_1
        z_1 = torch.normal(torch.tensor([0,1]).type(torch.FloatTensor), torch.ones(self.hparams['n_hid_units']) *1.5)
        z_samples.append(z_1)
        
        # sample x_1 | z_1
        
        ##################
        # for t = 2 to T
        ##################
        
        
        for t in range(1, num_samples):
            
            # sample y_t | y_(t-1), z_(t-1)
            
            py_t_probs = self.model['py_t_probs'](z_samples[t-1])
            
            
        
#             py_t_probs = (torch.reshape(py_t_probs, (y_dim,-1)))   
 
#             # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from t=0
#             y_ind = torch.argmax(y_samples[t-1])
#             py_t_probs = py_t_probs[y_ind] # (y_dim, y_dim)
            
            #print('py p', py_t_probs, py_t_probs.shape)
            py_t = Categorical(logits=py_t_probs)
            y_t = py_t.sample()
            
            y_t = MakeOneHot()(y_t.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
            y_t = torch.squeeze(y_t).type(torch.LongTensor)
            y_samples.append(y_t)

            # sample z_t | y_t, z_(t-1)
            pz_t_mean = self.model['pz_t_mean'](z_samples[t-1])
            
            pz_t_mean = torch.reshape(pz_t_mean, (y_dim,-1)) # (y dim, z dim)

            
            z_ind = torch.argmax(y_samples[t])
            pz_t_mean = pz_t_mean[z_ind]
            
            # use fix var for test
            pz_t_logvar = torch.ones_like(pz_t_mean) * 1e-4#self.model['pz_t_logvar'](y_samples[t].type(torch.FloatTensor))
            
            z_t = torch.normal(pz_t_mean, pz_t_logvar)
            z_samples.append(z_t)
        
        
        # sample x_t | z_t
        for t in range(0, num_samples):
            
            x_sample = self.model['decoder'](z_samples[t])
            x_samples.append(x_sample)
        
        
        return y_samples, z_samples, x_samples
    
    
#     def sampler_R(self, num_samples=1000):
        
#         y_samples = []
#         z_samples = []
#         x_samples = []
#         #self.hparams['py_1_probs'] = [.25,.25,.25,.25]
#         self.hparams['py_1_probs'] = [-1.09861, -1.09861, -1.09861, -1.09861]
#         y_dim = 4#self.hparams['n_total_classes']
#         self.hparams['n_total_classes'] = 4
        
#         ##################
#         # for t = 1
#         ##################
        
#         # sample y_1
#         py_1_probs = torch.tensor(self.hparams['py_1_probs'])
        
#         py_1 = Categorical(logits=py_1_probs)
#         y_1 = py_1.sample()#.type(torch.FloatTensor)
#         #print('un', y_1.unsqueeze(-1).shape)
#         y_1 = MakeOneHot()(y_1.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
#         y_samples.append(torch.squeeze(y_1))
        
#         # sample z_1
#         z_1 = torch.normal(torch.tensor([0,1]).type(torch.FloatTensor), torch.ones(self.hparams['n_hid_units']) * 1e-4)
#         z_samples.append(z_1)
        
#         # sample x_1 | z_1
        
#         ##################
#         # for t = 2 to T
#         ##################
        
        
#         for t in range(1, num_samples):
            
#             # sample y_t | y_(t-1), z_(t-1)
            
#             py_t_probs = self.model['py_t_probs'](z_samples[t-1])
         
#             py_t_probs = (torch.reshape(py_t_probs, (y_dim,-1)))
            
           

#             # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from t=0
#             y_ind = torch.argmax(y_samples[t-1])
#             py_t_probs = py_t_probs[y_ind] # (y_dim, y_dim)
            
#             #print('py p', py_t_probs, py_t_probs.shape)
#             py_t = Categorical(logits=py_t_probs)
#             y_t = py_t.sample()
            
#             y_t = MakeOneHot()(y_t.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
#             y_t = torch.squeeze(y_t).type(torch.LongTensor)
#             y_samples.append(y_t)

#             # sample z_t | y_t, z_(t-1)
#             pz_t_mean = self.model['pz_t_mean'](z_samples[t-1])
            
#             pz_t_mean = torch.reshape(pz_t_mean, (y_dim,-1)) # (y dim, z dim)

            
#             z_ind = torch.argmax(y_samples[t])
#             pz_t_mean = pz_t_mean[z_ind]
            
#             # use fix var for test
#             pz_t_logvar = torch.ones_like(pz_t_mean) * 1e-4#self.model['pz_t_logvar'](y_samples[t].type(torch.FloatTensor))
            
#             z_t = torch.normal(pz_t_mean, pz_t_logvar)
#             z_samples.append(z_t)
        
        
#         # sample x_t | z_t
        
#         return y_samples, z_samples
        
        
        
    def build_model(self):
        """Construct the model using hparams."""

        # set random seeds for control over model initialization
        rng_seed_model = self.hparams.get('rng_seed_model', 0)
        torch.manual_seed(rng_seed_model)
        np.random.seed(rng_seed_model)
        
        n_total_classes = self.hparams['n_observed_classes'] + self.hparams['n_aug_classes']
        self.hparams['n_total_classes'] = n_total_classes

        self.inference.build_model()
        self.generative.build_model()


    def forward(self, x, y):
        """Process input data.
        
        Parameters
        ----------
        x : torch.Tensor
            observation data of shape (n_sequences, sequence_length, n_markers)
        y : torch.Tensor
            label data of shape (n_sequences, sequence_length)

        Returns
        -------
        dict of model outputs/internals as torch tensors
            - 'y_probs' (torch.Tensor): model classification
               shape of (n_sequences, sequence_length, n_classes)
            - 'y_sample' (torch.Tensor): sample from concrete distribution
              shape of (n_sequences, sequence_length, n_classes)
            - 'z_mean' (torch.Tensor): mean of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'z_logvar' (torch.Tensor): logvar of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'reconstruction' (torch.Tensor): input decoder prediction
              shape of (n_sequences, sequence_length, n_markers)

        """
        
        inf_outputs = self.inference(x, y)
        
        # generative input
        gen_inputs = {
           'y_sample': inf_outputs['y_mixed'],
           'z_sample': inf_outputs['z_xy_sample'], 
        }
        
        #print('y samp', torch.argmax(inf_outputs['y_mixed'], axis=2))
        
        print('')
        print('y pred prop- qy ')
        for i in range(inf_outputs['qy_x_probs'].shape[0]):
            #print('yd', inf_outputs['y_mixed'][i].shape)
            y_sums = torch.argmax(inf_outputs['qy_x_probs'][i], axis=1)
            #print('ys', y_sums, y_sums.shape)
            y_prop = []
            bot = inf_outputs['qy_x_probs'].shape[1]
            for k in range(self.hparams['n_total_classes']):
                top = (y_sums[y_sums == k]).shape[0]
                y_prop.append(round(top/bot, 3))
            
            print( y_prop)
        
        
        print('')
        
        gen_outputs = self.generative(**gen_inputs)

        # merge the two outputs
        output_dict = {**inf_outputs, **gen_outputs}

        return output_dict
    
    def training_step(self, data, accumulate_grad=True, **kwargs):
        """Calculate negative log-likelihood loss for supervised models.
        The batch is split into chunks if larger than a hard-coded `chunk_size` to keep memory
        requirements low; gradients are accumulated across all chunks before a gradient step is
        taken.
        Parameters
        ----------
        data : dict
            signals are of shape (n_sequences, sequence_length, n_channels)
        accumulate_grad : bool, optional
            accumulate gradient for training step
        Returns
        -------
        dict
            - 'loss' (float): total loss (negative log-like under specified noise dist)
            - other loss terms depending on model hyperparameters
        """

        # define hyperparams
        kl_y_weight = self.hparams.get('kl_y_weight', 100)
        lambda_strong = self.hparams.get('lambda_strong', 1)
       # print('ls: ', lambda_strong)
        kl_weight = self.hparams.get('kl_weight', 1)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        labels_wpad = data['labels_strong']
        
#         # remove half of labels randomly by class
        
#         to_r = np.random.uniform(0,1,labels_wpad.shape)
        
#         for i in range(labels_wpad.shape[0]):

#                 #print('pre', labels_wpad[i])

#                 labels_wpad[i][to_r[i] < 0.5] = -1
            
#                 #print('post', labels_wpad[i])
        
        # remove labels for specific classes
        for i in range(labels_wpad.shape[0]):
            labels_wpad[i][labels_wpad[i]==0] = -1
            #labels_wpad[i][labels_wpad[i]==3] = -1


#         # remove all labels 
#         #print(labels_wpad.shape)
#         for i in range(labels_wpad.shape[0]):
#             y_prop = []
#             bot = labels_wpad.shape[1]
#             for k in range(self.hparams['n_total_classes']):
#                 top = (labels_wpad[i][labels_wpad[i] == k]).shape[0]
#                 y_prop.append(round(top/bot, 3))
            
#             print('y prop', y_prop)
            
#             labels_wpad[i]= -1

            
        

  
        outputs_dict = self.forward(markers_wpad, labels_wpad)

        # remove padding from supplied data
        if pad > 0:
            labels_strong = data['labels_strong'][:, pad:-pad, ...]
        else:
            labels_strong = data['labels_strong']
        
        # remove padding from model output
        if pad > 0:
            markers = markers_wpad[:, pad:-pad, ...]
            # remove padding from model output
            for key, val in outputs_dict.items():
                outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
        else:
            markers = markers_wpad
            
        # reshape everything to be (n_sequences * sequence_length, ...)
        N = markers.shape[0] * markers.shape[1]
        markers_rs = torch.reshape(markers, (N, markers.shape[-1]))
        labels_rs = torch.reshape(labels_strong, (N,))
        outputs_dict_rs = {}
        for key, val in outputs_dict.items():
            if isinstance(val, torch.Tensor):
                if len(val.shape) > 2:
                    outputs_dict_rs[key] = torch.reshape(val, (N, val.shape[-1]))
                else:
                    # when the input is (n_sequences, sequence_length), we want the output to be (n_sequences * sequence_length)
                    outputs_dict_rs[key] = torch.reshape(val, (N, 1))
            else:
                outputs_dict_rs[key] = val
                
        # pull out indices of labeled data for loss computation
        idxs_labeled = outputs_dict_rs['idxs_labeled'].squeeze(-1)

        # initialize loss to zero
        loss = 0
        loss_dict = {}
        
        # ----------------------------------------------
        # compute classification loss on labeled data
        # ----------------------------------------------
        if lambda_strong > 0:
            loss_strong = self.class_loss(outputs_dict_rs['y_logits'], labels_rs) * lambda_strong
            
            loss += loss_strong
            # log
            loss_dict['loss_classifier'] = loss_strong.item()
            #print('loss classifier: ', loss_strong.item())
        
        
        # ------------------------------------
        # compute reconstruction loss
        # ------------------------------------ 
        reconstruction = outputs_dict_rs['reconstruction']
        px_z_mean = reconstruction
        px_z_std = torch.ones_like(px_z_mean)* 0.3#1e-2

        px_z = Normal(px_z_mean, px_z_std)
        
        # diff between log prob of adding 1d Normals and MVN log prob
        k = markers_rs.shape[1]
        mvn_scalar =  (-1) * .5 * math.log(k)
        
        loss_reconstruction = torch.sum(px_z.log_prob(markers_rs), axis=1) 

        # average over batch dim
        loss_reconstruction = (torch.mean(loss_reconstruction, axis=0) + mvn_scalar )* (-1)
                              
        loss += loss_reconstruction
        # log
        loss_dict['loss_reconstruction'] = loss_reconstruction.item()
        #print('loss recon: ', loss_reconstruction)
        
        
        # -------------------------------------------------------
        # compute log p(y_t | y_(t-1), z_(t-1)) loss for labeled
        # -------------------------------------------------------
 
        # check for labeled data
        if idxs_labeled.sum() > 0:
            py_logits = outputs_dict_rs['py_logits'][idxs_labeled > 0, :]
            y_mixed = outputs_dict_rs['y_mixed']
            y_mixed_scalar = torch.argmax(y_mixed, axis=1)[idxs_labeled > 0]

            py = Categorical(logits=py_logits) 

            loss_py = torch.mean(py.log_prob(y_mixed_scalar), axis=0) * (-1)

            loss += loss_py

            # log
            loss_dict['loss_py'] = loss_py.item()
           
    
        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_t|y_(t-1), z_(t-1)) for unlabeled
        # ----------------------------------------------------------------------------------
        
        # check that we have unlabeled observatios
        if idxs_labeled.sum() < idxs_labeled.shape[0]:
   
            # create classifier q(y_t|x_t) 
            qy_logits = outputs_dict_rs['qy_x_logits'][idxs_labeled == 0, :]
            qy = Categorical(logits=qy_logits)

            # create prior p(y)
            #py_logits = torch.tensor([.1,.1, .4, .4])
            
            py_logits = outputs_dict_rs['py_logits'][idxs_labeled == 0, :]
            py = Categorical(logits=py_logits) 
            
            #py = Categorical(py_logits)

            loss_y_kl = torch.mean(kl_divergence(qy, py), axis=0) 
            loss_y_kl = loss_y_kl * kl_y_weight

            loss += loss_y_kl * 10

            loss_dict['loss_y_kl'] = loss_y_kl.item()
            #print('kl y loss', loss_y_kl)
            
            
        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_1) for all
        # ----------------------------------------------------------------------------------
        
        # check that we have unlabeled observatios
        if True:#idxs_labeled.sum() < idxs_labeled.shape[0]:
            print('')
            print('y pred prop- qy loss ')

            #y_sums = torch.argmax(outputs_dict_rs['qy_x_probs'][idxs_labeled == 0, :], axis=1)
            y_sums = torch.argmax(outputs_dict_rs['qy_x_probs'], axis=1)
            #print('ysum', y_sums)
            y_prop = []
            #bot = outputs_dict_rs['qy_x_probs'][idxs_labeled == 0, :].shape[0]
            bot = outputs_dict_rs['qy_x_probs'].shape[0]
            for k in range(self.hparams['n_total_classes']):
                top = (y_sums[y_sums == k]).shape[0]
                y_prop.append(round(top/bot, 3))

            print( y_prop)


            print('')
        
        
            # create classifier q(y_t|x_t) 
            qy_logits = outputs_dict_rs['qy_x_logits'] # (n_seq * seq_length, n_classes)
            qy = Categorical(logits=qy_logits)

            # create prior p(y)
            #py_logits = torch.tensor([.1,.1, .4, .4])

            #py_logits = torch.tensor([-1.09861, -1.09861, -1.09861, -1.09861])#outputs_dict_rs['py_logits'][0]

            # init biased py prior
    #         py_logits = torch.zeros_like(qy_logits)

    #         priors = torch.tensor([[1.09861, -2.3979, -2.3979, -2.3979],
    #                                [-2.3979, 1.09861, -2.3979, -2.3979],
    #                                [-2.3979, -2.3979, 1.09861, -2.3979],
    #                                [-2.3979, -2.3979, -2.3979, 1.09861]])

    #         rands = torch.randint(0, 4, [py_logits.shape[0]])
    #         for i in range(py_logits.shape[0]):
    #             py_logits[i] = priors[rands[i]]

            #ynp = [1-y for y in y_prop]
            #py_logits = torch.tensor(ynp)
            #py_logits = torch.log((py_logits/(1-py_logits)))
            py_logits = torch.tensor([.5, .167, .167, .167])
            py = Categorical(py_logits) 

            #py = Categorical(py_logits)
            print('var', torch.var(py_logits))
            loss_y_kl_uniform = torch.mean(kl_divergence(qy, py), axis=0) #* torch.var(py_logits)
            loss_y_kl_uniform = loss_y_kl_uniform * self.hparams['kl_y_weight_uniform']

            print('')
            print('kl weight', self.hparams['kl_y_weight_uniform'])
            print('kl lossl', loss_y_kl_uniform)

            loss += loss_y_kl_uniform

            loss_dict['loss_y_kl_uniform'] = loss_y_kl.item()
            #print('kl y loss', loss_y_kl)
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        # build MVN p(z|y)
        pz_mean = outputs_dict_rs['pz_mean']
        pz_std = outputs_dict_rs['pz_logvar'].exp().pow(0.5)
        pz = Normal(pz_mean, pz_std)
        
        # build MVN q(z|x,y)
        qz_mean = outputs_dict_rs['qz_xy_mean']
        qz_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)
        qz = Normal(qz_mean, qz_std)
        
        # sum over latent, mean over batch (mean?)
        
        loss_z_kl = torch.mean(torch.sum(kl_divergence(qz, pz), axis=1), axis=0)

        loss += kl_weight * loss_z_kl
        # log
        loss_dict['kl_weight'] = kl_weight
        loss_dict['loss_z_kl'] = loss_z_kl.item() * kl_weight 
        #print('KL weight: ', kl_weight)
       # print('loss KL (w/o weight): ', loss_kl)
            

        #print('TOTAL LOSS: ', loss)
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    

