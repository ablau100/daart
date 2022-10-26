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
        probs = [0.00001, .1, .4, .4, .1]#np.ones((self.hparams['n_total_classes'],))
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
        print('probs', probs)

        assert np.isclose([np.sum(np.array(probs))], [1])
        self.hparams['py_1_probs'] = probs
        
        # build updated label prior: p(y_t|y_(t-1), z_(t-1)), t > 1
        self.model['py_t_probs'] = self._build_linear(
            0, 'py_t_probs', self.hparams['n_hid_units'], (self.hparams['n_total_classes']*self.hparams['n_total_classes']))
        
        # build latent prior: p(z_1)
        self.hparams['pz_1_mean'] = 0
        self.hparams['pz_1_logvar'] = 0
        
        # build latent_generator: p(z_t|z_(t-1), y_t)
#         self.model['pz_t_mean'] = self._build_linear(
#             0, 'pz_t_mean', self.hparams['n_hid_units'], (self.hparams['n_hid_units']))
        
        self.model['pz_t_mean'] = self._build_linear(
            0, 'pz_t_mean', self.hparams['n_hid_units'], (self.hparams['n_hid_units']*self.hparams['n_total_classes']))
        
        self.model['pz_t_logvar'] = self._build_linear(
            0, 'pz_t_logvar', self.hparams['n_total_classes'], self.hparams['n_hid_units'])
        
        # build decoder: p(x_t|z_t)
        self.model['decoder'] = Module(
            self.hparams, 
            type='decoder',
            in_size=self.hparams['n_hid_units'],
            hid_size=self.hparams['n_hid_units'],
            out_size=self.hparams['input_size'])
        
    def __str__(self):
        """Pretty print generative model architecture."""
        
        format_str = 'Generative network architecture: %s\n' % self.hparams['backbone_generative'].upper()
        format_str += '------------------------\n'

        if 'decoder' in self.model:
            format_str += 'Decoder (p(x_t|z_t)):\n'
            for i, module in enumerate(self.model['decoder'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'py_t_probs' in self.model:
            format_str += 'p(y_t|y_(t-1), z_(t-1)):\n'
            for i, module in enumerate(self.model['py_t_probs']):
                format_str += str('    {}: {}\n'.format(i, module))
                format_str += str(' Weights py: {}\n'.format(module.weight))
            format_str += '\n'
                
        if 'pz_t_mean' in self.model:
            format_str += 'p(z_t|z_(t-1), y_t) mean:\n'
            for i, module in enumerate(self.model['pz_t_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                format_str += str(' Weights pz: {}\n'.format(module.weight))
            
                
#         if 'pz_t_logvar' in self.model:
#             format_str += 'p(z_t|z_(t-1), y_t) logvar:\n'
#             for i, module in enumerate(self.model['pz_t_logvar']):
#                 format_str += str('    {}: {}\n'.format(i, module))

        return format_str
    
    def forward(self, **kwargs): 
                
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
        py_t_probs = self.model['py_t_probs'](z_t_skip_last)
        #print('old shape py', py_t_probs.shape)
        py_t_probs = torch.reshape(py_t_probs, (py_t_probs.shape[0], py_t_probs.shape[1], y_dim,-1))
        #print('new shape py', py_t_probs.shape)
        # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from t=0
        py_indexer = y_t_skip_last.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_last.ndim, py_t_probs.shape[3])
        py_t_probs = torch.gather(py_t_probs, 2, py_indexer).squeeze(2)
        #print('indexed shape py', py_t_probs.shape)
        
        # concat py_1 with py_t, t > 1
        py_1_probs = torch.tensor(self.hparams['py_1_probs'])
        #print('py_1_probs', py_1_probs.shape)
        
        py_probs = torch.zeros(py_t_probs.shape[0], py_t_probs.shape[1]+1, py_t_probs.shape[2]).to(device=self.hparams['device'])
        py_probs[:, 1:, :] = py_1_probs#nn.Softmax(dim=2)(py_t_probs) # test using old prior
        py_probs[:, 0, :] = py_1_probs
        
        # push y_t and z_(t-1) through generative model to get parameters of p(z_t|z_(t-1), y_t)
        #pz_t_mean = self.model['pz_t_mean'](z_t_skip_last_with_y_dim)
        #print('old shape', pz_t_mean.shape, pz_t_mean.size)
        
        # index proper way
        # reshape output then index
        # first just try mean
        pz_t_mean = self.model['pz_t_mean'](z_t_skip_last)
        pz_t_mean = torch.reshape(pz_t_mean, (pz_t_mean.shape[0], pz_t_mean.shape[1], y_dim,-1))
        #print('new shape pz', pz_t_mean.shape)
        
        pz_t_logvar = self.model['pz_t_logvar'](y_t_skip_first)
        
        
        # index logvar by y_t - z_t starts from t=1, so our indexer y starts from y=1
        pz_indexer = y_t_skip_first.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_first.ndim, pz_t_mean.shape[3])
        pz_t_mean = torch.gather(pz_t_mean, 2, pz_indexer).squeeze(2)
        
        # concat pz_1 mean and logvar with pz_t mean and logvar, t > 1
        pz_1_mean = torch.tensor(self.hparams['pz_1_mean'])
        pz_mean = torch.zeros(pz_t_mean.shape[0], pz_t_mean.shape[1]+1, pz_t_mean.shape[2]).to(device=self.hparams['device'])
        
        pz_mean[:, 1:, :] = pz_t_mean
        pz_mean[:, 0, :] = pz_1_mean
        
        pz_1_logvar = torch.tensor(self.hparams['pz_1_logvar'])
        pz_logvar = torch.zeros(pz_t_mean.shape[0], pz_t_mean.shape[1]+1, pz_t_mean.shape[2]).to(device=self.hparams['device'])
        pz_logvar[:, 1:, :] = pz_t_logvar
        pz_logvar[:, 0, :] = pz_1_logvar
       
        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x_t|z_t)
        x_hat = self.model['decoder'](z_input)

        return {
            'py_probs': py_probs, # (n_sequences, sequence_length, n_total_classes)
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
        self.keys = ['py_probs','qy_x_probs','y_sample','qz_xy_mean','qz_xy_logvar'
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
        
        y_dim = self.hparams['n_total_classes']
        
        ##################
        # for t = 1
        ##################
        
        # sample y_1
        py_1_probs = torch.tensor(self.hparams['py_1_probs'])
        py_1 = Categorical(probs=py_1_probs)
        y_1 = py_1.sample()#.type(torch.FloatTensor)
        #print('un', y_1.unsqueeze(-1).shape)
        y_1 = MakeOneHot()(y_1.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
        y_samples.append(torch.squeeze(y_1))
        
        # sample z_1
        z_1 = torch.normal(0, torch.ones(self.hparams['n_hid_units']))
        z_samples.append(z_1)
        
        # sample x_1 | z_1
        
        ##################
        # for t = 2 to T
        ##################
        
        
        for t in range(1, num_samples):
            print('t', t)
            #print('y t-1', y_samples[t-1])
            # sample y_t | y_(t-1), z_(t-1)
            
            py_t_probs = self.model['py_t_probs'](z_samples[t-1])
            py_t_probs = torch.sigmoid(torch.reshape(py_t_probs, (y_dim,-1)))
            
            #print('py p sig', py_t_probs, py_t_probs.shape)
 
            # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from t=0
            y_ind = torch.argmax(y_samples[t-1])
            py_t_probs = py_t_probs[y_ind]
            
            #print('py p', py_t_probs, py_t_probs.shape)
            py_t = Categorical(probs=py_t_probs)
            y_t = py_t.sample()
            
            y_t = MakeOneHot()(y_t.unsqueeze(-1), self.hparams['n_total_classes']).type(torch.LongTensor)
            y_t = torch.squeeze(y_t).type(torch.LongTensor)
            #print('yt', y_t, y_t.shape)
            y_samples.append(y_t)

            # sample z_t | y_t, z_(t-1)
            pz_t_mean = self.model['pz_t_mean'](z_samples[t-1])
            pz_t_mean = torch.reshape(pz_t_mean, (y_dim,-1))
            
            z_ind = torch.argmax(y_samples[t])
            print('z ind', z_ind, z_ind.shape)
            #print('z ind un', z_ind.squeeze(-1), z_ind.squeeze(-1).shape)
            pz_t_mean = pz_t_mean[z_ind]

            pz_t_logvar = self.model['pz_t_logvar'](y_samples[t].type(torch.FloatTensor))
            
            z_t = torch.normal(pz_t_mean, pz_t_logvar)
            z_samples.append(z_t)
        
        
        # sample x_t | z_t
        
        return y_samples, z_samples
        
        
        
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
        
        # remove labels for walk/still
        for i in range(labels_wpad.shape[0]):
            labels_wpad[i][labels_wpad[i]==2] = 0
            labels_wpad[i][labels_wpad[i]==3] = 0
  
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
        idxs_labeled = outputs_dict_rs['idxs_labeled']

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
        px_z_std = torch.ones_like(px_z_mean)

        px_z = Normal(px_z_mean, px_z_std)
        
        # diff bwetween log prob of adding 1d Normals and MVN log prob
        k = markers_rs.shape[1]
        mvn_scalar =  (-1) * .5 * math.log(k)
        
        loss_reconstruction = torch.sum(px_z.log_prob(markers_rs), axis=1) 

        # average over batch dim
        loss_reconstruction = (torch.mean(loss_reconstruction, axis=0) + mvn_scalar )* (-1)
                              
        loss += loss_reconstruction
        # log
        loss_dict['loss_reconstruction'] = loss_reconstruction.item()
        #print('loss recon: ', loss_reconstruction)
    
    
        # ------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_t|y_(t-1), z_(t-1))
        # ------------------------------------------------------------------ 
        # create classifier q(y_t|x_t)
        #qy_x_logits = 
        qy_probs = outputs_dict_rs['qy_x_probs']
        qy = Categorical(probs=qy_probs)
        
        # create prior p(y)
        py_probs = outputs_dict_rs['py_probs']#.to(device=self.hparams.get('device'))
        py = Categorical(probs=py_probs) 

        loss_y_kl = torch.mean(kl_divergence(qy, py), axis=0) 
        #print('loss_y_kl w/o weight: ', loss_y_kl)
        loss_y_kl = loss_y_kl * kl_y_weight
        #print('loss_y_kl with weight: ', loss_y_kl)
        loss += loss_y_kl
  
        loss_dict['loss_y_kl'] = loss_y_kl.item()
        
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
    

