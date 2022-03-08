"""Base models/modules in PyTorch."""

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
from daart.models.base import BaseModel, reparameterize_gaussian, get_activation_func_from_str
from daart.transforms import MakeOneHot
    
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

class GMDGM(BaseModel):
    """Gaussian Mixture Deep Generative Model.
    
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
            - output_size (int): number of classes
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

        # model dict will contain some or all of the following components:
        # - classifier: q(y|x) [weighted by hparams['lambda_strong'] on labeled data]
        # - encoder: q(z|x,y)
        # - decoder: p(x|z)
        # - latent_generator: p(z|y)

        self.model = nn.ModuleDict()
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        ignore_index = hparams.get('ignore_class', 0)
        self.class_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        # this will turn into a log-likelihood calculation using \mu(z) as mean of normal
        # self.reconstruction_loss = nn.MSELoss(reduction='mean')
        
    def __str__(self):
        """Pretty print model architecture."""

        # list: encoder, decoder, qz_mean, qz_logvar, classifier,  
        
        # py, qy_x, encoder--, qz_xy_mean, qz_xy_logvar, 'pz_y_mean, 'pz_y_logvar, decoder--
        
        format_str = '\n%s architecture\n' % self.hparams['backbone'].upper()
        format_str += '------------------------\n'

        format_str += 'Encoder:\n'
        for i, module in enumerate(self.model['encoder'].model):
            format_str += str('    {}: {}\n'.format(i, module))
        format_str += '\n'

        if 'decoder' in self.model:
            format_str += 'Decoder:\n'
            for i, module in enumerate(self.model['decoder'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'py' in self.model:
            format_str += 'p(y):\n'
            for i, module in enumerate(self.model['py']):
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
                
        if 'pz_y_mean' in self.model:
            format_str += 'p(z|y) mean:\n'
            for i, module in enumerate(self.model['pz_y_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                
        if 'pz_y_logvar' in self.model:
            format_str += 'p(z|y) logvar:\n'
            for i, module in enumerate(self.model['pz_y_logvar']):
                format_str += str('    {}: {}\n'.format(i, module))

        return format_str
        
        
    def build_model(self):
        """Construct the model using hparams."""

        # set random seeds for control over model initialization
        rng_seed_model = self.hparams.get('rng_seed_model', 0)
        torch.manual_seed(rng_seed_model)
        np.random.seed(rng_seed_model)

        # select backbone network
        if self.hparams['backbone'].lower() == 'temporal-mlp':
            from daart.backbones.temporalmlp import TemporalMLP as Module
        elif self.hparams['backbone'].lower() == 'tcn':
            raise NotImplementedError('deprecated; use dtcn instead')
        elif self.hparams['backbone'].lower() == 'dtcn':
            from daart.backbones.tcn import DilatedTCN as Module
        elif self.hparams['backbone'].lower() in ['lstm', 'gru']:
            from daart.backbones.rnn import RNN as Module
        elif self.hparams['backbone'].lower() == 'tgm':
            raise NotImplementedError
            # from daart.models.tgm import TGM as Module
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone'])

        n_total_classes = self.hparams['output_size'] + self.hparams['n_aug_classes']
        self.hparams['n_total_classes'] = n_total_classes
        # build label prior: p(y)
        # prior prob for observed classes: 0.5 / n_observed_classes
        # prior prob for unobserved classes: 0.5 / n_aug_classes
        probs = 0.5 * np.ones((n_total_classes,))
        probs[:self.hparams['output_size']] /= self.hparams['output_size']
        probs[self.hparams['output_size']:] /= self.hparams['n_aug_classes']
       
        assert np.isclose([np.sum(probs)], [1])
        #self.model['py'] = Categorical(torch.tensor(probs))      
        
        # build classifier: q(y|x)
        self.model['qy_x'] = Module(
            self.hparams, 
            in_size=self.hparams['input_size'], 
            hid_size=self.hparams['n_hid_units'], 
            out_size=n_total_classes)
        
        self.hparams['qy_x_temperature'] = 1
        
        # build encoder: q(z|x,y)
        # for now we will concatenate x and y to infer z; perhaps in the future we
        # can try a lookup table?
        self.model['encoder'] = Module(
            self.hparams, 
            in_size=n_total_classes + self.hparams['input_size'],
            hid_size=self.hparams['n_hid_units'],
            out_size=self.hparams['n_hid_units'])
        self.model['qz_xy_mean'] = self._build_linear(
            global_layer_num=len(self.model['qy_x'].model), name='qz_xy_mean',
            in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])
        self.model['qz_xy_logvar'] = self._build_linear(
            global_layer_num=len(self.model['qy_x'].model), name='qz_xy_logvar',
            in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])
        
        # build latent_generator: p(z|y)
        # linear layer is essentially a lookup table of shape (n_hid_units, n_total_classes)
        self.model['pz_y_mean'] = self._build_linear(
            0, 'pz_y_mean', n_total_classes, self.hparams['n_hid_units'])
        self.model['pz_y_logvar'] = self._build_linear(
            0, 'pz_y_logvar', n_total_classes, self.hparams['n_hid_units'])
        
        # build decoder: p(x|z)
        self.model['decoder'] = Module(
            self.hparams, 
            in_size=self.hparams['n_hid_units'],
            hid_size=self.hparams['n_hid_units'],
            out_size=self.hparams['input_size'])
        
        self.hparams['kl_weight'] = 1  # weight in front of kl term; anneal this using callback

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
        # push inputs through classifier to get q(y|x)
        y_logits = self.model['qy_x'](x)
        
        # initialize and sample q(y|x) (should be a one-hot vector)
        y_probs = nn.Softmax(dim=2)(y_logits)
     
        qy_x = RelaxedOneHotCategorical(
            temperature=self.hparams['qy_x_temperature'], probs=y_probs)

        
        y_sample = qy_x.rsample()  # (n_sequences, sequence_length, n_total_classes)
        
        y_onehot = torch.zeros([y.shape[0], y.shape[1], self.hparams['n_total_classes']])
        for s in range(y.shape[0]):
            one_hot = MakeOneHot()(y[s], self.hparams['n_total_classes'])
            y_onehot[s] = one_hot

        # init y_mixed, which will contain true labels for labeled data, samples for unlabled data
        y_mixed = y_onehot.clone().detach()  # (n_sequences, sequence_length, n_total_classes)
        # loop over sequences in batch
        idxs_labeled = torch.zeros_like(y)
        for s in range(y_mixed.shape[0]):
            # for each sequence, update y_mixed with samples when true label is 0 
            # (i.e. no label)
            idxs_labeled[s] = y[s] != 0
            y_mixed[s, ~idxs_labeled[s], :] = y_sample[s, ~idxs_labeled[s]]
        
        # concatenate sample with input x
        # (n_sequences, sequence_length, n_total_classes))
        xy = torch.cat([x, y_mixed], dim=2)
        
        # push y through generative model to get parameters of p(z|y)
        pz_y_mean = self.model['pz_y_mean'](y_mixed)

        pz_y_logvar = self.model['pz_y_logvar'](y_mixed)

        
        # push [y, x] through encoder to get parameters of q(z|x,y)
        w = self.model['encoder'](xy)
        qz_xy_mean = self.model['qz_xy_mean'](w)
        qz_xy_logvar = self.model['qz_xy_logvar'](w)
        

        # init z_xy_sample 
        z_xy_sample = torch.zeros([y.shape[0], y.shape[1], self.hparams['n_hid_units']])
        
        for s in range(y.shape[0]):
            mean = qz_xy_mean[s]
            std = qz_xy_logvar[s].exp().pow(0.5)
            
            # sample with reparam trick
            sample = mean + torch.randn(std.shape) * std
            z_xy_sample[s] = sample

        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x|z)
        x_hat = self.model['decoder'](z_xy_sample)
        
        return {
            'y_logits': y_logits, # (n_sequences, sequence_length, n_classes)
            'y_probs': y_probs,  # (n_sequences, sequence_length, n_classes)
            'y_sample': y_sample,  # (n_sequences, sequence_length, n_classes)
            'qz_xy_mean': qz_xy_mean,  # (n_sequences, sequence_length, embedding_dim)
            'qz_xy_logvar': qz_xy_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'pz_y_mean': pz_y_mean,  # (n_sequences, sequence_length, embedding_dim)
            'pz_y_logvar': pz_y_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'reconstruction': x_hat,  # (n_sequences, sequence_length, n_markers)
            'idxs_labeled': idxs_labeled,  # (n_sequences, sequence_length)
        }
    
    
    def predict_labels(self, data_generator, return_scores=False, remove_pad=True):
        """
        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches
        return_scores : bool
            return scores before they've been passed through softmax
        remove_pad : bool
            remove batch padding from model outputs before returning
        Returns
        -------
        dict
            - 'predictions' (list of lists): first list is over datasets; second list is over
              batches in the dataset; each element is a numpy array of the label probability
              distribution
            - 'weak_labels' (list of lists): corresponding weak labels
            - 'labels' (list of lists): corresponding labels
        """
        self.eval()

        pad = self.hparams.get('sequence_pad', 0)


        # initialize outputs dict
        keys = ['y_logits','y_probs','y_sample','qz_xy_mean','qz_xy_logvar'
                ,'pz_y_mean','pz_y_logvar','reconstruction']
        
        results_dict = {}
        for key in keys:
            results_dict[key] = [[] for _ in range(data_generator.n_datasets)]

            for sess, dataset in enumerate(data_generator.datasets):
                results_dict[key][sess] = [np.array([]) for _ in range(dataset.n_sequences)]
                

        # partially fill container (gap trials will be included as nans)
        dtypes = ['train', 'val', 'test']
        for dtype in dtypes:
            data_generator.reset_iterators(dtype)
            for i in range(data_generator.n_tot_batches[dtype]):
                data, sess_list = data_generator.next_batch(dtype)
                outputs_dict = self.forward(data['markers'])
                # remove padding if necessary
                if pad > 0 and remove_pad:
                    for key, val in outputs_dict.items():
                        outputs_dict[key] = val[:, pad:-pad] if val is not None else None
                # loop over sequences in batch
                for s, sess in enumerate(sess_list):
                    batch_idx = data['batch_idx'][s].item()
                    for key in keys:
                        
                        # push through log-softmax, since this is included in the loss and not model
                        results_dict[key][sess][batch_idx] = \
                            softmax(outputs_dict[key][s]).cpu().detach().numpy()
                    
        return results_dict
    
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
        lambda_strong = self.hparams.get('lambda_strong', 1)
        kl_weight = self.hparams.get('kl_weight', 1)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        labels_wpad = data['labels_strong']
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
                    outputs_dict_rs[key] = torch.reshape(val, (N, -1))
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
            loss_strong = self.class_loss(outputs_dict_rs['y_logits'], labels_rs)
            loss += lambda_strong * loss_strong
            # log
            loss_dict['loss_classifier'] = loss_strong.item()
            
        # ------------------------------------
        # compute reconstruction loss
        # ------------------------------------ 
        reconstruction = outputs_dict_rs['reconstruction']
        if torch.isnan(outputs_dict_rs['reconstruction']).any():
            print('nan 432')
        px_z_mean = reconstruction
        px_z_std = torch.ones_like(px_z_mean)

        px_z = Normal(px_z_mean, px_z_std)
        
        loss_reconstruction = px_z.log_prob(markers_rs).sum()
        
        loss += loss_reconstruction
        # log
        loss_dict['loss_reconstruction'] = loss_reconstruction.item()
        print('loss recon: ', loss_reconstruction)
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        # built MVN p(z|y)
        pz_y_mean = outputs_dict_rs['pz_y_mean']
        pz_y_std = outputs_dict_rs['pz_y_logvar'].exp().pow(0.5)
        pz_y = Normal(pz_y_mean, pz_y_std)
        
        # built MVN q(z|x,y)
        qz_xy_mean = outputs_dict_rs['qz_xy_mean']
        qz_xy_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)
        

        qz_xy = Normal(qz_xy_mean, qz_xy_std)
        
        loss_kl = kl_divergence(qz_xy, pz_y).sum()
        
        loss -= kl_weight * loss_kl
        # log
        loss_dict['kl_weight'] = kl_weight
        loss_dict['loss_kl'] = loss_kl.item()
        print('KL weight: ', kl_weight)
        print('loss KL (w/o weight): ', loss_kl)
            
        # ---------------------------------------
        # entropy loss of qy_x on unlabeled data
        # ---------------------------------------
        loss_unlabeled = 0
        y_probs = outputs_dict_rs['y_probs'][~idxs_labeled]
        loss_entropy = Categorical(y_probs).entropy()
        
        loss_unlabeled -= loss_entropy.sum()
        print('entropy loss: ', loss_unlabeled)
        
        # log
        loss_dict['loss_unlabeled'] = loss_unlabeled.item()
        
        # adding unlabeled loss to total loss
        loss += loss_unlabeled
        print('TOTAL LOSS: ', loss)
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    