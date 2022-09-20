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
    
    def __init__(self, hparams):
        """
        
        Parameters
        ----------
        hparams : dict
            
        """
        super().__init__()
        self.hparams = hparams
        self.model = hparams['model']
    
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
        probs = np.ones((self.hparams['n_total_classes'],))
        background_prob = 0.01
        probs[0] = background_prob
        
        new_classes_indexes = [1,2]
        
        for i in range(self.hparams['n_total_classes']):
            if i in new_classes_indexes:
                probs[i] = (1-background_prob) * .7 * (1/len(new_classes_indexes))
            elif i > 0:
                probs[i] = (1-background_prob) * .3 * (1/(self.hparams['n_observed_classes']-1))

#         probs[:self.hparams['output_size']] /= (self.hparams['output_size'] * 2)
#         probs[self.hparams['output_size']:] /= (self.hparams['n_aug_classes'] * 2)
        print('probs', probs)

        assert np.isclose([np.sum(np.array(probs))], [1])
        self.hparams['py_1_probs'] = probs
        
        # build updated label prior: p(y_t|y_(t-1), z_(t-1)), t > 1
        # CHECK w MATT
        self.model['py_t_probs'] = self._build_linear(
            0, 'py_t_probs', self.hparams['n_hid_units'], self.hparams['n_total_classes'])
        
        # build latent prior: p(z_1)
        self.hparams['pz_1_mean'] = 0
        self.hparams['pz_1_logvar'] = 0
        
        # build latent_generator: p(z_t|z_(t-1), y_t)
        self.model['pz_t_mean'] = self._build_linear(
            0, 'pz_t_mean', self.hparams['n_hid_units'], self.hparams['n_hid_units'])
        
        self.model['pz_t_logvar'] = Module(
            self.hparams, 
            in_size=self.hparams['n_total_classes'],
            hid_size=self.hparams['n_hid_units'],
            out_size=self.hparams['n_hid_units'])
        
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
            format_str += '\n'
                
        if 'pz_t_mean' in self.model:
            format_str += 'p(z_t|z_(t-1), y_t) mean:\n'
            for i, module in enumerate(self.model['pz_t_mean']):
                format_str += str('    {}: {}\n'.format(i, module))
                
        if 'pz_t_logvar' in self.model:
            format_str += 'p(z_t|z_(t-1), y_t) logvar:\n'
            for i, module in enumerate(self.model['pz_t_logvar']):
                format_str += str('    {}: {}\n'.format(i, module))

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
        
        # push y_(t-1) and z_(t-1) through generative model to get parameters of p(y_t|y_(t-1), z_(t-1))
        py_t_probs = self.model['py_t_probs'](z_t_skip_last_with_y_dim)
        
        # index probs by y_(t-1) - y_t starts from t=1, so our indexer y starts from y=0
        py_indexer = y_t_skip_last.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_last.ndim, py_t_probs.shape[3])
        py_t_probs = torch.gather(py_t_probs, 2, py_indexer).squeeze(2)
        
        # push y_t and z_(t-1) through generative model to get parameters of p(z_t|z_(t-1), y_t)
        pz_t_mean = self.model['pz_t_mean'](z_t_skip_last_with_y_dim)
        pz_t_logvar = self.model['pz_t_logvar'](y_t_skip_first)
        
        # index logvar by y_t - z_t starts from t=1, so our indexer y starts from y=1
        pz_indexer = y_t_skip_first.argmax(2,True).unsqueeze(-1).expand(*(-1,)*y_t_skip_first.ndim, pz_t_mean.shape[3])
        pz_t_mean = torch.gather(pz_t_mean, 2, pz_indexer).squeeze(2)
        
        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x_t|z_t)
        x_hat = self.model['decoder'](z_input)

        return {
            'py1_probs': self.hparams['py_1_probs'], # (n_total_classes)
            'py_t_probs': py_t_probs, # (n_sequences, sequence_length, n_total_classes)
            'pz_t_mean': pz_t_mean,  # (n_sequences, sequence_length, embedding_dim)
            'pz_t_logvar': pz_t_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'pz_1_mean': self.hparams['pz_1_mean'],  # (embedding_dim)
            'pz_1_logvar': self.hparams['pz_1_logvar'],  # (embedding_dim)
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

        # model dict will contain some or all of the following components:
        # - classifier: q(y|x) [weighted by hparams['lambda_strong'] on labeled data]
        # - encoder: q(z|x,y)
        # - decoder: p(x|z)
        # - latent_generator: p(z|y)

        self.model = nn.ModuleDict()
        hparams['model'] = self.model
        
        self.inference = BaseInference(hparams)
        self.generative = RSLDSGenerative(hparams)
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        ignore_index = hparams.get('ignore_class', 0)
        self.class_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        
        # MSE loss for reconstruction
        self.recon_loss = nn.MSELoss(reduction='mean')
        
        # maybe init z sample array and y sample array here?
        self.z_samples_array = []
        self.y_mixed_array = []
        
    def __str__(self):
        """Pretty print model architecture."""

        format_str = '\n\nInference model\n'
        format_str += self.inference.__str__()

        format_str += '------------------------\n'

        format_str += '\n\nGenerative model\n'
        format_str += self.generative.__str__()

        return format_str
        
        
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
        
        # append new z sample and y mixed
        self.z_samples_array.append(inf_outputs['z_xy_sample'])
        self.y_mixed_array.append(inf_outputs['y_mixed'])
        
        # generative input
        gen_inputs = {
           'y_sample': inf_outputs['y_mixed'],
           'z_sample': inf_outputs['z_xy_sample'], 
        }
        
        gen_outputs = self.generative(**gen_inputs)

        # merge the two outputs
        output_dict = {**inf_outputs, **gen_outputs}

        return output_dict
    
    
    def predict_labels(self, data_generator, return_scores=False, remove_pad=True, mode='eval'):
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
        if mode == 'eval':
            self.eval()
        elif mode == 'train':
            self.train()
        else:
            raise NotImplementedError(
                'must choose mode="eval" or mode="train", not mode="%s"' % mode)

        pad = self.hparams.get('sequence_pad', 0)
        softmax = nn.Softmax(dim=1)

        # initialize outputs dict
        keys = ['y_logits','qy_x_probs','y_sample','qz_xy_mean','qz_xy_logvar'
                ,'pz_y_mean','pz_y_logvar','reconstruction']
        
        results_dict = {}
        
        results_dict['markers'] = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
                results_dict['markers'][sess] = [np.array([]) for _ in range(dataset.n_sequences)]

        
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
                    
                outputs_dict = self.forward(data['markers'], data['labels_strong'])
               # print('data 1', data['markers'].shape)
                # remove padding if necessary
                if pad > 0 and remove_pad:
                    for key, val in outputs_dict.items():
                        outputs_dict[key] = val[:, pad:-pad] if val is not None else None
                    data['markers'] = data['markers'][:, pad:-pad] 
                    #print('data 2', data['markers'].shape)
                # loop over sequences in batch
                for s, sess in enumerate(sess_list):
                    batch_idx = data['batch_idx'][s].item()
                    results_dict['markers'][sess][batch_idx] = \
                    data['markers'][s].cpu().detach().numpy()
                    for key in keys:
                        
                        # push through log-softmax, since this is included in the loss and not model
                        results_dict[key][sess][batch_idx] = \
                        outputs_dict[key][s].cpu().detach().numpy()
                        #softmax(outputs_dict[key][s]).cpu().detach().numpy()
                    
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
            labels_wpad[i][labels_wpad[i]==1] = 0
            labels_wpad[i][labels_wpad[i]==2] = 0
  
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
           # print('loss classifier: ', loss_strong.item())
        
        
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
       # print('loss recon: ', loss_reconstruction)
    
    
        # ----------------------------------------------
        # compute kl loss between q(y|x) and p(y)
        # ----------------------------------------------  
        # create classifier q(y|x)
        qy_x_probs = outputs_dict_rs['qy_x_probs']
        qy_x = Categorical(probs=qy_x_probs)
        
        # create prior p(y)
        py_probs = torch.tensor(self.hparams['py_probs']).to(device=self.hparams.get('device'))
        py = Categorical(probs=py_probs) 

        loss_y_kl = torch.mean(kl_divergence(qy_x, py), axis=0) 
        #print('loss_y_kl w/o weight: ', loss_y_kl)
        loss_y_kl = loss_y_kl * kl_y_weight
        #print('loss_y_kl with weight: ', loss_y_kl)
        loss += loss_y_kl
  
        loss_dict['loss_y_kl'] = loss_y_kl.item()
 
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        # build MVN p(z|y)
        pz_y_mean = outputs_dict_rs['pz_y_mean']
        pz_y_std = outputs_dict_rs['pz_y_logvar'].exp().pow(0.5)
        pz_y = Normal(pz_y_mean, pz_y_std)
        
        # build MVN q(z|x,y)
        qz_xy_mean = outputs_dict_rs['qz_xy_mean']
        qz_xy_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)
        qz_xy = Normal(qz_xy_mean, qz_xy_std)
        
        # sum over latent, mean over batch (mean?)
        
        loss_kl = torch.mean(torch.sum(kl_divergence(qz_xy, pz_y), axis=1), axis=0)

        loss += kl_weight * loss_kl
        # log
        loss_dict['kl_weight'] = kl_weight
        loss_dict['loss_kl'] = loss_kl.item() * kl_weight 
        #print('KL weight: ', kl_weight)
       # print('loss KL (w/o weight): ', loss_kl)
            

        #print('TOTAL LOSS: ', loss)
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    

