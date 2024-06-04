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
from daart.losses import FocalLoss
from daart.models.base import BaseModel, BaseInference, BaseGenerative, reparameterize_gaussian, get_activation_func_from_str
from daart.transforms import MakeOneHot
    
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from daart.models.gmdgmm import GMDGMMInference, GMDGMMGenerative





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
        #hparams['model'] = self.model
        
        self.keys = ['qy_x_probs']#, 'qz_xy_mean', 'py_logits', 'py_probs', 'reconstruction']
        
        self.inference = GMDGMMInference(hparams, self.model)
        self.generative = GMDGMMGenerative(hparams, self.model)
        self.build_model()

         # label loss based on cross entropy; don't compute gradient when target = 0
        ignore_index = hparams.get('ignore_class', 0)
        weight = hparams.get('alpha', None)
        if weight is not None:
            weight = torch.tensor(weight)
        focal_loss = self.hparams.get('focal_loss', False)
        if focal_loss:
            self.class_loss = FocalLoss(gamma=self.hparams['gamma'], alpha=weight, ignore_index=ignore_index)
        else:
            self.class_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')
            
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
        
        gen_inputs = {
           'y_probs': inf_outputs['qy_x_probs'],
           'z_sample': inf_outputs['z_xy_sample'], 
           'idxs_labeled': inf_outputs['idxs_labeled']
        }
 
        gen_outputs = self.generative(**gen_inputs)

        # merge the two outputs
        output_dict = {**inf_outputs, **gen_outputs}

        return output_dict
    
    def get_expectation(self, values, probs, axis=1):
        expectation = torch.sum(values*probs, axis=axis)
        return expectation
    
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
#         keys = ['y_logits','qy_x_probs','y_sample','qz_xy_mean','qz_xy_logvar'
#                 ,'pz_y_mean','pz_y_logvar','reconstruction']

        keys= self.keys
        
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
        y_dim = self.hparams['n_total_classes']
        device = self.hparams['device']
        kl_y_weight = self.hparams.get('kl_y_weight', 100)
        lambda_strong = self.hparams.get('lambda_strong', 1)
        ignore_class = self.hparams.get('ignore_class', 0)
        kl_z_weight = self.hparams.get('kl_z_weight', 1)
        ann_weight = self.hparams.get('ann_weight', 1)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        labels_wpad = data['labels_strong']
        
        outputs_dict = self.forward(markers_wpad, labels_wpad)

        # remove padding from supplied data
        # remove padding from supplied data
        if pad > 0:
            labels_strong = data['labels_strong'][:, pad:-pad, ...]
        else:
            labels_strong = data['labels_strong']
        
        # remove padding from model output
        y_dim = self.hparams['n_total_classes']
        
        if pad > 0:
            markers = markers_wpad[:, pad:-pad, ...]
            # remove padding from model output
            for key, val in outputs_dict.items():
                #print('key', key, 'val ', val.shape)
                if key not in ['qz_xy_mean', 'qz_xy_logvar', 'z_xy_sample', 'pz_mean', 'pz_logvar', 'reconstruction']:
                    #val.shape[0] != y_dim or key == 'idxs_labeled':
                    outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
                else:
                    outputs_dict[key] = val[:, :, pad:-pad, ...] if val is not None else None
        else:
            markers = markers_wpad
            
        # reshape everything to be (n_sequences * sequence_length, ...)
        N = markers.shape[0] * markers.shape[1]
        markers_rs = torch.reshape(markers, (N, markers.shape[-1]))
        labels_rs = torch.reshape(labels_strong, (N,))
        
        outputs_dict_rs = {}
      
        for key, val in outputs_dict.items():
            
            if isinstance(val, torch.Tensor):
                
                if key in ['qz_xy_mean', 'qz_xy_logvar', 'z_xy_sample', 'pz_mean', 'pz_logvar', 'reconstruction']:
                    shape = (y_dim, N) + tuple(val.shape[3:])
                    outputs_dict_rs[key] = torch.reshape(val, shape) 
                elif len(val.shape) > 2:
                    shape = (N, ) + tuple(val.shape[2:])
                    outputs_dict_rs[key] = torch.reshape(val, shape)
                else:
                    # when the input is (n_sequences, sequence_length), we want the output to be (n_sequences * sequence_length)
                    outputs_dict_rs[key] = torch.reshape(val, (N, 1))
            else:
                outputs_dict_rs[key] = val
         
        # pull out indices of labeled data for loss computation
        idxs_labeled = outputs_dict_rs['idxs_labeled'].squeeze(-1)
        
        
        
#         N = markers.shape[0] * markers.shape[1]
#         markers_rs = torch.reshape(markers, (N, markers.shape[-1]))
#         labels_rs = torch.reshape(labels_strong, (N,))
#         outputs_dict_rs = {}
#         for key, val in outputs_dict.items():
#             if isinstance(val, torch.Tensor):
#                 if len(val.shape) > 2:
#                     outputs_dict_rs[key] = torch.reshape(val, (N, val.shape[-1]))
#                 else:
#                     # when the input is (n_sequences, sequence_length), we want the output to be (n_sequences * sequence_length)
#                     outputs_dict_rs[key] = torch.reshape(val, (N, 1))
#             else:
#                 outputs_dict_rs[key] = val
                
#         # pull out indices of labeled data for loss computation
#         idxs_labeled = outputs_dict_rs['idxs_labeled']

        # initialize loss to zero
        loss = 0
        loss_dict = {}
        qy_e_probs = outputs_dict_rs['qy_e_probs']
        
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
        loss_y_kl = loss_y_kl * kl_y_weight * ann_weight
        loss += loss_y_kl
  
        loss_dict['loss_y_kl'] = loss_y_kl.item()

        # ----------------------------------------------
        # compute classification loss on labeled data
        # ----------------------------------------------
        if lambda_strong > 0:
            loss_strong = self.class_loss(outputs_dict_rs['qy_x_logits'], labels_rs) * lambda_strong
            
            loss += loss_strong
            # log
            loss_dict['loss_classifier'] = loss_strong.item()
           # print('loss classifier: ', loss_strong.item())
            
        # ------------------------------------
        # compute reconstruction loss
        # ------------------------------------ 
        reconstruction = outputs_dict_rs['reconstruction'] # (y_dim, N, n_markers)
        px_z_mean = reconstruction
        px_z_std = torch.ones_like(px_z_mean)* .5

        px_z = Normal(px_z_mean, px_z_std)
        
        # diff between log prob of adding 1d Normals and MVN log prob
        k = markers_rs.shape[1]
        mvn_scalar =  (-1) * .5 * math.log(k)
        
        # sum over marker dim
        loss_reconstruction = torch.sum(px_z.log_prob(markers_rs), axis=2) 
        
        # take transpose to get shape N * y_dim
        loss_reconstruction = torch.transpose(loss_reconstruction, 0, 1)
        
        # get expectation to get shape N
        loss_reconstruction = self.get_expectation(loss_reconstruction, qy_e_probs)

        # average over batch dim
        loss_reconstruction = (torch.mean(loss_reconstruction, axis=0) + mvn_scalar )* (-1) 
                              
        loss += loss_reconstruction
        # log
        loss_dict['loss_reconstruction'] = loss_reconstruction.item()
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        # build MVN q(z|x,y)
        qz_mean = outputs_dict_rs['qz_xy_mean'].to(device)  # qz_mean shape (y_dim, N, n_latents)
        qz_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5).to(device)
        qz = Normal(qz_mean, qz_std)
        
        # first dim is y_t
        pz_mean = outputs_dict_rs['pz_mean'].to(device) # (n_total_classes, N, embedding_dim) 
        
        pz_std = outputs_dict_rs['pz_logvar'].exp().pow(0.5).to(device) # (n_total_classes, N, embedding_dim)

        loss_z_kl = 0
        #for k_prime in range(y_dim): # k_prime loops over y_(t-1)

        # build MVN p(z|y)
        # pz_mean shape (y_dim, N, n_latents)) 
        pz = Normal(pz_mean, pz_std)

        # compute kl; final shape (N, y_dim)
        kl_temp = torch.transpose(torch.sum(kl_divergence(qz, pz), axis=2), 1, 0)

        # expectation over y_t
        loss_z_kl_temp = torch.sum(
            outputs_dict_rs['qy_e_probs'] * kl_temp, axis=1)

        # mean over batch
        loss_z_kl += torch.mean(loss_z_kl_temp, axis=0)

        loss += kl_z_weight * loss_z_kl * ann_weight
        # log
        #loss_dict['kl_weight'] = kl_weight
        loss_dict['loss_z_kl'] = loss_z_kl.item() * kl_z_weight * ann_weight
            

        #print('TOTAL LOSS: ', loss)
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    

