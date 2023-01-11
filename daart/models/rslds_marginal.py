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
from daart.models.rsldsm import RSLDSMInference, RSLDSMGenerative
from daart.transforms import MakeOneHot
    
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence



class RSLDSM(BaseModel):
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
        if len(hparams['label_names']) == 1:
            hparams['label_names'] = hparams['label_names'][0]
            
        self.hparams = hparams
        
        self.keys = ['qy_x_probs']
        
#         self.keys = ['qy_x_probs','qz_xy_mean','qz_xy_logvar'
#                 ,'pz_mean','pz_logvar','reconstruction']

        # model dict will contain some or all of the following components:
        # - classifier: q(y|x) [weighted by hparams['lambda_strong'] on labeled data]
        # - encoder: q(z|x,y)
        # - decoder: p(x|z)
        # - latent_generator: p(z|y)

        self.model = nn.ModuleDict()
#         self.inference = BaseInference(hparams, self.model)
#         self.generative = RSLDSGenerative(hparams, self.model)
        
        self.inference = RSLDSMInference(hparams, self.model)
        self.generative = RSLDSMGenerative(hparams, self.model)
        
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
        ignore_class = self.hparams.get('ignore_class', 0)
       # print('ls: ', lambda_strong)
        kl_weight = self.hparams.get('kl_weight', 1)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        labels_wpad = data['labels_strong']
        
        # remove half of labels randomly by class     
        to_r = np.random.uniform(0,1,labels_wpad.shape)
        for i in range(labels_wpad.shape[0]):
                labels_wpad[i][to_r[i] < 0.5] = -1

        
#         # remove labels for specific classes
#         for i in range(labels_wpad.shape[0]):
#             labels_wpad[i][labels_wpad[i]==0] = -1
#             labels_wpad[i][labels_wpad[i]==3] = -1


        # remove all labels 
#         for i in range(labels_wpad.shape[0]):
#             y_prop = []
#             bot = labels_wpad.shape[1]
#             for k in range(self.hparams['n_total_classes']):
#                 top = (labels_wpad[i][labels_wpad[i] == k]).shape[0]
#                 y_prop.append(round(top/bot, 3))
            
#             #print('y prop', y_prop)
            
#             labels_wpad[i]= -1


        outputs_dict = self.forward(markers_wpad, labels_wpad)
        #print('ot log p ', outputs_dict['pz_logvar'].shape)

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
                if val.shape[0] != y_dim:
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
        
        # this will be an issue when ydim = batch size
        
        
        for key, val in outputs_dict.items():
            
            if isinstance(val, torch.Tensor):
                if val.shape[0] == y_dim:
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

        # initialize loss to zero
        #torch.autograd.set_detect_anomaly(True)
        loss = 0
        loss_dict = {}
        qy_e_probs = outputs_dict_rs['qy_e_probs']
        
                
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
        #print('loss recon: ', loss_reconstruction)
        
        # -------------------------------------------------------
        # compute log p(y_t | y_(t-1), z_(t-1)) loss for labeled
        # -------------------------------------------------------
        
        
        if idxs_labeled.sum() > 0:
            
            py_logits = outputs_dict_rs['py_logits']  # (n_total_classes, N, n_total_classes)
            y_input = labels_rs
            y_input[labels_rs == ignore_class] = 1  # shape (N,)

            logpy = []
            for k in range(y_dim):
                py = Categorical(logits=py_logits[k])

                # evaluate prob of true labels
                logpy.append(py.log_prob(y_input))
                
            logpy = torch.stack(logpy, dim=0) # (y_dim, N)
            
            # marginalize over conditioning var y_{t-1}
            e_py = self.get_expectation(logpy[:, 1:].transpose(0,1), qy_e_probs[:-1, :], axis=1)
            
            if labels_rs[0] != ignore_class:
                e_py = torch.cat([logpy[labels_rs[0], 0].unsqueeze(0), e_py], dim=0)              
            else:
                e_py = torch.cat([torch.zeros(1), e_py], dim=0)
            
            # subselect results from labeled data
            e_py_labeled = e_py[labels_rs != ignore_class]
            
            loss_py = torch.mean(e_py_labeled, axis=0) * (-1)
            #print('loss_py', loss_py)
            
        # check for labeled data
#         if idxs_labeled.sum() > 0:
            
#             py_logits = outputs_dict_rs['py_logits'][:, idxs_labeled > 0, :]
#             y_gt_scalar = labels_rs[labels_rs != ignore_class]

#             py_logits_labeled = torch.zeros((py_logits.shape[1], py_logits.shape[2]))
#             for i, k in enumerate(y_gt_scalar):
#                 py_logits_labeled[i] = py_logits[k, i, :]

#             py = Categorical(logits=py_logits_labeled) 

#             loss_py = torch.mean(py.log_prob(y_gt_scalar), axis=0) * (-1)

            loss += loss_py

            # log
            loss_dict['loss_py'] = loss_py.item()


        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_t|y_(t-1), z_(t-1)) for unlabeled
        # ----------------------------------------------------------------------------------
        
         # 'py_logits': py_logits, # (n_total_classes, N, n_total_classes)
         #     'y_logits': y_logits, # (N, n_classes)
        
        
        
        
        # check that we have unlabeled observatios
        if idxs_labeled.sum() < idxs_labeled.shape[0]:
            
            # create classifier q(y_t|x_t) 
            qy_logits = outputs_dict_rs['qy_x_logits'].unsqueeze(0)  # (1, N, n_classes)
            qy = Categorical(logits=qy_logits)

            # create prior p(y)
            py_logits = outputs_dict_rs['py_logits']  # (n_classes, N, n_classes)
            py = Categorical(logits=py_logits)

            loss_y_kl = torch.transpose(kl_divergence(qy, py), 0, 1)

            # get expectation
            expectation_probs = torch.vstack((torch.ones(y_dim)/y_dim, qy_e_probs[:-1]))

            # compute expectation across all timepoints, labeled and unlabeled
            loss_y_kl = self.get_expectation(loss_y_kl, expectation_probs)

            # subselect unlabeled data, mean over batch dim
            loss_y_kl = torch.mean(loss_y_kl[idxs_labeled == 0], axis=0) * kl_y_weight
            #loss_y_kl *= kl_y_weight
            loss += loss_y_kl * 10
            loss_dict['loss_y_kl'] = loss_y_kl.item()

            
#             # create classifier q(y_t|x_t) 
#             qy_logits = outputs_dict_rs['qy_x_logits'][idxs_labeled == 0, :]
#             qy = Categorical(logits=qy_logits)

#             # create prior p(y)
#             py_logits = outputs_dict_rs['py_logits'][:, idxs_labeled == 0, :]
#             py = Categorical(logits=py_logits) 
#             loss_y_kl = torch.transpose(kl_divergence(qy, py), 0, 1)
         
#             # get expectation
#             expectation_probs = torch.vstack((torch.ones(y_dim)/y_dim, qy_e_probs[idxs_labeled == 0][:-1]))
            
#             loss_y_kl = self.get_expectation(loss_y_kl, qy_e_probs[idxs_labeled == 0])           
#             # mean over batch dim
#             loss_y_kl = torch.mean(loss_y_kl, axis=0)           
#             loss_y_kl = loss_y_kl * kl_y_weight
#             loss += loss_y_kl * 10
#             loss_dict['loss_y_kl'] = loss_y_kl.item()
#             #print('kl y loss', loss_y_kl)
         
        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_1) for all
        # ----------------------------------------------------------------------------------
        
#         y_sums = torch.argmax(outputs_dict_rs['qy_x_probs'], axis=1)
#         y_prop = []
#         bot = outputs_dict_rs['qy_x_probs'].shape[0]
#         for k in range(self.hparams['n_total_classes']):
#             top = (y_sums[y_sums == k]).shape[0]
#             y_prop.append(round(top/bot, 3))

#         # create classifier q(y_t|x_t) 
#         qy_logits = outputs_dict_rs['qy_x_logits'] # (n_seq * seq_length, n_classes)
#         qy = Categorical(logits=qy_logits)

#         # create prior p(y)
#         #py_logits = torch.tensor([.1,.1, .4, .4])

#         #py_logits = torch.tensor([-1.09861, -1.09861, -1.09861, -1.09861])#outputs_dict_rs['py_logits'][0]

#         # init biased py prior
#         # py_logits = torch.zeros_like(qy_logits)

#         #py_logits = torch.tensor([1-y for y in y_prop])
#         #py_logits = torch.tensor(ynp)
#         #py_logits = torch.log((py_logits/(1-py_logits)))

#         #py_logits =  torch.tensor([.25, .25, .25, .25])
#         #py_logits =  torch.tensor([.3, .2, .2, .3])
#         #py = Categorical(py_logits) 

#         loss_y_kl_uniform = torch.mean(kl_divergence(qy, py), axis=0) #* torch.var(py_logits)
#         loss_y_kl_uniform = loss_y_kl_uniform * self.hparams['kl_y_weight_uniform']

#         loss += loss_y_kl_uniform

#         loss_dict['loss_y_kl_uniform'] = loss_y_kl_uniform.item()
#         #print('kl y loss', loss_y_kl)
        
        # ----------------------------------------
        # compute kl divergence b/t qz_xy and pz_y
        # ----------------------------------------   
        
        # build MVN q(z|x,y)
        qz_mean = outputs_dict_rs['qz_xy_mean']  # qz_mean shape (y_dim, N, n_latents)
        qz_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)
        qz = Normal(qz_mean, qz_std)

        loss_z_kl = 0
        for k_prime in range(y_dim):

            # build MVN p(z|y)
            # pz_mean shape (y_dim, N, n_latents))
            pz_mean = torch.transpose(outputs_dict_rs['pz_mean'][k_prime], 1, 0)
            pz_std = torch.transpose(outputs_dict_rs['pz_logvar'], 1, 0).exp().pow(0.5)
            pz = Normal(pz_mean, pz_std)

            # compute kl; final shape (N, y_dim)
            kl_temp = torch.transpose(torch.sum(kl_divergence(qz, pz), axis=2), 1, 0)

            # expectation over y_t
            loss_z_kl_temp = torch.sum(
                outputs_dict_rs['qy_e_probs'][1:] * kl_temp[1:], axis=1)
            # multiply by probability for y_{t-1}
            loss_z_kl_temp *= outputs_dict_rs['qy_e_probs'][:1, k_prime]

            # mean over batch
            loss_z_kl += torch.mean(loss_z_kl_temp, axis=0)
        
#         # loop over k-1
#         loss_z_kl = 0
#         for k_prime in range(y_dim):
#             for k in range(y_dim):
        
#                 # build MVN p(z|y)
#                 pz_mean = outputs_dict_rs['pz_mean'][k_prime, :, k, :]
#                 #print('pzm', pz_mean.shape)
#                 pz_std = outputs_dict_rs['pz_logvar'].exp().pow(0.5)[:, k, :]

#                 #print('pm', pz_mean.shape)
#                 #print('pstd', pz_std.shape)
#                 pz = Normal(pz_mean, pz_std)

#                 # build MVN q(z|x,y)
#                 qz_mean = outputs_dict_rs['qz_xy_mean'][k]
#                 qz_std = outputs_dict_rs['qz_xy_logvar'].exp().pow(0.5)[k]
#                 qz = Normal(qz_mean, qz_std)

#                 #print('qm', qz_mean.shape)
#                 #print('qstd', qz_std.shape)
                
#                 # sum over latent, mean over batch
#                 loss_z_kl_temp = torch.sum(kl_divergence(qz, pz), axis=1) 
#                 loss_z_kl += torch.mean(loss_z_kl_temp, axis=0) 
#                 #print('loss_z_kl', loss_z_kl.shape)

        loss += kl_weight * loss_z_kl
        # log
        loss_dict['kl_weight'] = kl_weight
        loss_dict['loss_z_kl'] = loss_z_kl.item() * kl_weight 
       
        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict
    
    
    
    

