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

from daart.transforms import MakeOneHot
from torch.distributions import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

# to ignore imports for sphix-autoapidoc
__all__ = [
    'reparameterize_gaussian', 'get_activation_func_from_str', 'BaseModel', 'Segmenter',
    'Ensembler'
]


def reparameterize_gaussian(mu, logvar):
    """Sample from N(mu, var)

    Parameters
    ----------
    mu : torch.Tensor
        vector of mean parameters
    logvar : torch.Tensor
        vector of log variances; only mean field approximation is currently implemented

    Returns
    -------
    torch.Tensor
        sampled vector of shape (n_sequences, sequence_length, embedding_dim)

    """
    std = torch.exp(logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def get_activation_func_from_str(activation_str):

    if activation_str == 'linear':
        activation_func = None
    elif activation_str == 'relu':
        activation_func = nn.ReLU()
    elif activation_str == 'lrelu':
        activation_func = nn.LeakyReLU(0.05)
    elif activation_str == 'sigmoid':
        activation_func = nn.Sigmoid()
    elif activation_str == 'tanh':
        activation_func = nn.Tanh()
    else:
        raise ValueError('"%s" is an invalid activation function' % activation_str)

    return activation_func


class BaseModel(nn.Module):
    """Template for PyTorch models."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        """Pretty print model architecture."""
        raise NotImplementedError

    def build_model(self):
        """Build model from hparams."""
        raise NotImplementedError

    @staticmethod
    def _build_linear(global_layer_num, name, in_size, out_size):

        linear_layer = nn.Sequential()

        # add layer (cross entropy loss handles activation)
        layer = nn.Linear(in_features=in_size, out_features=out_size)
        layer_name = str('dense(%s)_layer_%02i' % (name, global_layer_num))
        linear_layer.add_module(layer_name, layer)

        return linear_layer

    @staticmethod
    def _build_mlp(
            global_layer_num, in_size, hid_size, out_size, n_hid_layers=1,
            activation='lrelu'):

        mlp = nn.Sequential()

        in_size_ = in_size

        # loop over hidden layers (0 layers <-> linear model)
        for i_layer in range(n_hid_layers + 1):

            if i_layer == n_hid_layers:
                out_size_ = out_size
            else:
                out_size_ = hid_size

            # add layer
            layer = nn.Linear(in_features=in_size_, out_features=out_size_)
            name = str('dense_layer_%02i' % global_layer_num)
            mlp.add_module(name, layer)

            # add activation
            if i_layer == n_hid_layers:
                # no activation for final layer
                activation_func = None
            else:
                activation_func = get_activation_func_from_str(activation)
            if activation_func:
                name = '%s_%02i' % (activation, global_layer_num)
                mlp.add_module(name, activation_func)

            # update layer info
            global_layer_num += 1
            in_size_ = out_size_

        return mlp

    def forward(self, *args, **kwargs):
        """Push data through model."""
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        """Compute loss."""
        raise NotImplementedError

    def save(self, filepath):
        """Save model parameters."""
        save(self.state_dict(), filepath)

    def get_parameters(self):
        """Get all model parameters that have gradient updates turned on."""
        return filter(lambda p: p.requires_grad, self.parameters())

    def load_parameters_from_file(self, filepath):
        """Load parameters from .pt file."""
        self.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))
        
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
        keys = self.keys
        
        results_dict = {}
        
        results_dict['markers'] = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
                results_dict['markers'][sess] = [np.array([]) for _ in range(dataset.n_sequences)]
                
        results_dict['labels_strong'] = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
                results_dict['labels_strong'][sess] = [np.array([]) for _ in range(dataset.n_sequences)]

        
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
                    data['labels_strong'] = data['labels_strong'][:, pad:-pad]
                    #print('data 2', data['markers'].shape)
                # loop over sequences in batch
                for s, sess in enumerate(sess_list):
                    batch_idx = data['batch_idx'][s].item()
                    
                    results_dict['markers'][sess][batch_idx] = \
                    data['markers'][s].cpu().detach().numpy()
                    
                    results_dict['labels_strong'][sess][batch_idx] = \
                    data['labels_strong'][s].cpu().detach().numpy()
                    
                    #print('lab', data['labels_strong'], data['labels_strong'][s].cpu().detach().numpy().shape)
                    
                    for key in keys:
                        #print('key', key)
                        # push through log-softmax, since this is included in the loss and not model
                        results_dict[key][sess][batch_idx] = \
                        outputs_dict[key][s].cpu().detach().numpy()
                        #softmax(outputs_dict[key][s]).cpu().detach().numpy()
                    
        return results_dict



class Ensembler(object):
    """Ensemble of models."""

    def __init__(self, models):
        self.models = models
        self.n_models = len(models)

    def predict_labels(self, data_generator, combine_before_softmax=False, weights=None):
        """Combine class predictions from multiple models by averaging before softmax.

        Parameters
        ----------
        data_generator : DataGenerator object
            data generator to serve data batches
        combine_before_softmax : bool, optional
            True to combine logits across models before taking softmax; False to take softmax for
            each model then combine probabilities
        weights: array-like, str, or NoneType, optional
            array-like: weight for each model
            str: 'entropy': weight each model at each time point by inverse entropy of distribution
            None: uniform weight for each model

        Returns
        -------
        dict
            - 'labels' (list of lists): corresponding labels

        """

        # initialize container for labels
        labels = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            labels[sess] = [np.array([]) for _ in range(dataset.n_sequences)]

        # process data for each model
        labels_all = []
        for model in self.models:
            outputs_curr = model.predict_labels(
                data_generator, return_scores=combine_before_softmax)
            if combine_before_softmax:
                labels_all.append(outputs_curr['scores'])
            else:
                labels_all.append(outputs_curr['labels'])
        # labels_all is a list of list of lists
        # access: labels_all[idx_model][idx_dataset][idx_batch]

        # ensemble prediction across models
        for sess, labels_sess in enumerate(labels):
            for batch, labels_batch in enumerate(labels_sess):

                # labels_curr is of shape (n_models, sequence_len, n_classes)
                labels_curr = np.vstack(l[sess][batch][None, ...] for l in labels_all)

                # combine predictions across models
                if weights is None:
                    # simple average across models
                    labels_curr = np.mean(labels_curr, axis=0)
                elif isinstance(weights, str) and weights == 'entropy':
                    # weight each model at each time point by inverse entropy of distribution
                    # so that more confident models have a higher weight
                    # compute entropy across labels
                    ent = entropy(labels_curr, axis=-1)
                    # low entropy = high confidence, weight these more
                    w = 1.0 / ent
                    # normalize over models
                    w /= np.sum(w, axis=0)  # shape of (n_models, sequence_len)
                    labels_curr = np.mean(labels_curr * w[..., None], axis=0)
                elif isinstance(weights, (list, tuple, np.ndarray)):
                    # weight each model according to user-supplied weights
                    labels_curr = np.average(labels_curr, axis=0, weights=weights)

                if combine_before_softmax:
                    labels[sess][batch] = scipy_softmax(labels_curr, axis=-1)
                else:
                    labels[sess][batch] = labels_curr

        return {'labels': labels}
    
    
    
    
class BaseInference(BaseModel):
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

        # build classifier: q(y|x)
        self.model['qy_x'] = Module(
            self.hparams, 
            type='decoder',
            in_size=self.hparams['input_size'], 
            hid_size=self.hparams['n_hid_units'], 
            out_size=self.hparams['n_total_classes'])


#         self.model['qy_x'] = self._build_linear(
#             global_layer_num=0, name='qy_x',
#             in_size=self.hparams['input_size'], out_size=self.hparams['n_total_classes'])

        #self.hparams['qy_x_temperature'] = 100#torch.tensor([1]).to(device=self.hparams['device'])

        # build encoder: q(z|x,y)
        self.model['encoder'] = Module(
            self.hparams, 
            in_size=self.hparams['n_total_classes'] + self.hparams['input_size'],
            hid_size=self.hparams['n_hid_units'],
            out_size=self.hparams['n_hid_units'])

#         self.model['qz_xy_mean'] = self._build_linear(
#             global_layer_num=len(self.model['qy_x'].model), name='qz_xy_mean',
#             in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])

#         self.model['qz_xy_logvar'] = self._build_linear(
#             global_layer_num=len(self.model['qy_x'].model), name='qz_xy_logvar',
#                     in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units']) 


        self.model['qz_xy_mean'] = self._build_linear(
                    global_layer_num=0, name='qz_xy_mean',
                    in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])

        self.model['qz_xy_logvar'] = self._build_linear(
            global_layer_num=0, name='qz_xy_logvar',
                    in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units']) 
        
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
        # push inputs through classifier to get q(y|x)
        y_logits = self.model['qy_x'](x)
        #print('qy logits', torch.argmax(y_logits, axis=2))
        # initialize and sample q(y|x) (should be a one-hot vector)
        qy_x_probs = nn.Softmax(dim=2)(y_logits)
        qy_x_logits = y_logits
        #print('q probs', qy_x_probs)
        qy_x = RelaxedOneHotCategorical(temperature=self.hparams['qy_x_temperature'], logits=qy_x_logits)

        y_sample = qy_x.rsample()  # (n_sequences, sequence_length, n_total_classes)
        #print('ysamp inf', y_sample)
        # make ground truth y into onehot
        y_onehot = torch.zeros([y.shape[0], y.shape[1], self.hparams['n_total_classes']], device=y_logits.device)
        
        if self.hparams.get('ignore_class', 0) != 0:
            for s in range(y.shape[0]):
                for i in range(y.shape[1]):
                    if y[s][i] != self.hparams.get('ignore_class', 0):
                        #print('y here', y[s][i].unsqueeze(0))
                        one_hot = MakeOneHot()(y[s][i].unsqueeze(0), self.hparams['n_total_classes'])
                        y_onehot[s][i] = one_hot
                    else:
                        y_onehot[s][i] = torch.zeros(self.hparams['n_total_classes'])

        else:
            for s in range(y.shape[0]):
                #print(y[s].shape)
                y_onehot[s] = MakeOneHot()(y[s], self.hparams['n_total_classes'])           

        # init y_mixed, which will contain true labels for labeled data, samples for unlabled data
        y_mixed = y_onehot.clone().detach()  # (n_sequences, sequence_length, n_total_classes)
        
        
        # loop over sequences in batch
        idxs_labeled = torch.zeros_like(y)
        if self.hparams.get('ignore_class', 0) != 0:
            for s in range(y_mixed.shape[0]):
                for i in range(y_mixed.shape[1]):
                    idxs_labeled[s][i] = y[s][i] != self.hparams.get('ignore_class', 0)
                    if idxs_labeled[s][i] == 0:
                        y_mixed[s, i, :] = y_sample[s, i]

        else:
            for s in range(y_mixed.shape[0]):
                idxs_labeled[s] = y[s] != self.hparams.get('ignore_class', 0)
                y_mixed[s][idxs_labeled[s] == 0] = y_sample[s][idxs_labeled[s] == 0]
                    
        
        # concatenate sample with input x
        # (n_sequences, sequence_length, n_total_classes))
        xy = torch.cat([x, y_mixed], dim=2)
        
        # push [y, x] through encoder to get parameters of q(z|x,y)
        w = self.model['encoder'](xy)
        
        qz_xy_mean = self.model['qz_xy_mean'](w)
        qz_xy_logvar = self.model['qz_xy_logvar'](w)
            
        # sample with reparam trick
        z_xy_sample = qz_xy_mean + torch.randn(qz_xy_mean.shape, device=y_logits.device) * qz_xy_logvar.exp().pow(0.5)
        
        
        return {
            'y_logits': y_logits, # (n_sequences, sequence_length, n_classes)
            'qy_x_probs': qy_x_probs,  # (n_sequences, sequence_length, n_classes)
            'qy_x_logits': qy_x_logits,  # (n_sequences, sequence_length, n_classes)
            'y_sample': y_sample,  # (n_sequences, sequence_length, n_classes)
            'y_mixed': y_mixed,  # (n_sequences, sequence_length, n_classes)
            'qz_xy_mean': qz_xy_mean,  # (n_sequences, sequence_length, embedding_dim)
            'qz_xy_logvar': qz_xy_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'z_xy_sample': z_xy_sample, # (n_sequences, sequence_length, embedding_dim)
            'idxs_labeled': idxs_labeled,  # (n_sequences, sequence_length)
        }
    
    
class BaseGenerative(BaseModel):
    """
    Generative model template
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
        self.hparams['py_probs'] = probs
        
        # build latent_generator: p(z|y)
        # linear layer is essentially a lookup table of shape (n_hid_units, n_total_classes)
        self.model['pz_y_mean'] = self._build_linear(
            0, 'pz_y_mean', self.hparams['n_total_classes'], self.hparams['n_hid_units'])
        self.model['pz_y_logvar'] = self._build_linear(
            0, 'pz_y_logvar', self.hparams['n_total_classes'], self.hparams['n_hid_units'])
        
        # build decoder: p(x|z)
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
    
    def forward(self, x, y, **kwargs): 
        
        # push y through generative model to get parameters of p(z|y)
        pz_y_mean = self.model['pz_y_mean'](kwargs['y_mixed'])

        pz_y_logvar = self.model['pz_y_logvar'](kwargs['y_mixed'])

        # push sampled z from through decoder to get reconstruction
        # this will be the mean of p(x|z)
        x_hat = self.model['decoder'](kwargs['z_xy_sample'])
        
        return {
            'pz_y_mean': pz_y_mean,  # (n_sequences, sequence_length, embedding_dim)
            'pz_y_logvar': pz_y_logvar,  # (n_sequences, sequence_length, embedding_dim)
            'reconstruction': x_hat,  # (n_sequences, sequence_length, n_markers)
        }
