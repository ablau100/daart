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
            self, global_layer_num, in_size, hid_size, out_size, n_hid_layers=1,
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
