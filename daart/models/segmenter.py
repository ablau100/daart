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
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from daart import losses
from daart.losses import FocalLoss
from daart.models.base import BaseModel, reparameterize_gaussian, get_activation_func_from_str

# to ignore imports for sphix-autoapidoc
__all__ = [
    'Segmenter'
]


class Segmenter(BaseModel):
    """General wrapper class for behavioral segmentation models."""

    def __init__(self, hparams):
        """

        Parameters
        ----------
        hparams : dict
            - backbone (str): 'temporal-mlp' | 'dtcn' | 'lstm' | 'gru'
            - rng_seed_model (int): random seed to control weight initialization
            - input_size (int): number of input channels
            - output_size (int): number of classes
            - task_size (int): number of regression tasks
            - batch_pad (int): padding needed to account for convolutions
            - n_hid_layers (int): hidden layers of network architecture
            - n_hid_units (int): hidden units per layer
            - n_lags (int): number of lags in input data to use for temporal convolution
            - activation (str): 'linear' | 'relu' | 'lrelu' | 'sigmoid' | 'tanh'
            - classifier_type (str): 'multiclass' | 'binary' | 'multibinary'
            - class_weights (array-like): weights on classes
            - variational (bool): whether or not model is variational
            - lambda_weak (float): hyperparam on weak label classification
            - lambda_strong (float): hyperparam on srong label classification
            - lambda_pred (float): hyperparam on next step prediction
            - lambda_task (float): hyperparam on task regression

        """
        super().__init__()
        self.hparams = hparams

        # model dict will contain some or all of the following components:
        # - encoder: inputs -> latents
        # - classifier: latents -> hand labels
        # - classifier_weak: latents -> heuristic/pseudo labels
        # - task_predictor: latents -> tasks
        # - decoder: latents[t] -> inputs[t]
        # - predictor: latents[t] -> inputs[t+1]
        self.model = nn.ModuleDict()
        self.build_model()

        # label loss based on cross entropy; don't compute gradient when target = 0
        classifier_type = hparams.get('classifier_type', 'multiclass')
        if classifier_type == 'multiclass':
            # multiple mutually exclusive classes, 0 is backgroud class
            ignore_index = hparams.get('ignore_class', 0)
        elif classifier_type == 'binary':
            # single class
            ignore_index = -100  # pytorch default
        elif classmethod == 'multibinary':
            # multiple non-mutually exclusive classes (each a binary classification)
            raise NotImplementedError
        else:
            raise NotImplementedError("classifier type must be 'multiclass' or 'binary'")
        #weight = hparams.get('class_weights', None)
        weight = hparams.get('alpha', None)
        if weight is not None:
            weight = torch.tensor(weight)
        
        focal_loss = self.hparams.get('focal_loss', False)
        if focal_loss:
            #self.class_loss = FocalLoss(gamma=2, alpha=self.hparams['alpha'], ignore_index=ignore_index)
            self.class_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean', label_smoothing=.3)
        else:
            self.class_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')
                                                  #, label_smoothing=0.1)
        self.pred_loss = nn.MSELoss(reduction='mean')
        self.task_loss = nn.MSELoss(reduction='mean')

    def __str__(self):
        """Pretty print model architecture."""

        format_str = '\n%s architecture\n' % self.hparams['backbone'].upper()
        format_str += '------------------------\n'

        format_str += 'Encoder:\n'
        for i, module in enumerate(self.model['encoder'].model):
            format_str += str('    {}: {}\n'.format(i, module))
        format_str += '\n'

        if self.hparams.get('variational', False):
            format_str += 'Variational Layers:\n'
            for l in ['latent_mean', 'latent_logvar']:
                for i, module in enumerate(self.model[l]):
                    format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'decoder' in self.model:
            format_str += 'Decoder:\n'
            for i, module in enumerate(self.model['decoder'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'predictor' in self.model:
            format_str += 'Predictor:\n'
            for i, module in enumerate(self.model['predictor'].model):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'classifier' in self.model:
            format_str += 'Classifier:\n'
            for i, module in enumerate(self.model['classifier']):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'classifier_weak' in self.model:
            format_str += 'Classifier Weak:\n'
            for i, module in enumerate(self.model['classifier_weak']):
                format_str += str('    {}: {}\n'.format(i, module))
            format_str += '\n'

        if 'task_predictor' in self.model:
            format_str += 'Task Predictor:\n'
            for i, module in enumerate(self.model['task_predictor']):
                format_str += str('    {}: {}\n'.format(i, module))

        return ''#format_str

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
        elif self.hparams['backbone'].lower() == 'ms_dtcn':
            from daart.backbones.ms_tcn import DilatedTCN as Module
        elif self.hparams['backbone'].lower() == 'msm':
            from daart.backbones.ms_tcn import MultiStageModel as Module
        elif self.hparams['backbone'].lower() in ['lstm', 'gru']:
            from daart.backbones.rnn import RNN as Module
        elif self.hparams['backbone'].lower() == 'tgm':
            raise NotImplementedError
            # from daart.models.tgm import TGM as Module
        else:
            raise ValueError('"%s" is not a valid backbone network' % self.hparams['backbone'])

        global_layer_num = 0

        # build encoder module
        self.model['encoder'] = Module(self.hparams, type='encoder')
        if self.hparams.get('variational', False):
            self.hparams['kl_weight'] = 1  # weight in front of kl term; anneal this using callback
            self.model['latent_mean'] = self._build_linear(
                global_layer_num=len(self.model['encoder'].model), name='latent_mean',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])
            self.model['latent_logvar'] = self._build_linear(
                global_layer_num=len(self.model['encoder'].model), name='latent_logvar',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['n_hid_units'])

        # classifier: single linear layer for hand labels
        if self.hparams.get('lambda_strong', 0) > 0:
            self.model['classifier'] = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['output_size'])
            
            format_str = 'seed: ' + str(self.hparams['rng_seed_model']) + '\n'

            for i, module in enumerate(self.model['classifier']):
                format_str += str('    {}: {}\n'.format(i, module))
                format_str += str(' Weights py: {}\n'.format(module.weight))
                format_str += str(' bias py: {}\n'.format(module.bias))
            format_str += '\n'
            

        # build predictor module
        if self.hparams.get('lambda_pred', 0) > 0:
            self.model['predictor'] = Module(self.hparams, type='decoder')
        
            
       # build decoder module
        if self.hparams.get('lambda_recon', 0) > 0:
            self.model['decoder'] = Module(self.hparams, type='decoder')

        # classifier: single linear layer for heuristic labels
        if self.hparams.get('lambda_weak', 0) > 0:
            self.model['classifier_weak'] = self._build_linear(
                global_layer_num=global_layer_num, name='classification',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['output_size'])

        # task regression: single linear layer
        if self.hparams.get('lambda_task', 0) > 0:
            self.model['task_predictor'] = self._build_linear(
                global_layer_num=global_layer_num, name='regression',
                in_size=self.hparams['n_hid_units'], out_size=self.hparams['task_size'])

    def forward(self, x):
        """Process input data.

        Parameters
        ----------
        x : torch.Tensor
            input data of shape (n_sequences, sequence_length, n_markers)

        Returns
        -------
        dict of model outputs/internals as torch tensors
            - 'labels' (torch.Tensor): model classification
               shape of (n_sequences, sequence_length, n_classes)
            - 'labels_weak' (torch.Tensor): model classification of weak/pseudo labels
              shape of (n_sequences, sequence_length, n_classes)
            - 'reconstruction' (torch.Tensor): input decoder prediction
              shape of (n_sequences, sequence_length, n_markers)
            - 'prediction' (torch.Tensor): one-step-ahead prediction
              shape of (n_sequences, sequence_length, n_markers)
            - 'task_prediction' (torch.Tensor): prediction of regression tasks
              (n_sequences, sequence_length, n_tasks)
            - 'embedding' (torch.Tensor): behavioral embedding used for classification/prediction
              in non-variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'mean' (torch.Tensor): mean of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'logvar' (torch.Tensor): logvar of appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)
            - 'sample' (torch.Tensor): sample from appx posterior of latents in variational models
              shape of (n_sequences, sequence_length, embedding_dim)

        """
        # push data through encoder to get latent embedding
        # x = B x T x N (e.g. B = 2, T = 500, N = 16)
        x = self.model['encoder'](x)
        if self.hparams.get('variational', False):
            mean = self.model['latent_mean'](x)
            logvar = self.model['latent_logvar'](x)
            z = reparameterize_gaussian(mean, logvar)
        else:
            mean = x
            logvar = None
            z = x

        # push embedding through classifiers to get hand labels
        if self.hparams.get('lambda_strong', 0) > 0:
            y = self.model['classifier'](z)
        else:
            y = None

        # push embedding through linear layer to heuristic/pseudo labels
        if self.hparams.get('lambda_weak', 0) > 0:
            y_weak = self.model['classifier_weak'](z)
        else:
            y_weak = None

        # push embedding through linear layer to get task predictions
        if self.hparams.get('lambda_task', 0) > 0:
            w = self.model['task_predictor'](z)
        else:
            w = None

        # push embedding through decoder network to get data at current time point
        if self.hparams.get('lambda_recon', 0) > 0:
            xt = self.model['decoder'](z)
        else:
            xt = None

        # push embedding through predictor network to get data at subsequent time points
        if self.hparams.get('lambda_pred', 0) > 0:
            xtp1 = self.model['predictor'](z)
        else:
            xtp1 = None

        return {
            'labels': y,  # (n_sequences, sequence_length, n_classes)
            'labels_weak': y_weak,  # (n_sequences, sequence_length, n_classes)
            'reconstruction': xt,  # (n_sequences, sequence_length, n_markers)
            'prediction': xtp1,  # (n_sequences, sequence_length, n_markers)
            'task_prediction': w,  # (n_sequences, sequence_length, n_tasks)
            'embedding': mean,  # (n_sequences, sequence_length, embedding_dim)
            'latent_mean': mean,  # (n_sequences, sequence_length, embedding_dim)
            'latent_logvar': logvar,  # (n_sequences, sequence_length, embedding_dim)
            'sample': z,  # (n_sequences, sequence_length, embedding_dim)
        }

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
        mode : str
            'eval' | 'train'

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

        # initialize containers
        
        # markers
        markers = [[] for _ in range(data_generator.n_datasets)]
        # softmax outputs
        labels = [[] for _ in range(data_generator.n_datasets)]
        # logits
        scores = [[] for _ in range(data_generator.n_datasets)]
        # latent representation
        embedding = [[] for _ in range(data_generator.n_datasets)]
        # predictions on regression task
        task_predictions = [[] for _ in range(data_generator.n_datasets)]
        for sess, dataset in enumerate(data_generator.datasets):
            labels[sess] = [np.array([]) for _ in range(dataset.n_sequences)]
            scores[sess] = [np.array([]) for _ in range(dataset.n_sequences)]
            embedding[sess] = [np.array([]) for _ in range(dataset.n_sequences)]
            markers[sess] = [np.array([]) for _ in range(dataset.n_sequences)]
            task_predictions[sess] = [np.array([]) for _ in range(dataset.n_sequences)]

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
                    # push through log-softmax, since this is included in the loss and not model
                    labels[sess][batch_idx] = \
                        softmax(outputs_dict['labels'][s]).cpu().detach().numpy()
                    embedding[sess][batch_idx] = \
                        outputs_dict['embedding'][s].cpu().detach().numpy()
                    markers[sess][batch_idx] = \
                        data['markers'][:, pad:-pad][s].cpu().detach().numpy()
                    if return_scores:
                        scores[sess][batch_idx] = \
                            outputs_dict['labels'][s].cpu().detach().numpy()
                    if outputs_dict.get('task_prediction', None) is not None:
                        task_predictions[sess][batch_idx] = \
                            outputs_dict['task_prediction'][s].cpu().detach().numpy()

        return {
            'labels': labels,
            'scores': scores,
            'embedding': embedding,
            'task_predictions': task_predictions,
            'markers': markers
        }

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
        lambda_weak = self.hparams.get('lambda_weak', 0)
        lambda_strong = self.hparams.get('lambda_strong', 0)
        lambda_pred = self.hparams.get('lambda_pred', 0)
        lambda_task = self.hparams.get('lambda_task', 0)
        kl_weight = self.hparams.get('kl_weight', 1)

        # index padding for convolutions
        pad = self.hparams.get('sequence_pad', 0)

        # push data through model
        markers_wpad = data['markers']
        outputs_dict = self.forward(markers_wpad)
        
        #np.random.seed(0)

        # remove padding from supplied data
        if lambda_strong > 0:
            if pad > 0:
                labels_strong = data['labels_strong'][:, pad:-pad, ...]
            else:
                labels_strong = data['labels_strong']
            # reshape to fit into class loss; needs to be (n_examples,)
            labels_strong = torch.flatten(labels_strong)
        else:
            labels_strong = None

        if lambda_weak > 0:
            if pad > 0:
                labels_weak = data['labels_weak'][:, pad:-pad, ...]
            else:
                labels_weak = data['labels_weak']
            # reshape to fit into class loss; needs to be (n_examples,)
            labels_weak = torch.flatten(labels_weak)
        else:
            labels_weak = None

        if lambda_task > 0:
            if pad > 0:
                tasks = data['tasks'][:, pad:-pad, ...]
            else:
                tasks = data['tasks']
        else:
            tasks = None

        # remove padding from model output
        if pad > 0:
            markers = markers_wpad[:, pad:-pad, ...]
            # remove padding from model output
            for key, val in outputs_dict.items():
                outputs_dict[key] = val[:, pad:-pad, ...] if val is not None else None
        else:
            markers = markers_wpad

        # initialize loss to zero
        loss = 0
        loss_dict = {}

        # ------------------------------------
        # compute loss on weak labels
        # ------------------------------------
        if lambda_weak > 0:
            # reshape predictions to fit into class loss; needs to be (n_examples, n_classes)
            labels_weak_reshape = torch.reshape(
                outputs_dict['labels_weak'], (-1, outputs_dict['labels_weak'].shape[-1]))
            # only compute loss where strong labels do not exist [indicated by a zero]
            if labels_strong is not None:
                loss_weak = self.class_loss(
                    labels_weak_reshape[labels_strong == 0], labels_weak[labels_strong == 0])
            else:
                loss_weak = self.class_loss(labels_weak_reshape, labels_weak)
            loss += lambda_weak * loss_weak
            loss_weak_val = loss_weak.item()
            loss_dict['loss_weak'] = loss_weak_val
            # compute fraction correct on weak labels
            if 'labels' in outputs_dict.keys():
                fc = accuracy_score(
                    labels_weak.cpu().detach().numpy().flatten(),
                    np.argmax(outputs_dict['labels'].cpu().detach().numpy(), axis=2).flatten(),
                )
                # log
                loss_dict['fc'] = fc

        # ------------------------------------
        # compute loss on strong labels
        # ------------------------------------
        if lambda_strong > 0:
            # reshape predictions to fit into class loss; needs to be (n_examples, n_classes)
            labels_strong_reshape = torch.reshape(
                outputs_dict['labels'], (-1, outputs_dict['labels'].shape[-1]))
            loss_strong = self.class_loss(labels_strong_reshape, labels_strong)
            loss += lambda_strong * loss_strong
            loss_strong_val = loss_strong.item()
            # log
            loss_dict['loss_strong'] = loss_strong_val

        # ------------------------------------
        # compute loss on one-step predictions
        # ------------------------------------
        if lambda_pred > 0:
            loss_pred = self.pred_loss(markers[:, 1:], outputs_dict['prediction'][:, :-1])
            loss += lambda_pred * loss_pred
            loss_pred_val = loss_pred.item()
            # log
            loss_dict['loss_pred'] = loss_pred_val

        # ------------------------------------
        # compute regression loss on tasks
        # ------------------------------------
        if lambda_task > 0:
            loss_task = self.task_loss(tasks, outputs_dict['task_prediction'])
            loss += lambda_task * loss_task
            loss_task_val = loss_task.item()
            r2 = r2_score(
                tasks.cpu().detach().numpy().flatten(),
                outputs_dict['task_prediction'].cpu().detach().numpy().flatten(),
            )
            # log
            loss_dict['loss_task'] = loss_task_val
            loss_dict['task_r2'] = r2

        # ------------------------------------
        # compute kl divergence on appx posterior
        # ------------------------------------
        if self.hparams.get('variational', False):
            # multiply by 2 to take into account the fact that we're computing raw mse for decoding
            # and prediction rather than (1 / 2\sigma^2) * MSE
            loss_kl = 2.0 * losses.kl_div_to_std_normal(
                outputs_dict['latent_mean'], outputs_dict['latent_logvar'])
            loss += kl_weight * loss_kl
            # log
            loss_dict['kl_weight'] = kl_weight
            loss_dict['loss_kl'] = loss_kl.item()
            
            
        # ----------------------------------------------------------------------------------
        # compute kl loss between q(y_t|x_(T_t) and p(y_1) for all (uniform kl)
        # ----------------------------------------------------------------------------------   
        if self.hparams['kl_y_weight_uniform'] > 0:
            y_dim = self.hparams['output_size']
            py_logits =  torch.cat((torch.tensor([.001]), torch.ones((y_dim-1))/(y_dim-1))).to(device=labels_strong_reshape.device)

            #print('py', py_logits.shape,py_logits)
            py = Categorical(py_logits) 

            qy_logits = labels_strong_reshape.mean(dim=0) # (N, n_classes)

            qy = Categorical(logits=qy_logits)

            loss_y_kl_uniform = torch.mean(kl_divergence(qy, py), axis=0) 
            loss_y_kl_uniform = loss_y_kl_uniform * self.hparams['kl_y_weight_uniform']
            #print("self.hparams['kl_y_weight_uniform']", self.hparams['kl_y_weight_uniform'])

            loss += loss_y_kl_uniform
            loss_dict['loss_y_kl_uniform'] = loss_y_kl_uniform.item()

        if accumulate_grad:
            loss.backward()

        # collect loss vals
        loss_dict['loss'] = loss.item()

        return loss_dict


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
