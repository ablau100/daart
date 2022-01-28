import copy
import pytest
import torch

from daart.models import Segmenter
from daart.models.rnn import RNN


def test_lstm(hparams, data_generator):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'lstm'

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.LSTM)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    dtypes = output_dict.keys()
    assert 'labels' in dtypes
    assert 'labels_weak' in dtypes
    assert 'prediction' in dtypes
    assert 'task_prediction' in dtypes
    assert 'embedding' in dtypes

    batch_size = output_dict['embedding'].shape[0]
    assert len(output_dict['embedding'].shape) == 3  # (n_seqs, seq_len, embedding_dim)
    if output_dict['labels'] is not None:
        assert output_dict['labels'].shape[0] == batch_size
        assert len(output_dict['labels'].shape) == 3  # (n_seqs, seq_len, n_classes)
    if output_dict['labels_weak'] is not None:
        assert output_dict['labels_weak'].shape[0] == batch_size
        assert len(output_dict['labels_weak'].shape) == 3  # (n_seqs, seq_len, n_classes)
    if output_dict['prediction'] is not None:
        assert output_dict['prediction'].shape[0] == batch_size
        assert len(output_dict['prediction'].shape) == 3  # (n_seqs, seq_len, n_markers)
    if output_dict['task_prediction'] is not None:
        assert output_dict['task_prediction'].shape[0] == batch_size  # (n_seqs, seq_len, n_tasks)
        assert len(output_dict['task_prediction'].shape) == 3

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict


def test_gru(hparams, data_generator):

    hp = copy.deepcopy(hparams)
    hp['backbone'] = 'gru'

    model = Segmenter(hp)
    model.to(hp['device'])

    # load the correct backbone
    assert isinstance(model.model['encoder'], RNN)
    for name, layer in model.model['encoder'].model.named_children():
        assert isinstance(layer, torch.nn.GRU)
        break

    # print doesn't fail
    assert model.__str__()

    # process a batch of data
    batch = data_generator.datasets[0][0]
    output_dict = model(batch['markers'].unsqueeze(0))  # add batch dim
    dtypes = output_dict.keys()
    assert 'labels' in dtypes
    assert 'labels_weak' in dtypes
    assert 'prediction' in dtypes
    assert 'task_prediction' in dtypes
    assert 'embedding' in dtypes

    batch_size = output_dict['embedding'].shape[0]
    assert len(output_dict['embedding'].shape) == 3  # (n_seqs, seq_len, embedding_dim)
    if output_dict['labels'] is not None:
        assert output_dict['labels'].shape[0] == batch_size
        assert len(output_dict['labels'].shape) == 3  # (n_seqs, seq_len, n_classes)
    if output_dict['labels_weak'] is not None:
        assert output_dict['labels_weak'].shape[0] == batch_size
        assert len(output_dict['labels_weak'].shape) == 3  # (n_seqs, seq_len, n_classes)
    if output_dict['prediction'] is not None:
        assert output_dict['prediction'].shape[0] == batch_size
        assert len(output_dict['prediction'].shape) == 3  # (n_seqs, seq_len, n_markers)
    if output_dict['task_prediction'] is not None:
        assert output_dict['task_prediction'].shape[0] == batch_size  # (n_seqs, seq_len, n_tasks)
        assert len(output_dict['task_prediction'].shape) == 3

    # compute losses
    batch_ext = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch_ext[key] = val.unsqueeze(0)  # add batch dim
        else:
            batch_ext[key] = val

    loss_dict = model.training_step(batch_ext, accumulate_grad=False)
    assert 'loss' in loss_dict
