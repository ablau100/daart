"""Example script for fitting daart models from the command line with test-tube package."""

import copy
import logging
import numpy as np
from sklearn.decomposition import PCA
import bisect
from scipy import stats

import os
import sys
import time
import torch
from torch.utils.data import SubsetRandomSampler

from daart.eval import plot_training_curves
from daart.io import export_hparams
from daart.testtube import get_all_params, print_hparams, create_tt_experiment, clean_tt_dir
from daart.train import Trainer
from daart.utils import build_data_generator
from daart.data import load_label_csv, compute_sequences, load_marker_csv, load_feature_csv, load_marker_h5


def run_main(hparams, *args):
    
    if not isinstance(hparams, dict):
        hparams = vars(hparams)
        
    # start at random times (so test tube creates separate folders)
    t = time.time()
    np.random.seed(int(100000000000 * t) % (2 ** 32 - 1))
    time.sleep(np.random.uniform(2))

    # create test-tube experiment
    hparams['expt_ids'] = hparams['expt_ids'].split(';')
    hparams, exp = create_tt_experiment(hparams)
    if hparams is None:
        print('Experiment exists! Aborting fit')
        return

    # set up error logging (different from train logging)
    logging.basicConfig(
        filename=os.path.join(hparams['tt_version_dir'], 'console.log'),
        filemode='w', level=logging.DEBUG,
        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))  # add logging to console

    # run train model script
    try:
        train_model(hparams)
    except:
        logging.exception('error traceback')


def train_model(hparams):

    # -------------------------------------
    # build data generator
    # -------------------------------------
    data_gen = build_data_generator(hparams)
    seed = hparams['rng_seed_model']
    np.random.seed(seed)
    
    
    # calculate latent dim z for non TCN models
    if 'huga' in hparams['data_dir']:
        hparams['n_latents'] = 16
    elif 'ssm' in hparams['expt_ids_to_keep']:
        hparams['n_latents'] = 2
    else:
        #load data
        input_type = hparams['input_type']
        for dset in data_gen.datasets:
            path = dset.paths['markers']
            file_ext = path.split('.')[-1]

            if file_ext == 'csv':
                if input_type == 'markers':
                    xs, ys, ls, marker_names = load_marker_csv(path)
                    data_curr = np.hstack([xs, ys])
                else:
                    vals, feature_names = load_feature_csv(path)
                    data_curr = vals

            elif file_ext == 'h5':
                if input_type != 'markers':
                    raise NotImplementedError
                xs, ys, ls, marker_names = load_marker_h5(path)
                data_curr = np.hstack([xs, ys])

            elif file_ext == 'npy':
                # assume single array
                data_curr = np.load(path)

            else:
                raise ValueError('"%s" is an invalid file extension' % file_ext)

            if 'all_data' not in locals():
                all_data = data_curr
                continue
            all_data = np.vstack((all_data, data_curr))

        # do pca
        # Perform PCA
        all_data = stats.zscore(all_data, axis=1) 
        #print(all_data, all_data.shape)
        pca = PCA(n_components=all_data.shape[1])
        pca.fit(all_data)

        # Calculate the percentage of variance explained by each principal component
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        #print(cumulative_variance_ratio)
        idx = bisect.bisect_left(cumulative_variance_ratio, .97)

        # choose latent dim (num pve > .95)
        hparams['n_latents'] = idx+1
    
  
    # calculate alpha for focal loss
    y_dim = hparams['output_size']
    alpha = [1 for _ in range(y_dim)]
#     alpha = [0 for _ in range(y_dim)]
#     n = 0
#     for dset in data_gen.datasets:
#         if not dset.paths['labels_strong']:
#             continue
#         labels, label_names = load_label_csv(dset.paths['labels_strong'])
#         classes = np.argmax(labels, axis=1)
        
#         for c in range(1, y_dim):
#             num_c = np.where(classes==c)[0].shape[0]
#             n += num_c
#             alpha[c] += num_c
            
#     alpha = list(alpha)
#     alpha = [0] + [(n/a) for a in alpha[1:]]
#     hparams['alpha'] = alpha

    # only keep data with labels
    #if hparams.get('labels_only', False):
        
        
    # keep num_labels_per_class
    if hparams['remove_labels']:
        for dset in data_gen.datasets:
            if not dset.paths['labels_strong']:
                continue
            labels, label_names = load_label_csv(dset.paths['labels_strong'])
            print("dset.paths['labels_strong']",dset.paths['labels_strong'])
            data_curr = np.argmax(labels, axis=1)
            # choose labels to keep for each class
            num_classes = hparams['n_observed_classes']
            num_keep_per_class = hparams['num_labels_keep_per_class']
            for label in range(1,num_classes):
                lab_idxes = np.where(data_curr == label)[0]
                print('class ', label, ': ', len(lab_idxes))
                np.random.shuffle(lab_idxes)
                #print('shuffled class ', label, ': ', lab_idxes[:100])
                data_curr[lab_idxes[num_keep_per_class:]] = 0
#             skip = 550; keep = 1; keeps = [] 
#             for i in range(20):
#                 temp = list(range(skip*i + keep*i, skip*i+ keep*(i+1)))
#                 keeps += temp
#             for i in range(data_curr.shape[0]): 
#                 #if i not in keeps:
#                 data_curr[i] = 0  
            print('data_curr')
            unique, counts = np.unique(data_curr, return_counts=True)
            print(np.asarray((unique, counts)).T)
            
            data_curr = compute_sequences(data_curr, dset.sequence_length, dset.sequence_pad)         
            dset.data['labels_strong'] = data_curr
            # create data loaders (will shuffle/batch/etc datasets)
            data_gen.dataset_loaders = [None] * data_gen.n_datasets
            for i, dataset in enumerate(data_gen.datasets):
                data_gen.dataset_loaders[i] = {}
                for dtype in data_gen._dtypes:
                    data_gen.dataset_loaders[i][dtype] = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=1,  # keep 1 here so we can combine batches from multiple datasets
                        sampler=SubsetRandomSampler(dataset.batch_idxs[dtype]),
                        num_workers=data_gen.num_workers,
                        pin_memory=data_gen.pin_memory)
            # create all iterators (will iterate through data loaders)
            data_gen.dataset_iters = [None] * data_gen.n_datasets
            for i in range(data_gen.n_datasets):
                data_gen.dataset_iters[i] = {}
                for dtype in data_gen._dtypes:
                    data_gen.dataset_iters[i][dtype] = iter(data_gen.dataset_loaders[i][dtype])

    
    # print hparams to console
    print_str = print_hparams(hparams)
    logging.info(print_str)
    
    logging.info(data_gen)
    
    data_gen_test = build_data_generator(hparams, test=True)
    logging.info(data_gen_test)

    # -------------------------------------
    # build model
    # -------------------------------------
    torch.manual_seed(hparams.get('rng_seed_model', 0))
    print(',od class', hparams['model_class'].lower())

    
    
    if hparams['model_class'].lower() == 'segmenter':
        from daart.models import Segmenter
        model = Segmenter(hparams)
    elif hparams['model_class'].lower() == 'gmdgm':
        from daart.models import GMDGM
        model = GMDGM(hparams)
    elif hparams['model_class'].lower() == 'rslds_marginal':
        from daart.models import RSLDSM
        print('using rslds marginal')
        model = RSLDSM(hparams)
    elif hparams['model_class'].lower() == 'rslds_sample':
        from daart.models import RSLDSS
        print('using rslds sample')
        model = RSLDSS(hparams)
    else:
        raise NotImplementedError
    model.to(hparams['device'])
    logging.info(model)

    # -------------------------------------
    # set up training callbacks
    # -------------------------------------
    callbacks = []
    if hparams['enable_early_stop']:
        from daart.callbacks import EarlyStopping
        # Note that patience does not account for val check interval values greater than 1;
        # for example, if val_check_interval=5 and patience=20, then the model will train
        # for at least 5 * 20 = 100 epochs before training can terminate
        callbacks.append(EarlyStopping(patience=hparams['early_stop_history']))
    if hparams.get('semi_supervised_algo', 'none') == 'pseudo_labels':
        from daart.callbacks import AnnealHparam, PseudoLabels
        if model.hparams['lambda_weak'] == 0:
            print('warning! use lambda_weak in model.yaml to weight pseudo label loss')
        else:
            callbacks.append(AnnealHparam(
                hparams=model.hparams, key='lambda_weak', epoch_start=hparams['anneal_start'],
                epoch_end=hparams['anneal_end']))
            callbacks.append(PseudoLabels(
                prob_threshold=hparams['prob_threshold'], epoch_start=hparams['anneal_start']))
            
    if hparams.get('variational', False):
        from daart.callbacks import AnnealHparam
        callbacks.append(AnnealHparam(
            hparams=model.hparams, key='kl_weight', epoch_start=0, epoch_end=100, val_end=model.hparams['kl_weight']))
        
    # callback for uniform KL loss and entropy loss
    
    if hparams['kl_y_weight_uniform'] > 0:
    
        from daart.callbacks import AnnealHparam
        callbacks.append(AnnealHparam(
            hparams=model.hparams, key='kl_y_weight_uniform', epoch_start=model.hparams['kl_y_weight_uniform_start_anneal'], epoch_end=model.hparams['kl_y_weight_uniform_end'], val_start=model.hparams['kl_y_weight_uniform'], val_end=model.hparams['kl_y_weight_uniform_end_val']))

#         callbacks.append(AnnealHparam(
#             hparams=model.hparams, key='entropy_weight', epoch_start=model.hparams['entropy_start_anneal'], epoch_end=3000, val_start=model.hparams['entropy_weight'], val_end=0))
    
    # call back for kl y, kl y, and log py
    from daart.callbacks import AnnealHparam
    callbacks.append(AnnealHparam(
        hparams=model.hparams, key='ann_weight', epoch_start=model.hparams['ann_start'],
        epoch_end=model.hparams['ann_end'], val_start=0, val_end=model.hparams['ann_weight']))

    # -------------------------------------
    # train model + cleanup
    # -------------------------------------
    trainer = Trainer(**hparams, callbacks=callbacks)
    trainer.fit(model, data_gen, save_path=hparams['tt_version_dir'], data_gen_test=data_gen_test)

    # update hparams upon successful training
    hparams['training_completed'] = True
    export_hparams(hparams)

    # save training curves
    if hparams.get('plot_train_curves', False):
        plot_training_curves(
            os.path.join(hparams['tt_version_dir'], 'metrics.csv'), dtype='train',
            expt_ids=hparams['expt_ids'],
            save_file=os.path.join(hparams['tt_version_dir'], 'train_curves'),
            format='png')
#         plot_training_curves(
#             os.path.join(hparams['tt_version_dir'], 'metrics.csv'), dtype='val',
#             expt_ids=hparams['expt_ids'],
#             save_file=os.path.join(hparams['tt_version_dir'], 'val_curves'),
#             format='png')
        
    # save diagnostic summary

    # get rid of unneeded logging info
    clean_tt_dir(hparams)


if __name__ == '__main__':

    """To run:

    (daart) $: python fit_models.py --data_config /path/to/data.yaml 
       --model_config /path/to/model.yaml --train_config /path/to/train.yaml

    For example yaml files, see the `configs` subdirectory inside the daart home directory

    NOTE: this script assumes a specific naming convention for markers and labels (see L54-L65). 
    You'll need to update these lines to be consistent with your own naming conventions.
    
    """

    hyperparams = get_all_params()
    
    if hyperparams.device == 'cuda':
        if isinstance(hyperparams.gpus_vis, int):
            gpu_ids = [str(hyperparams.gpus_vis)]
        else:
            gpu_ids = hyperparams.gpus_vis.split(';')
        hyperparams.optimize_parallel_gpu(
            run_main,
            gpu_ids=gpu_ids)

    elif hyperparams.device == 'cpu':
        hyperparams.optimize_parallel_cpu(
            run_main,
            nb_trials=hyperparams.tt_n_cpu_trials,
            nb_workers=hyperparams.tt_n_cpu_workers)
