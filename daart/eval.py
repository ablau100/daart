"""Evaluation functions for the daart package."""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import recall_score, precision_score
from typeguard import typechecked
from typing import List, Optional, Union
import itertools

from daart.io import make_dir_if_not_exists

# to ignore imports for sphix-autoapidoc
__all__ = ['get_precision_recall', 'int_over_union', 'run_lengths', 'plot_training_curves', 'get_all_diagnostics']

def confusion_matrix(true_states, inf_states, num_states):
    confusion = np.zeros((num_states, num_states))
    ztotal = np.zeros((num_states, 1))
    for i in range(num_states):
        for ztrue, zinf in zip(true_states, inf_states):
            for j in range(num_states):
                confusion[i, j] += np.sum((ztrue == i) & (zinf == j))
            ztotal[i] += np.sum(ztrue==i)
    return confusion / ztotal

def get_all_diagnostics(model, hparams, data_gen, save_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    start = 0
    stop=500
    # get model output
    tmp = model.predict_labels(data_gen, return_scores=True)
    y_hat = np.vstack(tmp['qy_x_probs'][0])
#     z_hat = np.vstack(tmp['qz_xy_mean'][0])
#     x_hat = np.vstack(tmp['reconstruction'][0])
    
#     x_gt = np.vstack(tmp['markers'][0])
    y_gt = np.vstack(tmp['labels_strong'][0]).reshape(-1)
    
#     print('xhat', x_hat.shape)
#     print('xgt', x_gt.shape)
    
    print('y_hat', y_hat.shape)
    print('ygt', y_gt.shape)
     
#     # save reonstruction metrics
#     mse = x_gt - x_hat # recon MSE
#     mse = np.mean(mse*mse, axis=1)
#     mse = np.mean(mse, axis=0)
    
#     nrow = int((x_gt.shape[1]+2)/2)
    
#     fig, axs = plt.subplots(nrows=nrow, ncols=2, figsize=(7, 10))

#     i = 0
#     for row in axs:
#         for plot in row:
#             if i >= x_gt.shape[1]:
#                 break
#             plot.set_title("Marker "+ str(i))
#             plot.plot(np.arange(start,stop), x_gt[start:stop, i],color='blue',label='x')
#             plot.plot(np.arange(start,stop), x_hat[start:stop, i],color='orange',label='reconstruction')
#             if i >= (x_gt.shape[1]-2):
#                 plot.set_xlabel("epoch")

#             i+=1

#     axs[0, 0].legend(loc='best')
#     bar1 = axs[nrow-1][1].bar('MSE', [mse], width=1.04, color='red')
#     axs[nrow-1][1].set_title("Recon MSE")

#     axs[nrow-1][1].bar_label(bar1, fmt='%.3f', size = 15)
#     axs[nrow-1][1].set_ylim(0,mse+0.5)
#     fig.tight_layout()
#     plt.savefig(os.path.join(save_path, 'recon_metrics.pdf'))
    
    # save classification metrics
    # get F1
    label_names = hparams['label_names']
    #print('LNNN', label_names)
    f1 = np.zeros(len(label_names))
     
    f1_groups = list(itertools.permutations(list(range(len(label_names)))))
    
    y_hat = np.argmax(y_hat, axis=1)
    print('y_hat', y_hat.shape, y_hat[:120])
    y_hat_best = np.zeros_like(y_hat)
    #print('yhat', y_hat, y_hat.shape)
    background = 0 if hparams['ignore_class']==0 else None
    for perm in f1_groups:
        y_hat_temp = np.ones_like(y_hat) * -1
        for i in range(len(label_names)):
            y_hat_temp[y_hat==i] = perm[i]
        
        #print('')
        #print('ygt', y_gt, y_gt.shape)
        #print('yp', y_hat_temp, y_hat_temp.shape)
       
        f1_temp = np.array(get_precision_recall(y_gt, y_hat_temp, background, len(label_names))['f1']).astype(float)
        #print('f1temp', f1_temp)
        #print('f1', f1)
        if np.mean(f1_temp) > np.mean(f1):
            f1 = f1_temp
            y_hat_best = y_hat_temp
    #print('F1', f1)
    
    f1s = list(f1)
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    state_overlaps_all_0 = confusion_matrix(
        y_gt, y_hat_best, num_states=len(label_names))

    im = axs[0].imshow(state_overlaps_all_0[:, :], vmin=0, vmax=1, cmap='Reds')

    axs[0].set_title('Confusion Matrix')

    plt.sca(axs[0])
    plt.xticks(range(len(label_names)), label_names, rotation=25)
    plt.yticks(np.arange(len(label_names)), label_names)
    plt.colorbar(im, ax=axs[0])

    y_pos = np.arange(len(label_names)+1)
    
    hbars = axs[1].barh(y_pos, f1s+[sum(f1s)/len(f1s)], align='center')
    axs[1].set_yticks(y_pos, labels=label_names + ['mean'])
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_xlabel('F1')
    axs[1].set_title('F1 scores')

    # Label with specially formatted floats
    axs[1].bar_label(hbars, fmt='%.3f')
    axs[1].set_xlim(right=1.5)  # adjust xlim to fit labels

    plt.savefig(os.path.join(save_path, 'class_metrics.pdf'))
    
    
    
    
@typechecked
def get_precision_recall(
        true_classes: np.ndarray,
        pred_classes: np.ndarray,
        background: Union[int, None] = 0,
        n_classes: Optional[int] = None
) -> dict:
    """Compute precision and recall for classifier.

    Parameters
    ----------
    true_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    pred_classes : array-like
        entries should be in [0, K-1] where K is the number of classes
    background : int or NoneType
        defines the background class that identifies points with no supervised label; these time
        points are omitted from the precision and recall calculations; if NoneType, no background
        class is utilized
    n_classes : int, optional
        total number of non-background classes; if NoneType, will be inferred from true classes

    Returns
    -------
    dict:
        'precision' (array-like): precision for each class (including background class)
        'recall' (array-like): recall for each class (including background class)

    """

    assert true_classes.shape[0] == pred_classes.shape[0]

    # find all data points that are not background
    if background is not None:
        assert background == 0  # need to generalize
        obs_idxs = np.where(true_classes != background)[0]
    else:
        obs_idxs = np.arange(true_classes.shape[0])

    if n_classes is None:
        n_classes = len(np.unique(true_classes[obs_idxs]))

    # set of labels to include in metric computations
    if background is not None:
        labels = np.arange(1, n_classes + 1)
    else:
        labels = np.arange(n_classes)

    precision = precision_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)
    recall = recall_score(
        true_classes[obs_idxs], pred_classes[obs_idxs],
        labels=labels, average=None, zero_division=0)

    # replace 0s with NaNs for classes with no ground truth
    # for n in range(precision.shape[0]):
    #     if precision[n] == 0 and recall[n] == 0:
    #         precision[n] = np.nan
    #         recall[n] = np.nan

    # compute f1
    p = precision
    r = recall
    f1 = 2 * p * r / (p + r + 1e-10)
    return {'precision': p, 'recall': r, 'f1': f1}


@typechecked
def int_over_union(array1: np.ndarray, array2: np.ndarray) -> dict:
    """Compute intersection over union for two 1D arrays.

    Parameters
    ----------
    array1 : array-like
        integer array of shape (n,)
    array2 : array-like
        integer array of shape (n,)

    Returns
    -------
    dict
        keys are integer values in arrays, values are corresponding IoU (float)

    """
    vals = np.unique(np.concatenate([np.unique(array1), np.unique(array2)]))
    iou = {val: np.nan for val in vals}
    for val in vals:
        intersection = np.sum((array1 == val) & (array2 == val))
        union = np.sum((array1 == val) | (array2 == val))
        iou[val] = intersection / union
    return iou


@typechecked
def run_lengths(array: np.ndarray) -> dict:
    """Compute distribution of run lengths for an array with integer entries.

    Parameters
    ----------
    array : array-like
        single-dimensional array

    Returns
    -------
    dict
        keys are integer values up to max value in array, values are lists of run lengths


    Example
    -------
    >>> a = [1, 1, 1, 0, 0, 4, 4, 4, 4, 4, 4, 0, 1, 1, 1, 1]
    >>> run_lengths(a)
    {0: [2, 1], 1: [3, 4], 2: [], 3: [], 4: [6]}

    """
    seqs = {k: [] for k in np.arange(np.max(array) + 1)}
    for key, iterable in itertools.groupby(array):
        seqs[key].append(len(list(iterable)))
    return seqs


@typechecked
def plot_training_curves(
        metrics_file: str,
        dtype: str = 'val',
        expt_ids: Optional[list] = None,
        save_file: Optional[str] = None,
        format: str = 'pdf'
) -> None:
    """Create training plots for each term in the objective function.

    The `dtype` argument controls which type of trials are plotted ('train' or 'val').
    Additionally, multiple models can be plotted simultaneously by varying one (and only one) of
    the following parameters:

    TODO

    Each of these entries must be an array of length 1 except for one option, which can be an array
    of arbitrary length (corresponding to already trained models). This function generates a single
    plot with panels for each of the following terms:

    - total loss
    - weak label loss
    - strong label loss
    - prediction loss

    Parameters
    ----------
    metrics_file : str
        csv file saved during training
    dtype : str
        'train' | 'val'
    expt_ids : list, optional
        dataset names for easier parsing
    save_file : str, optional
        absolute path of save file; does not need file extension
    format : str, optional
        format of saved image; 'pdf' | 'png' | 'jpeg' | ...

    """

    metrics_list = [
        'loss', 'loss_weak', 'loss_strong', 'loss_pred', 'loss_task', 'loss_kl', 'fc',
        'loss_unlabeled', 'loss_reconstruction', 'loss_classifier', 'loss_entropy', 'loss_y_logprob',
        'loss_y_kl', 'loss_z_kl', 'loss_y_kl_uniform'
    ]

    metrics_dfs = [load_metrics_csv_as_df(metrics_file, metrics_list, expt_ids=expt_ids)]
    metrics_df = pd.concat(metrics_dfs, sort=False)

    if isinstance(expt_ids, list) and len(expt_ids) > 1:
        hue = 'dataset'
    else:
        hue = None

    sns.set_style('white')
    sns.set_context('talk')
    data_queried = metrics_df[
        (metrics_df.epoch > 10) & ~pd.isna(metrics_df.val) & (metrics_df.dtype == dtype)]
    g = sns.FacetGrid(
        data_queried, col='loss', col_wrap=2, hue=hue, sharey=False, height=4)
    g = g.map(plt.plot, 'epoch', 'val').add_legend()

    if save_file is not None:
        make_dir_if_not_exists(save_file)
        g.savefig(save_file + '.' + format, dpi=300, format=format)

    plt.close()


@typechecked
def load_metrics_csv_as_df(
        metric_file: str,
        metrics_list: List[str],
        expt_ids: Optional[List[str]] = None,
        test: bool = False
) -> pd.DataFrame:
    """Load metrics csv file and return as a pandas dataframe for easy plotting.

    Parameters
    ----------
    metric_file : str
        csv file saved during training
    metrics_list : list
        names of metrics to pull from csv; do not prepend with 'tr', 'val', or 'test'
    expt_ids : list, optional
        dataset names for easier parsing
    test : bool, optional
        True to only return test values (computed once at end of training)

    Returns
    -------
    pandas.DataFrame object

    """

    metrics = pd.read_csv(metric_file)

    # collect data from csv file
    metrics_df = []
    for i, row in metrics.iterrows():

        if row['dataset'] == -1:
            dataset = 'all'
        elif expt_ids is not None:
            dataset = expt_ids[int(row['dataset'])]
        else:
            dataset = row['dataset']

        if test:
            test_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'test'}
            for metric in metrics_list:
                name = 'test_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**test_dict, 'loss': metric, 'val': row[name]}, index=[0]))
        else:
            # make dict for val data
            val_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'val'}
            for metric in metrics_list:
                name = 'val_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**val_dict, 'loss': metric, 'val': row[name]}, index=[0]))
            # make dict for train data
            tr_dict = {'dataset': dataset, 'epoch': row['epoch'], 'dtype': 'train'}
            for metric in metrics_list:
                name = 'tr_%s' % metric
                if name not in row.keys():
                    continue
                metrics_df.append(pd.DataFrame(
                    {**tr_dict, 'loss': metric, 'val': row[name]}, index=[0]))

    return pd.concat(metrics_df, sort=True)
