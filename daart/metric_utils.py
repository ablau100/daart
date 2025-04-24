# This file contains the implementation of the r2 score metric
from torcheval.metrics import R2Score
import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import gammaln
import logging
from sklearn.metrics import r2_score as r2_score_sklearn

logger = logging.getLogger(__name__)

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def clip_contrastive_loss(similarity_matrix):
    """
    Compute CLIP's contrastive loss given a similarity matrix.
    The matrix contains cosine similarities of two sets of features.
    """
    labels = torch.arange(len(similarity_matrix)).to(similarity_matrix.device)
    forw_correct  = topk(similarity_matrix, labels, k=1)
    back_correct  = topk(similarity_matrix.t(), labels, k=1)
    loss_i = F.cross_entropy(similarity_matrix, labels)
    loss_t = F.cross_entropy(similarity_matrix.t(), labels)
    return {
        "contrast_loss": (loss_i + loss_t) / 2,
        "percent_correct": (forw_correct + back_correct) / 2,
        "forw_correct": forw_correct,
        "back_correct": back_correct,
        "loss_f": loss_i,
        "loss_b": loss_t
    }

def neg_log_likelihood(rates, spikes, zero_warning=True, threshold=1e-9):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = threshold
    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)

def bits_per_spike(rates, spikes, threshold=1e-9):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes, threshold=threshold)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, threshold=threshold)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def batch_wise_sim_matrix(z):
    sim_matrix = z @ z.T
    return sim_matrix

def batch_wise_contrastive_loss(sim_matrix):
    N = sim_matrix.shape[0]
    # remove the diagonal from the sim_matrix
    mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
    sim_matrix = sim_matrix[~mask].view(N, N-1)
    labels = torch.arange(N).to(sim_matrix.device)
    labels_i, labels_j = labels[:N//2], labels[N//2:] -1
    labels = torch.cat([labels_j, labels_i]).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    percent_correct = topk(sim_matrix, labels, k=1)
    return{
        "contrast_loss": loss,
        "percent_correct": percent_correct
    }

def compute_varexp(y_true, y_pred):
    """variance explained of y_true by y_pred across axis=0"""
    y_var = ((y_true - y_true.mean(axis=0)) ** 2).mean(axis=0)
    residual = ((y_true - y_pred) ** 2).mean(axis=0)
    varexp = 1 - residual / y_var
    return varexp    

# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}

    if "bps" in metrics:
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        bps_list = []
        for i in range(gt.shape[-1]): 
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    
    if "r2" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2s = [r2_score(y_true=gt[i].T[k], y_pred=pred[i].T[k], device=device) for k in range(len(gt[i].T))]
            r2_list.append(np.ma.masked_invalid(r2s).mean())
        r2 = np.mean(r2_list)
        results["r2"] = r2
        
    if "rsquared" in metrics:
        r2 = 0
        for i in range(gt.shape[-1]):
            r2_list = []
            for j in range(gt.shape[0]):
                r2 = r2_score(y_true=gt[j,:,i], y_pred=pred[j,:,i], device=device) 
                r2_list.append(r2)
            r2 += np.mean(r2_list)
        results["rsquared"] = r2 / gt.shape[-1]
        
    return results

# get neuron metrics results
def compute_neuron_metrics(data_dict, metrics=["bps", "r2", "ve"], norm=True):
    results = {}
    gt, pred = data_dict["gt"], data_dict["pred"]
    norm_gt, norm_pred = data_dict["norm_gt"], data_dict["norm_pred"]
    # loop over the neurons
    if "bps" in metrics:
        num_neurons = gt.shape[-1]
        bps_list = []
        for i in range(num_neurons):
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        bps_list = np.array(bps_list)
        results["bps"] = bps_list

    if "ve" in metrics:
        # reshape the gt and pred
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        ven = compute_varexp(y_true=_gt, y_pred=_pred)
        results["ve"] = ven

    if "r2" in metrics:
        if norm:
            _gt, _pred = norm_gt, norm_pred
        else:
            _gt, _pred = gt, pred
        _gt = _gt.reshape(-1, _gt.shape[-1])
        _pred = _pred.reshape(-1, _pred.shape[-1])
        r2 = r2_score_sklearn(y_true=_gt, y_pred=_pred, multioutput="raw_values")
        results["r2"] = r2
    return results