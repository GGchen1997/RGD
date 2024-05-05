import numpy as np

import random
import copy
import torch
import torch.backends.cudnn as cudnn



def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



y_min = None
y_max = None

def load_y(task_name):
    global y_min
    global y_max
    dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    y_min, y_max = dic2y[task_name]


def process_data(task, task_name, task_y0=None):
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                     'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                     'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
        task_x = task.to_logits(task.x)
        if task_name == 'TFBind10-Exact-v0':
            interval = np.arange(0, 4161482, 830, dtype=int)[0: 5000]
            index = np.argsort(task_y0.squeeze())
            index = index[interval]
            task_y0 = task_y0[index]
            task_x = task_x[index]
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'Ackley', 'Rastrigin', 'Rosenbrock']:
        task_x = copy.deepcopy(task.x)
    task_x = task.normalize_x(task_x)
    shape0 = task_x.shape
    task_x = task_x.reshape(task_x.shape[0], -1)
    task_y = task.normalize_y(task_y0)
    return task_x, task_y, shape0

def evaluate_sample(task, x_init, task_name, shape0):
    x_init = x_init.cpu().numpy()
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                     'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                     'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
        X1 = x_init.reshape(-1, shape0[1], shape0[2])
    elif task_name in ['Superconductor-RandomForest-v0', 'HopperController-Exact-v0',
                       'AntMorphology-Exact-v0', 'DKittyMorphology-Exact-v0', 'Ackley', 'Rastrigin', 'Rosenbrock']:
        X1 = x_init
    X1 = task.denormalize_x(X1)
    if task_name in ['TFBind8-Exact-v0', 'GFP-Transformer-v0', 'UTR-ResNet-v0',
                     'CIFARNAS-Exact-v0', 'TFBind10-Exact-v0',
                     'ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0']:
        X1 = task.to_integers(X1)
    Y1 = task.predict(X1)
    Y1_norm = (Y1 - y_min)/(y_max - y_min)
    print("mean", np.mean(Y1_norm))
    max_v = (np.max(Y1) - y_min) / (y_max - y_min)
    med_v = (np.median(Y1) - y_min) / (y_max - y_min)
    return max_v, med_v

def adjust_learning_rate(optimizer, lr0, epoch, T):
    lr = lr0 * (1 + np.cos((np.pi * epoch * 1.0) / (T * 1.0))) / 2.0
    print("epoch {} lr {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_pcc(valid_preds, valid_labels):
    vx = valid_preds - torch.mean(valid_preds)
    vy = valid_labels - torch.mean(valid_labels)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + 1e-12) * torch.sqrt(torch.sum(vy ** 2) + 1e-12))
    return pcc
def heun_sampler_ode_hyper(sde, x_0, ya, num_steps=1000, classifier=None, classifier_s=None, gamma_learn=True, direct_grad=False, sample_lr=0.01):
    device = sde.gen_sde.T.device
    batch_size = x_0.size(0)
    ndim = x_0.dim() - 1
    T_ = sde.gen_sde.T.cpu().item()
    delta = T_ / num_steps
    ts = torch.linspace(0, 1, num_steps + 1) * T_

    # sample
    x_t = x_0.detach().clone().to(device)
    t = torch.zeros(batch_size, *([1] * ndim), device=device)
    t_n = torch.zeros(batch_size, *([1] * ndim), device=device)#t_next
    # bs * 1
    #with torch.no_grad():
    #gamma = torch.nn.Parameter(2 * torch.ones(batch_size, 1), requires_grad=True).to(device)
    gamma = 2 * torch.ones(batch_size, 1).to(device)
    #gamma = 2 * torch.ones_like(x_t.data)
    gamma.requires_grad = True
    opt = torch.optim.Adam([gamma], lr=sample_lr)
    #opt = torch.optim.SGD([gamma], lr=0.01)
    for i in range(num_steps):
        t.fill_(ts[i].item())
        if i < num_steps - 1:
            t_n.fill_(ts[i + 1].item())

        x_t0 = copy.deepcopy(x_t.detach())
        if gamma_learn:
            mu = sde.gen_sde.mu_ode(t, x_t, ya, gamma=gamma)
            x_t = x_t + mu * delta
            # one step update of Euler Maruyama method with a step size delta
            # Additional terms for Heun's method
            if i < num_steps - 1:
                mu2 = sde.gen_sde.mu_ode(t_n,
                                     x_t,
                                     ya,
                                     gamma=gamma)
                x_t = x_t + (mu2 - mu)/2 * delta
            #transform x
            s = sde.gen_sde.a(x_t, t, ya).detach()
            var = sde.gen_sde.base_sde.var(t)
            mean_weight = sde.gen_sde.base_sde.mean_weight(t)
            x_t = (x_t + var * s)/mean_weight
            #optimize gamma
            loss = - torch.sum(classifier(x_t)[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            #gamma.data = torch.clip(gamma.data, 0, 1000000000)
            gamma.data = torch.clamp(gamma.detach(), min=0)
        x_t = x_t0.detach()
        mu = sde.gen_sde.mu_ode(t, x_t, ya, gamma=gamma.detach())
        x_t = x_t + mu * delta
        # one step update of Euler Maruyama method with a step size delta
        # Additional terms for Heun's method
        if i < num_steps - 1:
            mu2 = sde.gen_sde.mu_ode(t_n,
                                 x_t,
                                 ya,
                                 gamma=gamma.detach())
            x_t = x_t + (mu2 - mu) / 2 * delta

        if direct_grad:
            x_t = copy.deepcopy(x_t.detach())
            x_t.requires_grad = True
            x_t_opt = torch.optim.SGD([x_t], lr=0.1)
            s = sde.gen_sde.a(x_t, t, ya).detach()
            var = sde.gen_sde.base_sde.var(t)
            mean_weight = sde.gen_sde.base_sde.mean_weight(t)
            x_t_ = (x_t + var * s) / mean_weight
            # optimize x_t
            loss = - torch.sum(classifier(x_t_)[0])
            x_t_opt.zero_grad()
            loss.backward()
            x_t_opt.step()


    return x_t.detach().to(device)


## REWEIGHTING
def adaptive_temp_v2(scores_np, q=None):
    """Calculate an adaptive temperature value based on the
    statistics of the scores array

    Args:

    scores_np: np.ndarray
        an array that represents the vectorized scores per data point

    Returns:

    temp: np.ndarray
        the scalar 90th percentile of scores in the dataset
    """

    inverse_arr = scores_np
    max_score = inverse_arr.max()
    scores_new = inverse_arr - max_score
    if q is None:
        quantile_ninety = np.quantile(scores_new, q=0.9)
    else:
        quantile_ninety = np.quantile(scores_new, q=q)
    return np.maximum(np.abs(quantile_ninety), 0.001)


def softmax(arr, temp=1.0):
    """Calculate the softmax using numpy by normalizing a vector
    to have entries that sum to one

    Args:

    arr: np.ndarray
        the array which will be normalized using a tempered softmax
    temp: float
        a temperature parameter for the softmax

    Returns:

    normalized: np.ndarray
        the normalized input array which sums to one
    """

    max_arr = arr.max()
    arr_new = arr - max_arr
    exp_arr = np.exp(arr_new / temp)
    return exp_arr / np.sum(exp_arr)


def get_weights(scores, temp='90'):
    """Calculate weights used for training a model inversion
    network with a per-sample reweighted objective

    Args:

    scores: np.ndarray
        scores which correspond to the value of data points in the dataset

    Returns:

    weights: np.ndarray
        an array with the same shape as scores that reweights samples
    """

    scores_np = scores[:, 0]
    hist, bin_edges = np.histogram(scores_np, bins=20)
    hist = hist / np.sum(hist)

    # if base_temp is None:
    if temp == '90':
        base_temp = adaptive_temp_v2(scores_np, q=0.9)
    elif temp == '75':
        base_temp = adaptive_temp_v2(scores_np, q=0.75)
    elif temp == '50':
        base_temp = adaptive_temp_v2(scores_np, q=0.5)
    else:
        raise RuntimeError("Invalid temperature")
    softmin_prob = softmax(bin_edges[1:], temp=base_temp)

    provable_dist = softmin_prob * (hist / (hist + 1e-3))
    provable_dist = provable_dist / (np.sum(provable_dist) + 1e-7)
    #print(provable_dist)

    bin_indices = np.digitize(scores_np, bin_edges[1:])
    hist_prob = hist[np.minimum(bin_indices, 19)]

    weights = provable_dist[np.minimum(bin_indices, 19)] / (hist_prob + 1e-7)
    weights = np.clip(weights, a_min=0.0, a_max=5.0)
    return weights.astype(np.float32)[:, np.newaxis]

