import torch
import numpy as np
from scipy import integrate
from nets import DiffusionScore, SimpleMLP
import design_bench
from utils import *
from scipy.stats import gaussian_kde
from torch.distributions.normal import Normal


mysoftmax = torch.nn.Softmax(dim=0)

def sum_norm(prob):
    return prob/torch.sum(prob)

def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  #input NxD
  #output (NxD, )
  return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def prior_logp_func(z):
  shape = z.shape
  D = np.prod(shape[1:])
  logps = -D / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1)) / 2.
  return logps

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
  #divergence == trace of jacob == Hutchinson-Skilling trace estimation
  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn

class StopCondition:
    def __init__(self, max_nfev):
        self.max_nfev = max_nfev
        self.nfev = 0
        self.terminal = True  # Set as an instance attribute

    def __call__(self, t, y):
        self.nfev += 1
        if self.nfev > self.max_nfev:
            return 0
        else:
            return 1
def get_likelihood_fn(sde, hutchinson_type='Gaussian',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, y, t):
      return -model.gen_sde.mu_ode(t, x, y, gamma=0.0)

  def div_fn(model, x, y, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, y, tt))(x, t, noise)

  def likelihood_fn(model, data, label0):
      # data NxD; label0 Nx1.
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data[0, :])
        epsilon = epsilon.repeat(data.shape[0], 1)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data[0, :], low=0, high=2).float() * 2 - 1.
        epsilon = epsilon.repeat(data.shape[0], 1)
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        # (N*D + N,)
        sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        #x[:-shape[0]]: (NxD, ); sample: (N, D)
        vec_t = torch.ones(sample.shape[0], device=sample.device).reshape(-1, 1) * t
        #vec_t: (N, 1); label0: (N, 1)
        drift = to_flattened_numpy(drift_fn(model, sample, label0, vec_t))
        #drift: (NxD, )
        logp_grad = to_flattened_numpy(div_fn(model, sample, label0, vec_t, epsilon))
        #logp_grad: (N, )
        return np.concatenate([drift, logp_grad], axis=0)
        #(N*D + N,)

      init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      stop_condition = StopCondition(max_nfev=500)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol * 0.1, atol=atol * 0.1, method="RK23", events=stop_condition)
      zp = solution.y[:, -1]
      z = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = prior_logp_func(z)

      x_logp = prior_logp + delta_logp
      return x_logp

  return likelihood_fn
