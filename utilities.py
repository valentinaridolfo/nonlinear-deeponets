import matplotlib.pyplot as plt
import torch
import numpy as np
import random


device = torch.device('cpu')
torch.set_default_device(device)
np.random.seed(0)
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

#########################################
# L^p relative loss for N-D functions
#########################################
class LprelLoss(): 
    """ 
    Sum of relative errors in L^p norm 
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, *n, d_u)
          where *n indicates that the spatial dimensions can be arbitrary
    """
    def __init__(self, p:int, size_mean=False):
        self.p = p
        self.size_mean = size_mean

    def rel(self, x, y):
        num_examples = x.size(0)
        
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), p=self.p, dim=1)
        
        # check division by zero
        if torch.any(y_norms <= 1e-5):
            raise ValueError("Division by zero")
        
        if self.size_mean is True:
            return torch.mean(diff_norms/y_norms)
        elif self.size_mean is False:
            return torch.sum(diff_norms/y_norms) # sum along batchsize
        elif self.size_mean is None:
            return diff_norms/y_norms # no reduction
        else:
            raise ValueError("size_mean must be a boolean or None")
    
    def __call__(self, x, y):
        return self.rel(x, y)


#########################################
# Functions for visualization
#########################################

import numpy as np
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, funcs,selected_indices=[1, 2, 4]):
    idx = 0
    fig, axs = plt.subplots(1, len(selected_indices), figsize=(10, 5))
    for k, v in funcs.items():
        for i in selected_indices:
            axs[idx].plot(x.cpu().numpy(), v[i].cpu().numpy(), label=f'{k} {i+1}')
            axs[idx].set_xlabel('x')
            axs[idx].set_title(k)
            axs[idx].legend()
            axs[idx].grid()
        idx += 1
    plt.tight_layout()
    plt.show()




import matplotlib.pyplot as plt

def plot_results(x, true_solutions, predicted_solutions, tr, selected_indices=[1, 2, 4]):
    true_solutions = true_solutions.to('cpu')
    predicted_solutions = predicted_solutions.to('cpu')
    fig, axs = plt.subplots(1, len(selected_indices), figsize=(12, 5))
    for i, idx in enumerate(selected_indices):
        axs[i].plot(x, true_solutions[idx, :], label='Exact', linestyle='--', linewidth=1)
        axs[i].plot(x, predicted_solutions[idx, :], label='Computed', marker='.', linewidth=1)
        axs[i].set_xlabel('x', fontsize=14)
        axs[i].tick_params(axis='both', labelsize=12)
        axs[i].legend(fontsize=12)
        axs[i].grid()
    fig.suptitle(tr, fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    plt.show()
