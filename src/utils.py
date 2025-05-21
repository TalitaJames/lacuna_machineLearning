'''
Made by Talita James,
Where noted, inspired or adapted from other work
'''
import numpy as np
import torch.nn as nn


def to_np(t):
    '''Coppied from:  https://github.com/denisyarats/pytorch_sac'''
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()



def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    '''
    Coppied from:  https://github.com/denisyarats/pytorch_sac
    Create a multi-layer perceptron (MLP) with ReLU activations.
    Params:
    - input_dim: int, input dimension
    - hidden_dim: int, hidden layer dimension
    - output_dim: int, output dimension
    - hidden_depth: int, number of hidden layers
    - output_mod: nn.Module, optional output modification layer
    Returns:
    - trunk: nn.Sequential, the MLP model
    '''

    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        # Create the first layer
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1): # Create the hidden layers
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim)) # add the final layer
    if output_mod is not None:
        mods.append(output_mod)

    return nn.Sequential(*mods)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)