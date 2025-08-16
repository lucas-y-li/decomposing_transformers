import torch
import itertools
from torch import nn
from torch.nn import functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LayerNormDecomposed:
    def __init__(self, layer, debug=False):
        # layer: RobertaLayerNorm, BertLayerNorm, etc.
        self.layer = layer.to(torch.torch.double)
        self.weight = layer.gamma
        self.bias = layer.beta
        self.variance_epsilon = layer.variance_epsilon
        self.debug = debug

    def __call__(self, xs):
        x = torch.sum(xs, dim=0)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        div = torch.sqrt(s + self.variance_epsilon)

        x_components = xs[:-1, :]
        x_bias = xs[-1, :].unsqueeze(0)

        u_components = x_components.mean(-1, keepdim=True)
        x_components = (x_components - u_components) / div
        y_components = self.weight.unsqueeze(0) * x_components

        u_bias = x_bias.mean(-1, keepdim=True)
        x_bias = (x_bias - u_bias) / div
        y_bias = self.weight.unsqueeze(0) * x_bias + self.bias

        ys = torch.cat([y_components, y_bias], dim=0)

        if self.debug:
            # verify correctness
            y = self.layer(x)
            assert torch.allclose(y, torch.sum(ys, dim=0))

        return ys


class LinearDecomposed:
    def __init__(self, layer: nn.Linear, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.weight = layer.weight
        self.bias = layer.bias
        self.debug = debug

    def __call__(self, xs):
        # linear : hidden_state @ weight + beta
        # decomposed : hidden_state = hidden_beta + hidden_gamma
        # decomposed linear: hidden_beta @ weight + hidden_gamma @ weight + beta
        # shape = [batch_size, seq_len, hidden_size]

        x_components = xs[:-1, :]
        x_bias = xs[-1, :]

        y_components = F.linear(x_components, self.weight, bias=None)
        y_bias = F.linear(x_bias, self.weight, bias=self.bias).unsqueeze(0)

        ys = torch.cat([y_components, y_bias], dim=0)

        if self.debug:
            # verify correctness
            x = torch.sum(xs, dim=0)
            y = self.layer(x)

            try:
                assert torch.allclose(y, torch.sum(
                    ys, dim=0), atol=1e-5)
            except:
                print(y)
                print(ys)
                print(y - ys)

        return ys


def decomp_activation_fixed(xs, activation, debug=False, perms=None):
    """
    Use permutations that _exclude_ the bias (fixed setting), following Murdoch et al. (2018)
    """
    if debug:
        contributions_ = activation(torch.sum(xs, axis=0)).to(device)

    xs_components = xs[:-1]
    xs_bias = xs[-1].clone()

    num_contributions = xs_components.shape[0]

    if debug or perms is None:
        perms_ = torch.tensor(
            list(itertools.permutations(torch.arange(num_contributions)))).to(device)
        if perms is None:
            perms = perms_
        if debug:
            assert torch.equal(perms, perms_)

    biases = xs_bias.unsqueeze(0)
    xs[-1] = 0

    contributions = torch.empty(
        (num_contributions + 1, *xs_components.shape[1:])
    ).to(torch.device('cpu')).to(torch.torch.double)

    for k in range(num_contributions):
        # position of k in each permutation
        n_idx = torch.argwhere(perms == k)[:, 1].unsqueeze(0).T

        including_idx = perms.clone()
        # in each permutation, zero out values with index > n_idx
        including_idx[torch.where(
            torch.arange(num_contributions).unsqueeze(0) > n_idx)] = -1

        # sum the x-values at these indices and apply activation
        including_values = torch.sum(xs[including_idx], axis=1) + biases
        including_activation = torch.mean(
            activation(including_values), axis=0
        )
        # excluding_values is the including_values, minus the k values
        excluding_activation = torch.mean(
            activation(including_values - xs[k]), axis=0
        )

        contributions[k] = including_activation - excluding_activation
    # xs_bias is not included in the permutation, so we keep it separate
    contributions[-1] = activation(xs_bias)

    if debug:
        assert torch.allclose(
            torch.sum(contributions, axis=0),
            contributions_, atol=1e-5
        )
    return contributions.to(device)


def decomp_activation_all(xs, activation, debug=False, perms=None):
    """
    Use permutations that _include_ the bias, following Jumulet et al. (2019)
    """
    if debug:
        contributions_ = activation(torch.sum(xs, axis=0)).to(device)

    num_contribution_plus_one = xs.shape[0]
    if debug or perms is None:
        perms_ = torch.tensor(
            list(itertools.permutations(torch.arange(num_contribution_plus_one)))).to(device)
        if perms is None:
            perms = perms_
        if debug:
            assert torch.equal(perms, perms_)

    xs_padded = torch.concat([xs, torch.zeros((1, *xs.shape[1:]))])

    contributions = torch.empty_like(xs).to(
        torch.device('cpu')).to(torch.torch.double)

    for k in range(num_contribution_plus_one):
        # position of k in each permutation
        n_idx = torch.argwhere(perms == k)[:, 1].unsqueeze(0).T

        including_idx = perms.clone()
        # in each permutation, zero out values with index > n_idx
        including_idx[torch.where(
            torch.arange(num_contribution_plus_one).unsqueeze(0) > n_idx)] = -1

        # sum the x-values at these indices and apply activation
        including_values = torch.sum(xs_padded[including_idx], axis=1)
        including_activation = torch.mean(
            activation(including_values), axis=0
        )
        # excluding_values is the including_values, minus the k values
        excluding_activation = torch.mean(
            activation(including_values - xs_padded[k]), axis=0
        )

        contributions[k] = including_activation - excluding_activation

    if debug:
        assert torch.allclose(
            torch.sum(contributions, axis=0),
            contributions_, atol=1e-5
        ), torch.max(torch.abs(torch.sum(contributions, axis=0) - contributions_))
    return contributions.to(device)


def decomp_activation(xs, activation, debug=False, perms=None, setting=None):
    """
    Estimate contributions towards non-linear activation function using Shapley Values: https://en.wikipedia.org/wiki/Shapley_value

    setting: whether to use fixed-bias permutations (`fixed`) or all permutations (`all`)
    """
    if perms is not None:
        # number of contributions (eg. beta, gamma) + 1 (from bias)
        num_contribution_plus_one = xs.shape[0]
        num_permutations, permuted_size = perms.shape

        if num_contribution_plus_one == permuted_size:
            # biases were included in the permutations
            return decomp_activation_all(xs, activation, debug=debug, perms=perms)
        elif num_contribution_plus_one - 1 == permuted_size:
            # biases were not included in the permutations
            return decomp_activation_fixed(xs, activation, debug=debug, perms=perms)
        else:
            raise ValueError(
                f"Permutations must be of the set [num_contributions] ({xs.shape[0]}) or [num_contributions + 1] ({xs.shape[0] + 1}), instead got: {permuted_size}"
            )
    elif setting is not None:
        print("Warning: you should probably set the permutations")
        if setting == "fixed":
            return decomp_activation_fixed(xs, activation, debug=debug)
        elif setting == "all":
            return decomp_activation_all(xs, activation, debug=debug)
        else:
            raise ValueError(
                "Setting should be either `fixed` or `all`, instead got: ", setting
            )
    else:
        raise ValueError(
            "Need to set either the permutations (perm) or the permutation setting (setting)"
        )
