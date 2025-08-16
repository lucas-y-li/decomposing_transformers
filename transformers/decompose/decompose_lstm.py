import torch
from lstm_model import RNNModel
import numpy as np
import itertools
from decompose_util import decomp_activation

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DecomposedLSTM:
    def __init__(self, model: RNNModel, shapley_include_bias=False, generalized=True):
        """
        shapley_include_bias: includes bias as a free index the Shapley permutation if True, or fixed if False
        generalized: if True, then use Generalized Contextual Decomposition (Jumulet et al., 2019), else use Contextual Decomposition (Murdoch et al., 2018)
        """
        self.model = model.to(device)
        self.rnn_weights = self.model.rnn._parameters
        self.dtype = next(model.parameters()).dtype
        self.shapley_include_bias = shapley_include_bias
        self.generalized = generalized

    def _init_hidden(self, batch_size, init_in_beta):
        hidden_size = self.model.nhid
        num_layers = self.model.nlayers
        # beta states set to zero
        h_t_minus_1_beta = torch.zeros(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        h_t_minus_1_gamma = torch.zeros(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        c_t_minus_1_beta = torch.zeros(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        c_t_minus_1_gamma = torch.zeros(
            (num_layers, batch_size, hidden_size)).to(self.dtype)

        with torch.no_grad():
            # activations for ". <eos>"
            init_phrase = torch.LongTensor([18, 19]).unsqueeze(0)
            embed = self.model.encoder(init_phrase)
            _, (h_n, c_n) = self.model.rnn(embed)

        if init_in_beta:
            h_t_minus_1_beta = h_t_minus_1_beta + h_n
            c_t_minus_1_beta = c_t_minus_1_beta + c_n
        else:
            # gamma states set to 'initial lstm states'
            h_t_minus_1_gamma = h_t_minus_1_gamma + h_n
            c_t_minus_1_gamma = c_t_minus_1_gamma + c_n
        return h_t_minus_1_beta, h_t_minus_1_gamma, c_t_minus_1_beta, c_t_minus_1_gamma

    def __call__(self, inputs, beta_mask, init_in_beta=False):
        embeddings = self.model.encoder(inputs).to(self.dtype)
        beta_mask = beta_mask.to(self.dtype)
        batch_size, tokens, hidden_size = embeddings.shape
        num_layers = self.model.nlayers

        assert beta_mask.shape == (batch_size, tokens)

        beta_embeddings = embeddings * beta_mask.unsqueeze(2)
        gamma_embeddings = embeddings - beta_embeddings

        h_t_beta = torch.empty(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        h_t_gamma = torch.empty(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        c_t_beta = torch.empty(
            (num_layers, batch_size, hidden_size)).to(self.dtype)
        c_t_gamma = torch.empty(
            (num_layers, batch_size, hidden_size)).to(self.dtype)

        # save hidden from previous layers
        h_t_minus_1_beta, h_t_minus_1_gamma, c_t_minus_1_beta, c_t_minus_1_gamma = self._init_hidden(
            batch_size, init_in_beta)

        output_beta = []
        output_gamma = []

        for t in range(tokens):
            inside = beta_mask[:, t].unsqueeze(-1)
            for layer in range(num_layers):
                # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
                w_ii, w_if, w_ig, w_io = np.split(
                    self.rnn_weights[f"weight_ih_l{layer}"], 4, 0)
                w_hi, w_hf, w_hg, w_ho = np.split(
                    self.rnn_weights[f"weight_hh_l{layer}"], 4, 0)
                b_ii, b_if, b_ig, b_io = np.split(
                    self.rnn_weights[f"bias_ih_l{layer}"], 4, 0)
                b_hi, b_hf, b_hg, b_ho = np.split(
                    self.rnn_weights[f"bias_hh_l{layer}"], 4, 0)

                if layer == 0:
                    beta_x = beta_embeddings[:, t]
                    gamma_x = gamma_embeddings[:, t]
                else:
                    # use the gamma, beta from the previous layer
                    beta_x = beta_h
                    gamma_x = gamma_h

                beta_h = h_t_minus_1_beta[layer]
                gamma_h = h_t_minus_1_gamma[layer]
                beta_c = c_t_minus_1_beta[layer]
                gamma_c = c_t_minus_1_gamma[layer]

                beta_i = beta_h @ torch.t(w_hi)
                gamma_i = gamma_h @ torch.t(w_hi)

                beta_f = beta_h @ torch.t(w_hf)
                gamma_f = gamma_h @ torch.t(w_hf)

                beta_g = beta_h @ torch.t(w_hg)
                gamma_g = gamma_h @ torch.t(w_hg)

                beta_o = beta_h @ torch.t(w_ho)
                gamma_o = gamma_h @ torch.t(w_ho)

                beta_i += beta_x @ torch.t(w_ii)
                beta_f += beta_x @ torch.t(w_if)
                beta_g += beta_x @ torch.t(w_ig)
                beta_o += beta_x @ torch.t(w_io)

                gamma_i += gamma_x @ torch.t(w_ii)
                gamma_f += gamma_x @ torch.t(w_if)
                gamma_g += gamma_x @ torch.t(w_ig)
                gamma_o += gamma_x @ torch.t(w_io)

                # apply activations
                beta_i, gamma_i, bias_i = self.decomp_activation_three(
                    beta_i, gamma_i, b_ii + b_hi, torch.sigmoid)
                beta_g, gamma_g, bias_g = self.decomp_activation_three(
                    beta_g, gamma_g, b_ig + b_hg, torch.tanh)
                o = torch.sigmoid(beta_o + gamma_o + b_io + b_ho)

                # element-wise products
                # LSTM eq. 5 (calculate next cell state)
                if self.generalized:
                    # GCD implementation
                    # forget gate determines what information from c_{t-1} is kept
                    # input gate determines what information from g_t is kept
                    # GCD allows full f and i context

                    # f_t * c_{t-1}
                    f = torch.sigmoid(beta_f + gamma_f + b_if + b_hf)
                    beta_c = beta_c * f
                    gamma_c = gamma_c * f

                    # i_t * g_t
                    i = beta_i + gamma_i + bias_i
                    beta_c += beta_g * i + bias_g * beta_i
                    gamma_c += gamma_g * i + bias_g * gamma_i

                else:
                    # CD implementation
                    # https://github.com/jamie-murdoch/ContextualDecomposition/blob/master/train_model/sent_util.py
                    beta_f, gamma_f, bias_f = self.decomp_activation_three(
                        beta_f, gamma_f, b_if + b_hf, torch.sigmoid)

                    # f_t * c_{t-1}
                    beta_c = (beta_f + bias_f) * beta_c
                    gamma_c = (beta_f + gamma_f + bias_f) * \
                        gamma_c + gamma_f * beta_c

                    # i_t * g_t
                    beta_c += beta_i * (beta_g + bias_g) + bias_i * beta_g
                    gamma_c += gamma_i * \
                        (beta_g + gamma_g + bias_g) + \
                        (beta_i + bias_i) * gamma_g

                # add to beta_c if inside, else add to gamma_c
                beta_c += bias_g * bias_i * inside
                gamma_c += bias_g * bias_i * (1 - inside)

                # LSTM eq. 6 (output gate)
                beta_ht, gamma_ht = self.decomp_activation_two(
                    beta_c, gamma_c, torch.tanh)

                beta_h = beta_ht * o
                gamma_h = gamma_ht * o

                h_t_beta[layer] = beta_h
                h_t_gamma[layer] = gamma_h
                c_t_beta[layer] = beta_c
                c_t_gamma[layer] = gamma_c

            output_beta.append(beta_h)
            output_gamma.append(gamma_h)
            c_t_minus_1_beta = c_t_beta
            c_t_minus_1_gamma = c_t_gamma
            h_t_minus_1_beta = h_t_beta
            h_t_minus_1_gamma = h_t_gamma

        # calculate decoder output
        # as input, takes hidden values of final layer from all input tokens (flattened)
        # LSTM eq. 7
        w_d = self.model.decoder.weight
        bias_z = self.model.decoder.bias

        output_beta = torch.stack(output_beta)
        output_gamma = torch.stack(output_gamma)
        flattened_beta = output_beta.view(
            output_beta.shape[0] * output_beta.shape[1], output_beta.shape[2])
        flattened_gamma = output_gamma.view(
            output_gamma.shape[0] * output_gamma.shape[1], output_gamma.shape[2])

        beta_z = flattened_beta @ torch.t(w_d)
        gamma_z = flattened_gamma @ torch.t(w_d)
        beta_z = beta_z.view(
            output_beta.shape[0], output_beta.shape[1], beta_z.shape[1])
        gamma_z = gamma_z.view(
            output_gamma.shape[0], output_gamma.shape[1], gamma_z.shape[1])

        return beta_z.transpose(0, 1), gamma_z.transpose(0, 1), bias_z

    @classmethod
    def from_pretrained(cls, state_dict_path, dtype=None,
                        shapley_include_bias=False, generalized=True):
        state_dict = torch.load(
            state_dict_path, weights_only=True)
        # minus encoder_weight, decoder_bias, decoder_weight
        # / 4 (lstm weights)
        num_layers = (len(state_dict.keys()) - 3) // 4
        vocab_size, input_size = state_dict["encoder.weight"].shape
        output_size, hidden_size = state_dict["decoder.weight"].shape

        model = RNNModel(
            rnn_type="LSTM",
            ntoken=vocab_size,
            ninp=input_size,
            nhid=hidden_size,
            nlayers=num_layers
        )
        if dtype is not None:
            model = model.to(dtype)

        model.load_state_dict(state_dict)
        model.eval()
        return DecomposedLSTM(model, shapley_include_bias=shapley_include_bias,
                              generalized=generalized)

    def decomp_activation_three(self, a, b, c, activation):
        """
        Linearize nonlinear activation function with three inputs using Shapely approximation
        """
        if self.shapley_include_bias:
            a_contrib = 1/3 * (activation(a)) + \
                1/6 * (activation(a + b) - activation(b)) + \
                1/6 * (activation(a + c) - activation(c)) + \
                1/3 * (activation(a + b + c) - activation(b + c))
            b_contrib = 1/3 * (activation(b)) + \
                1/6 * (activation(b + c) - activation(c)) + \
                1/6 * (activation(b + a) - activation(a)) + \
                1/3 * (activation(a + b + c) - activation(a + c))
            c_contrib = 1/3 * (activation(c)) + \
                1/6 * (activation(c + a) - activation(a)) + \
                1/6 * (activation(c + b) - activation(b)) + \
                1/3 * (activation(a + b + c) - activation(a + b))

        else:
            # https://github.com/jamie-murdoch/ContextualDecomposition/blob/02858835e51d98ee378fefe62c1487a215934ba1/train_model/sent_util.py#L168
            a_contrib = 0.5 * (activation(a + c) - activation(c) +
                               activation(a + b + c) - activation(b + c))
            b_contrib = 0.5 * (activation(b + c) - activation(c) +
                               activation(a + b + c) - activation(a + c))
            c_contrib = activation(c)

        assert torch.allclose(a_contrib + b_contrib +
                              c_contrib, activation(a + b + c), atol=1e-5)
        return a_contrib, b_contrib, c_contrib

    @staticmethod
    def decomp_activation_two(a, b, activation):
        """
        Linearize nonlinear activation function with two inputs using Shapely approximation
        """
        a_contrib = 0.5 * (activation(a + b) - activation(b) +
                           activation(a))
        b_contrib = 0.5 * (activation(a + b) - activation(a) +
                           activation(b))

        assert torch.allclose(a_contrib + b_contrib,
                              activation(a + b),  atol=1e-5)
        return a_contrib, b_contrib
