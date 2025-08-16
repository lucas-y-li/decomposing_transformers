import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import itertools
import transformers
from collections import OrderedDict

from gpt2_model import *
from decompose_util import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Conv1DDecomposed:
    def __init__(self, layer: Conv1D, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.weight = self.layer.weight
        self.bias = self.layer.bias
        self.debug = debug

    def __call__(self, xs):
        # https://discuss.huggingface.co/t/some-clarification-on-conv1d/51022
        x_components = xs[:-1, :]
        x_bias = xs[-1, :]

        y_components = F.linear(x_components, self.weight.T, bias=None)
        y_bias = F.linear(x_bias, self.weight.T, bias=self.bias).unsqueeze(0)
        ys = torch.cat([y_components, y_bias], dim=0)
        assert ys.shape[-1] == self.layer.nf

        if self.debug:
            # verify correctness
            x = torch.sum(xs, dim=0)
            y = self.layer(x)

            assert torch.allclose(y, ys.sum(0), atol=1e-5)

        return ys


class AttentionDecomposed:
    def __init__(self, layer: Attention, debug=False, generalized=True):
        self.layer = layer.to(torch.torch.double)
        self.debug = debug
        self.c_attn_decomposed = Conv1DDecomposed(
            self.layer.c_attn, debug=debug)
        self.c_proj_decomposed = Conv1DDecomposed(
            self.layer.c_proj, debug=debug)

        self.generalized = generalized

    def cd_attention(self, queries, keys, values, attention_mask=None):
        # requires num_components to be 2 (beta and gamma)
        num_components = queries.shape[0] - 1
        if num_components != 2:
            raise ValueError(
                "num_components must be 2, instead got: ", num_components)

        # raise NotImplementedError
        query_beta, query_gamma, query_bias = queries
        key_beta, key_gamma, key_bias = keys.transpose(-1, -2)
        value_beta, value_gamma, value_bias = values

        # allow only information from the beta/bias component to contribute to the beta attn scores
        # based on CD's i * g decomposition
        attn_beta = query_beta @ key_beta + query_beta @ key_bias + query_bias @ key_beta
        attn_gamma = query_gamma @ (key_beta + key_gamma +
                                    key_bias) + (query_beta + query_bias) @ key_gamma
        attn_bias = query_bias @ key_bias
        # TODO: need to add the attn_bias to the attn_beta and attn_gamma based on timestep (original beta masks)
        attn_beta += attn_bias

        if self.layer.scale:
            attn_beta = attn_beta / torch.full(
                [], values.size(-1) ** 0.5, dtype=attn_beta.dtype, device=device
            )
            attn_gamma = attn_gamma / torch.full(
                [], values.size(-1) ** 0.5, dtype=attn_beta.dtype, device=device
            )

        # create causal mask
        query_length, key_length = queries.size(-2), keys.size(-2)
        causal_mask = self.layer.bias[:, :, key_length -
                                      query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_beta.dtype).min
        mask_value = torch.full(
            [], mask_value, dtype=attn_beta.dtype, device=device)

        # it's okay to add the masks to all of them because it's a matrix of 0 and -inf
        def softmax_mask(attn_weights):
            # apply causal mask
            attn_weights = torch.where(
                causal_mask.to(bool), attn_weights.to(attn_weights.dtype), mask_value)
            # apply attention mask
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            return nn.Softmax(dim=-1)(attn_weights)

        # setting='all' because bias is not in the input
        attn_beta, attn_gamma = decomp_activation_all(
            torch.stack([attn_beta, attn_gamma], dim=0),
            activation=softmax_mask,
            debug=self.debug,
        )
        output_beta = attn_beta @ value_beta
        output_gamma = attn_gamma @ (value_beta +
                                     value_gamma) + attn_beta @ value_gamma
        output_bias = (attn_beta + attn_gamma) @ value_bias
        attn_output = torch.stack([output_beta, output_gamma, output_bias])
        return attn_output

    def gcd_attention(self, queries, keys, values, attention_mask=None):
        # calculate the attention weights as normal (using all contributions)
        query = queries.sum(0)
        key = keys.sum(0)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.layer.scale:
            attn_weights = attn_weights / torch.full(
                [], values.size(-1) ** 0.5, dtype=attn_weights.dtype, device=device
            )

        # create causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.layer.bias[:, :, key_length -
                                      query_length: key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full(
            [], mask_value, dtype=attn_weights.dtype, device=device)
        # apply causal mask
        attn_weights = torch.where(
            causal_mask.to(bool), attn_weights.to(attn_weights.dtype), mask_value)

        # apply the attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        # no dropout at eval time

        # apply attn_weight to all values independently
        attn_output = torch.matmul(attn_weights.unsqueeze(0), values)
        return attn_output

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 1, 3, 2, 4)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def __call__(self, hidden_states, past=None, attention_mask=None):

        # add one to split dim because first dim is num_contributions + 1
        queries, keys, values = self.c_attn_decomposed(
            hidden_states).split(self.layer.split_size, dim=3)

        queries = self._split_heads(
            queries, self.layer.num_heads, self.layer.head_dim)
        keys = self._split_heads(
            keys, self.layer.num_heads, self.layer.head_dim)
        values = self._split_heads(
            values, self.layer.num_heads, self.layer.head_dim)

        present = keys, values
        if past is not None:
            past_key, past_value = past
            keys = torch.cat((past_key, keys), dim=-2)
            values = torch.cat((past_value, values), dim=-2)

        if self.generalized:
            attn_output = self.gcd_attention(
                queries, keys, values, attention_mask=attention_mask)
        else:
            attn_output = self.cd_attention(
                queries, keys, values, attention_mask=attention_mask)
        attn_output = self._merge_heads(
            attn_output, self.layer.num_heads, self.layer.head_dim)
        attn_output = self.c_proj_decomposed(attn_output)

        if self.debug:
            # verify correctness
            attn_output_, _ = self.layer(
                hidden_states.sum(0), past=past, attention_mask=attention_mask)
            assert torch.allclose(attn_output_, attn_output.sum(0), atol=1e-5)

        return attn_output, present


class MLPDecomposed:
    def __init__(self, layer: MLP, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.debug = debug
        self.c_fc_decomposed = Conv1DDecomposed(self.layer.c_fc, debug=debug)
        self.c_proj_decomposed = Conv1DDecomposed(
            self.layer.c_proj, debug=debug)

    def __call__(self, xs, perms):
        h = self.c_fc_decomposed(xs)
        h = decomp_activation(h, gelu, debug=self.debug, perms=perms)
        h2 = self.c_proj_decomposed(h)

        if self.debug:
            # verify correctness
            x = xs.sum(0)
            y = self.layer(x)

            assert torch.allclose(y, h2.sum(0), atol=1e-5)
        return h2


class BlockDecomposed:
    def __init__(self, block: Block, debug=False, generalized=True):
        self.layer = block.to(torch.torch.double)
        self.debug = debug
        self.ln_1_decomposed = LayerNormDecomposed(
            self.layer.ln_1, debug=debug)
        self.attention_decomposed = AttentionDecomposed(
            self.layer.attn, debug=debug,
            generalized=generalized)
        self.ln_2_decomposed = LayerNormDecomposed(
            self.layer.ln_2, debug=debug)
        self.mlp_decomposed = MLPDecomposed(self.layer.mlp, debug=debug)

    def __call__(self, hidden_states, perms, past=None, attention_mask=None):
        if self.debug:
            hidden_states_ = hidden_states.sum(0)

        residuals = hidden_states

        hidden_states = self.ln_1_decomposed(hidden_states)
        attn_outputs, presents = self.attention_decomposed(
            hidden_states, past=past, attention_mask=attention_mask)
        hidden_states = attn_outputs + residuals

        residuals = hidden_states
        hidden_states = self.ln_2_decomposed(hidden_states)
        feed_forward_hidden_states = self.mlp_decomposed(hidden_states, perms)
        hidden_states = residuals + feed_forward_hidden_states

        if self.debug:
            # verify correctness
            hidden_states_, _ = self.layer(
                hidden_states_, past=past, attention_mask=attention_mask)
            assert torch.allclose(
                hidden_states_, hidden_states.sum(0), atol=1e-5), torch.mean(hidden_states_ - hidden_states.sum(0))

        return hidden_states, presents


class GPT2ModelDecomposed:
    def __init__(self, model: GPT2Model, debug=False, generalized=True):
        self.model = model.to(torch.torch.double)
        self.debug = debug
        self.h_decomposed = [BlockDecomposed(
            block, debug=debug, generalized=generalized) for block in self.model.h]
        self.ln_f_decomposed = LayerNormDecomposed(
            self.model.ln_f, debug=debug)

    def __call__(self, input_ids, beta_mask, num_contributions, perms, position_ids=None, token_type_ids=None, past=None, attention_mask=None):
        if past is None:
            past_length = 0
            past = tuple([None] * len(self.model.h))
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.model.wte(input_ids)
        position_embeds = self.model.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.model.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(
                dtype=hidden_states.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * \
                torch.finfo(hidden_states.dtype).min

        # segment hidden_states into contributions based on beta_mask
        beta_mask = beta_mask.permute(1, 0, 2).unsqueeze(-1)
        if num_contributions != beta_mask.shape[0]:
            raise RuntimeError(
                f"num_contributions {num_contributions} does not match beta_mask length at dim 0 ({beta_mask.shape[0]})")

        # make copies of full embeddings
        hidden_states_masked = torch.empty(
            num_contributions + 1, *hidden_states.shape).to(torch.torch.double).to(device)
        # zero out tokens outside of each component
        rem_dims = [1] * (len(hidden_states_masked.shape) - 1)
        hidden_states_masked[:-1] = hidden_states.repeat(
            num_contributions, *rem_dims) * beta_mask
        # initialized bias contribution values to 0
        hidden_states_masked[-1] = 0

        presents = []
        for i, (block_decomposed, layer_past) in enumerate(zip(self.h_decomposed, past)):
            hidden_states_masked, present = block_decomposed(
                hidden_states_masked, perms, past=layer_past, attention_mask=attention_mask)
            presents.append(present)

            if self.debug:
                # verify correctness
                hidden_states, _ = block_decomposed.layer(
                    hidden_states, past=layer_past, attention_mask=attention_mask)
                error = torch.mean(torch.abs(
                    hidden_states - hidden_states_masked.sum(0)
                )).item()
                print(f"GPT2 Block Layer {i} error: {error}")

        output_shape = input_shape + (hidden_states_masked.size(-1),)
        hidden_states_masked = self.ln_f_decomposed(hidden_states_masked)
        return hidden_states_masked.view(num_contributions + 1, *output_shape), presents


class GPT2ForSequenceClassificationDecomposed:
    def __init__(self, config=None, model=None, num_labels=2, debug=False, shapley_include_bias=False, num_contributions=2, generalized=True):
        if config is not None:
            self.model = GPT2ForSequenceClassification(
                config=config,
                num_labels=num_labels
            ).to(torch.torch.double)
        if model is not None:
            self.model = model.to(torch.torch.double)
        if config is None and model is None:
            raise ValueError("Need to provide either GPT2 config or model")

        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.debug = debug
        self.transformer_decomposed = GPT2ModelDecomposed(
            self.model.transformer, debug=debug, generalized=generalized)
        self.score_decomposed = LinearDecomposed(self.model.score, debug=debug)

        # whether or not to fix bias in permuations when computing shapley values
        if shapley_include_bias:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions + 1)))).to(device)
        else:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions)))).to(device)
        self.num_contributions = num_contributions

    def __call__(self, input_ids, beta_mask, token_type_ids=None, attention_mask=None, position_ids=None, labels=None):
        transformer_outputs = self.transformer_decomposed(
            input_ids=input_ids, beta_mask=beta_mask, perms=self.permutations, token_type_ids=token_type_ids,
            position_ids=position_ids, attention_mask=attention_mask, num_contributions=self.num_contributions
        )
        hidden_states, past_key_values = transformer_outputs
        logits = self.score_decomposed(hidden_states)

        batch_size, _ = input_ids.shape[:2]

        # LL: use the attention_mask to determine the sequence lengths
        sequence_lengths = torch.eq(
            attention_mask, 0).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)

        # extra dim because first dim is # contributions + 1
        pooled_logits = logits[:, torch.arange(
            batch_size, device=device), sequence_lengths]

        if self.debug:
            # verify correctness
            output_ = self.model(
                input_ids, token_type_ids=token_type_ids, position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None)
            logits_error = torch.mean(torch.abs(
                output_["logits"] - pooled_logits.sum(0)
            )).item()
            print("GPT2 Classifier logits error: ", logits_error)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                pooled_logits.view(-1, self.num_labels), labels.view(-1))
            return loss

        return {
            "logits": pooled_logits,
            "past_key_values": past_key_values
        }

    @classmethod
    def from_pretrained(cls, pretrained_model, num_labels=2, num_contributions=2, shapley_include_bias=False, debug=False, generalized=True):
        """
        load from pretrained model name or file path from huggingface
        """
        model_hf = transformers.GPT2ForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
        )

        new_state_dict = OrderedDict()
        for key, value in model_hf.state_dict().items():
            # ref : https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            new_key = key.replace("ln_1.weight", "ln_1.gamma").replace(
                "ln_1.bias", "ln_1.beta")
            new_key = new_key.replace("ln_2.weight", "ln_2.gamma").replace(
                "ln_2.bias", "ln_2.beta")
            new_key = new_key.replace("ln_f.weight", "ln_f.gamma").replace(
                "ln_f.bias", "ln_f.beta")
            new_state_dict[new_key] = value

            new_state_dict[new_key] = value

        decomposed_model = GPT2ForSequenceClassificationDecomposed(
            config=GPT2Config.from_dict(model_hf.config.to_dict()),
            num_labels=num_labels,
            debug=debug,
            num_contributions=num_contributions,
            shapley_include_bias=shapley_include_bias,
            generalized=generalized)
        decomposed_model.model.load_state_dict(new_state_dict)
        decomposed_model.model.eval()
        return decomposed_model


class GPT2LMHeadModelDecomposed:
    def __init__(self, config=None, model=None, debug=False, shapley_include_bias=False, num_contributions=2, generalized=True):
        if config is not None:
            self.model = GPT2LMHeadModel(
                config=config,
            ).to(torch.torch.double)
        if model is not None:
            self.model = model.to(torch.torch.double)
        if config is None and model is None:
            raise ValueError("Need to provide either GPT2 config or model")

        self.model.config.pad_token_id = self.model.config.eos_token_id

        self.debug = debug
        self.transformer_decomposed = GPT2ModelDecomposed(
            self.model.transformer, debug=debug, generalized=generalized)
        self.lm_head_decomposed = LinearDecomposed(
            self.model.lm_head, debug=debug)

        # whether or not to fix bias in permuations when computing shapley values
        if shapley_include_bias:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions + 1)))).to(device)
        else:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions)))).to(device)
        self.num_contributions = num_contributions

    def __call__(self, input_ids, beta_mask, token_type_ids=None, attention_mask=None, position_ids=None, labels=None, past=None):
        transformer_outputs = self.transformer_decomposed(
            input_ids=input_ids, beta_mask=beta_mask, perms=self.permutations, token_type_ids=token_type_ids,
            position_ids=position_ids, attention_mask=attention_mask, num_contributions=self.num_contributions,
            past=past
        )
        hidden_states, past_key_values = transformer_outputs
        lm_logits = self.lm_head_decomposed(hidden_states)

        if labels is not None:
            lm_logits_ = lm_logits.sum(0)
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                lm_logits_.view(-1, lm_logits_.size(-1)), labels.view(-1))
            return loss

        return {
            "logits": lm_logits,
            "past_key_values": past_key_values
        }

    @classmethod
    def from_pretrained(cls, pretrained_model, debug=False, num_contributions=2, shapley_include_bias=False, generalized=True):
        """
        load from pretrained model name or file path from huggingface
        """
        model_hf = transformers.GPT2LMHeadModel.from_pretrained(
            pretrained_model,
        )

        new_state_dict = OrderedDict()
        for key, value in model_hf.state_dict().items():
            # ref : https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            new_key = key.replace("ln_1.weight", "ln_1.gamma").replace(
                "ln_1.bias", "ln_1.beta")
            new_key = new_key.replace("ln_2.weight", "ln_2.gamma").replace(
                "ln_2.bias", "ln_2.beta")
            new_key = new_key.replace("ln_f.weight", "ln_f.gamma").replace(
                "ln_f.bias", "ln_f.beta")
            new_state_dict[new_key] = value

        decomposed_model = GPT2LMHeadModelDecomposed(
            config=GPT2Config.from_dict(model_hf.config.to_dict()),
            debug=debug,
            num_contributions=num_contributions,
            shapley_include_bias=shapley_include_bias,
            generalized=generalized)
        decomposed_model.model.load_state_dict(new_state_dict)
        decomposed_model.model.eval()
        return decomposed_model
