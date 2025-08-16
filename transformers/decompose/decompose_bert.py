from __future__ import division
from bert_model import *
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
import torch
import logging
import math
import itertools
import transformers
from collections import OrderedDict
from decompose_util import LinearDecomposed, decomp_activation, decomp_activation_all
from decompose_util import LayerNormDecomposed as BertLayerNormDecomposed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class BertSelfAttentionDecomposed:
    def __init__(self, layer: BertSelfAttention, debug=False, generalized=True):
        self.layer = layer.to(torch.torch.double)
        self.value_decomposed = LinearDecomposed(self.layer.value, debug=debug)
        self.debug = debug

        if not generalized:
            self.key_decomposed = LinearDecomposed(self.layer.key, debug=debug)
            self.query_decomposed = LinearDecomposed(
                self.layer.query, debug=debug)
        self.generalized = generalized

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.layer.num_attention_heads, self.layer.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def __call__(self, xs, attention_mask):
        if self.generalized:
            return self.gcd_attention(xs, attention_mask)
        else:
            return self.cd_attention(xs, attention_mask)

    def cd_attention(self, xs, attention_mask):
        # requires num_components to be 2 (beta and gamma)
        # dim 0 is (beta, gamma, bias)
        num_components = xs.shape[0] - 1
        if num_components != 2:
            raise ValueError(
                "num_components must be 2, instead got: ", num_components)

        # decompose all of {Q, K, V}
        mixed_query_layers = self.query_decomposed(xs)
        query_layers = self.transpose_for_scores(mixed_query_layers)
        query_beta, query_gamma, query_bias = query_layers

        mixed_key_layers = self.key_decomposed(xs)
        key_layers = self.transpose_for_scores(mixed_key_layers)
        key_beta, key_gamma, key_bias = key_layers.transpose(-1, -2)

        mixed_value_layers = self.value_decomposed(xs)
        value_layers = self.transpose_for_scores(mixed_value_layers)
        value_beta, value_gamma, value_bias = value_layers

        # allow only information from the beta/bias component to contribute to the beta attn scores
        # based on CD's i * g decomposition
        attn_beta = query_beta @ key_beta + query_beta @ key_bias + query_bias @ key_beta
        attn_gamma = query_gamma @ (key_beta + key_gamma +
                                    key_bias) + (query_beta + query_bias) @ key_gamma
        attn_bias = query_bias @ key_bias
        attn_beta += attn_bias

        attn_beta = attn_beta / \
            math.sqrt(self.layer.attention_head_size)
        attn_gamma = attn_gamma / \
            math.sqrt(self.layer.attention_head_size)

        # it's okay to add the attention mask to all of them because it's a matrix of 0 and -inf
        def softmax_mask(x):
            return nn.Softmax(dim=-1)(x + attention_mask)

        # setting='all' because bias is not in the input
        attn_beta, attn_gamma = decomp_activation_all(
            torch.stack([attn_beta, attn_gamma], dim=0),
            activation=softmax_mask,
            debug=self.debug,
        )
        # output_beta = attn_beta @ value_beta + \
        #     attn_bias @ value_beta + attn_beta @ value_bias
        # output_gamma = attn_gamma @ (value_beta + value_gamma +
        #                              value_bias) + (attn_beta + attn_bias) @ value_gamma
        # output_bias = attn_bias @ value_bias

        # based on CD's f * c decomposition
        output_beta = attn_beta @ value_beta
        output_gamma = attn_gamma @ (value_beta +
                                     value_gamma) + attn_beta @ value_gamma
        output_bias = (attn_beta + attn_gamma) @ value_bias

        context_layers = torch.stack([output_beta, output_gamma, output_bias]).permute(
            0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layers.size(
        )[:-2] + (self.layer.all_head_size,)
        context_layers = context_layers.view(*new_context_layer_shape)

        if self.debug:
            # verify correctness
            hidden_states = xs.sum(0)
            context_layer = self.layer(hidden_states, attention_mask)
            assert torch.allclose(
                torch.sum(context_layers, dim=0),
                context_layer,  atol=1e-5
            ), torch.mean(torch.sum(context_layers, axis=0) - context_layer)

        return context_layers

    def gcd_attention(self, xs, attention_mask):
        hidden_states = torch.sum(xs, axis=0)

        # attention weights are preserved, meaning query/key are the same
        # however, split values into beta/gamma/bias

        mixed_query_layer = self.layer.query(hidden_states)
        query_layer = self.layer.transpose_for_scores(mixed_query_layer)

        mixed_key_layer = self.layer.key(hidden_states)
        key_layer = self.layer.transpose_for_scores(mixed_key_layer)

        mixed_value_layers = self.value_decomposed(xs)
        value_layers = self.transpose_for_scores(mixed_value_layers)

        # same attention scores as normal
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)) \
            / math.sqrt(self.layer.attention_head_size) + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layers = torch.matmul(
            attention_probs.unsqueeze(0),
            value_layers).permute(0, 1, 3, 2, 4).contiguous()
        new_context_layer_shape = context_layers.size(
        )[:-2] + (self.layer.all_head_size,)
        context_layers = context_layers.view(*new_context_layer_shape)

        if self.debug:
            # verify correctness
            context_layer = self.layer(hidden_states, attention_mask)
            assert torch.allclose(
                torch.sum(context_layers, dim=0),
                context_layer,  atol=1e-5
            )

        return context_layers


class BertSelfOutputDecomposed:
    def __init__(self, layer: BertSelfOutput, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.dense_decomposed = LinearDecomposed(self.layer.dense, debug=debug)
        self.LayerNorm_decomposed = BertLayerNormDecomposed(
            self.layer.LayerNorm, debug=debug)
        self.debug = debug

    def __call__(self, xs_hidden, xs_input):
        if self.debug:
            hidden_states = torch.sum(xs_hidden, dim=0)
            input_tensor = torch.sum(xs_input, dim=0)

        xs_hidden = self.dense_decomposed(xs_hidden)

        xs_hidden = self.LayerNorm_decomposed(xs_hidden + xs_input)

        if self.debug:
            # verify correctness
            hidden_states = self.layer(hidden_states, input_tensor)
            assert torch.allclose(
                hidden_states,
                torch.sum(xs_hidden, dim=0),
                atol=1e-5
            )

        return xs_hidden


class BertAttentionDecomposed:
    def __init__(self, layer: BertAttention, debug=False, generalized=True):
        self.layer = layer.to(torch.torch.double)
        self.self_decomposed = BertSelfAttentionDecomposed(
            self.layer.self, debug, generalized=generalized)
        self.output_decomposed = BertSelfOutputDecomposed(
            self.layer.output, debug)
        self.debug = debug

    def __call__(self, xs, attention_mask):
        self_outputs = self.self_decomposed(
            xs, attention_mask)

        attention_outputs = self.output_decomposed(
            self_outputs, xs
        )

        if self.debug:
            input_tensor = torch.sum(xs, axis=0)

            # verify correctness
            attention_output = self.layer(input_tensor, attention_mask)
            assert torch.allclose(
                attention_output,
                torch.sum(attention_outputs, dim=0),
                atol=1e-5
            )

        return attention_outputs


class BertIntermediateDecomposed:
    def __init__(self, layer: BertIntermediate, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.dense_decomposed = LinearDecomposed(self.layer.dense, debug=debug)
        self.debug = debug

    def __call__(self, xs, perms):
        if self.debug:
            hidden_states = torch.sum(xs, axis=0)

        xs = self.dense_decomposed(xs)
        xs = decomp_activation(
            xs,
            self.layer.intermediate_act_fn, perms=perms, debug=self.debug
        )

        if self.debug:
            # verify correctness
            hidden_states = self.layer(hidden_states)

            assert torch.allclose(
                hidden_states,
                torch.sum(xs, dim=0),
                atol=1e-5
            )

        return xs


class BertOutputDecomposed:
    def __init__(self, layer: BertOutput, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.dense_decomposed = LinearDecomposed(self.layer.dense, debug=debug)
        self.LayerNorm_decomposed = BertLayerNormDecomposed(
            self.layer.LayerNorm, debug=debug)
        self.debug = debug

    def __call__(self, xs_hidden, xs_input):

        if self.debug:
            hidden_states = torch.sum(xs_hidden, 0)
            input_tensor = torch.sum(xs_input, 0)

        xs_hidden = self.dense_decomposed(
            xs_hidden
        )
        xs_hidden = self.LayerNorm_decomposed(
            xs_hidden + xs_input
        )

        if self.debug:
            # verify correctness
            hidden_states = self.layer(hidden_states, input_tensor)

            assert torch.allclose(
                hidden_states,
                torch.sum(xs_hidden, axis=0),
                atol=1e-5
            )

        return xs_hidden


class BertLayerDecomposed:
    def __init__(self, layer: BertLayer, debug=False, generalized=True):
        self.layer = layer.to(torch.torch.double)
        self.attention_decomposed = BertAttentionDecomposed(
            self.layer.attention, debug=debug, generalized=generalized)
        self.intermediate_decomposed = BertIntermediateDecomposed(
            self.layer.intermediate, debug=debug)
        self.output_decomposed = BertOutputDecomposed(
            self.layer.output, debug=debug)
        self.debug = debug

    def __call__(self, xs, attention_mask, perms):

        hidden_states = torch.sum(xs, axis=0)

        attention_outputs = self.attention_decomposed(
            xs, attention_mask)
        intermediate_outputs = self.intermediate_decomposed(
            attention_outputs, perms=perms)
        layer_outputs = self.output_decomposed(
            intermediate_outputs, attention_outputs)

        if self.debug:
            # verify correctness
            layer_outputs_ = self.layer(hidden_states, attention_mask)
            assert torch.allclose(
                layer_outputs_,
                torch.sum(layer_outputs, dim=0),
                atol=1e-5
            )

        return layer_outputs


class BertEncoderDecomposed:
    def __init__(self, model: BertEncoder, debug=False, generalized=True):
        self.model = model.to(torch.torch.double)
        self.layer_decomposed = [BertLayerDecomposed(
            layer, debug=debug, generalized=generalized) for layer in self.model.layer]
        self.debug = debug

    def __call__(self, xs, attention_mask, perms):
        hidden_states = torch.sum(xs, axis=0)

        for i, layer_module in enumerate(self.layer_decomposed):
            xs = layer_module(xs, attention_mask, perms=perms)

            if self.debug:
                # verify correctness
                hidden_states = layer_module.layer(
                    hidden_states, attention_mask)
                error = torch.mean(torch.abs(
                    hidden_states - torch.sum(xs, dim=0)
                )).item()
                print(f"Bert Encoder Layer {i} error: {error}")

        return xs


class BertPoolerDecomposed:
    def __init__(self, layer: BertPooler, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.dense_decomposed = LinearDecomposed(self.layer.dense, debug=debug)
        self.debug = debug

    def __call__(self, xs, perms):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensors = xs[:, :, 0]

        pooled_outputs = self.dense_decomposed(first_token_tensors)

        pooled_outputs = decomp_activation(
            pooled_outputs, self.layer.activation.forward, debug=self.debug,
            perms=perms
        )

        if self.debug:
            # verify correctness
            pooled_output = self.layer(torch.sum(xs, axis=0))
            assert torch.allclose(
                pooled_output,
                torch.sum(pooled_outputs, dim=0),
                atol=1e-5
            )
        return pooled_outputs


class BertEmbeddingsDecomposed:
    def __init__(self, layer: BertEmbeddings, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.LayerNorm_decomposed = BertLayerNormDecomposed(
            self.layer.LayerNorm, debug=debug)
        self.debug = debug

    def __call__(self, input_ids, beta_mask, token_type_ids=None, num_contributions=None):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.layer.word_embeddings(input_ids)
        position_embeddings = self.layer.position_embeddings(position_ids)
        token_type_embeddings = self.layer.token_type_embeddings(
            token_type_ids)

        embeddings = words_embeddings + \
            position_embeddings + token_type_embeddings

        # embeddings shape = [batch_size, seq_len, hidden_size]
        # beta_mask shape = [batch_size, num_contributions, seq_len]
        # -> reshape to [num_contributions, batch_size, seq_len, 1]
        beta_mask = beta_mask.permute(1, 0, 2).unsqueeze(-1)
        if num_contributions != beta_mask.shape[0]:
            raise RuntimeError(
                f"num_contributions ({num_contributions}) does not match beta_mask length at dim 0 ({beta_mask.shape[0]})")

        # make copies of full embeddings
        embeddings_masked = torch.empty(
            num_contributions + 1, *embeddings.shape
        ).to(torch.torch.double).to(device)
        # zero out tokens outside of each component
        embeddings_masked[:-1, :] = embeddings.repeat(
            num_contributions, 1, 1, 1) * beta_mask
        # initialize bias contribution values to 0
        embeddings_masked[-1, :] = 0

        embeddings_masked = self.LayerNorm_decomposed(
            embeddings_masked
        )

        if self.debug:
            # verify correctness
            embeddings_ = self.layer(
                input_ids, token_type_ids=None)
            assert torch.allclose(
                torch.sum(embeddings_masked, axis=0),
                embeddings_, atol=1e-5
            )

        return embeddings_masked


class BertModelDecomposed:
    def __init__(self, model: BertModel, debug=False, generalized=True):
        self.model = model.to(torch.torch.double)
        self.embeddings_decomposed = BertEmbeddingsDecomposed(
            self.model.embeddings, debug=debug)
        self.encoder_decomposed = BertEncoderDecomposed(
            self.model.encoder, debug=debug, generalized=generalized)
        self.pooler_decomposed = BertPoolerDecomposed(
            self.model.pooler, debug=debug)

    def __call__(self, input_ids, beta_mask, perms, token_type_ids=None,
                 attention_mask=None, num_contributions=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # keep attention mask the same as BertModel
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.model.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layers
        embedding_outputs = self.embeddings_decomposed(
            input_ids, beta_mask, token_type_ids=token_type_ids,
            num_contributions=num_contributions)

        # encoding layers
        encoded_layers = self.encoder_decomposed(
            embedding_outputs, extended_attention_mask, perms=perms)

        pooled_outputs = self.pooler_decomposed(encoded_layers, perms)

        return (encoded_layers, pooled_outputs)


class BertForSequenceClassificationDecomposed:
    def __init__(self, config=None, model=None, num_labels=2, debug=False, shapley_include_bias=False, num_contributions=2, generalized=True):
        if config is not None:
            self.model = BertForSequenceClassification(
                config=config,
                num_labels=num_labels
            ).to(torch.torch.double)
        if model is not None:
            self.model = model.to(torch.torch.double)
        if config is None and model is None:
            raise ValueError("Need to provide either Bert config or model")
        self.bert_decomposed = BertModelDecomposed(
            self.model.bert, debug=debug, generalized=generalized)
        self.classifier_decomposed = LinearDecomposed(
            self.model.classifier, debug=debug)
        self.debug = debug

        # whether or not to fix bias in permuations when computing shapley values
        if shapley_include_bias:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions + 1)))).to(device)
        else:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions)))).to(device)
        self.num_contributions = num_contributions

    def __call__(self, input_ids, beta_mask, token_type_ids=None,
                 attention_mask=None, labels=None):

        _, pooled_outputs = self.bert_decomposed(
            input_ids=input_ids, beta_mask=beta_mask, token_type_ids=token_type_ids,
            attention_mask=attention_mask, perms=self.permutations,
            num_contributions=self.num_contributions)

        logits = self.classifier_decomposed(pooled_outputs)

        if self.debug:
            # verify correctness
            _, pooled_outputs_ = self.model.bert(
                input_ids=input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask)

            pooled_outputs_error = torch.mean(torch.abs(
                pooled_outputs_ - sum(pooled_outputs)
            )).item()
            print("\nPooled output error: ", pooled_outputs_error)

            logits_ = self.model(input_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, labels=None)
            logits_error = torch.mean(torch.abs(
                logits_ - torch.sum(logits, dim=0)
            )).item()
            print("\nBert Classifier logits error: ", logits_error)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.model.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    @classmethod
    def from_pretrained(cls, pretrained_model, num_labels=2, num_contributions=2, shapley_include_bias=False, debug=False, generalized=True):
        """
        load from pretrained model name or file path from huggingface
        """
        model_hf = transformers.BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
        )

        new_state_dict = OrderedDict()
        for key, value in model_hf.state_dict().items():
            new_key = key.replace(
                "classifier.dense", "bert.pooler.dense").replace(
                    "classifier.out_proj", "classifier")
            # ref : https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            new_key = new_key.replace("LayerNorm.weight",
                                      "LayerNorm.gamma").replace("LayerNorm.bias",
                                                                 "LayerNorm.beta")
            new_key = new_key.replace("lm_head.layer_norm.weight",
                                      "cls.predictions.transform.LayerNorm.gamma").replace("lm_head.layer_norm.bias",
                                                                                           "cls.predictions.transform.LayerNorm.beta")
            new_key = new_key.replace(
                "lm_head.decoder", "cls.predictions.decoder")
            new_key = new_key.replace(
                "lm_head.dense", "cls.predictions.transform.dense")
            new_key = new_key.replace(
                "cls.predictions.decoder.bias", "cls.predictions.transform.dense")
            new_key = new_key.replace("lm_head.bias", "cls.predictions.bias")

            new_state_dict[new_key] = value

        decomposed_model = BertForSequenceClassificationDecomposed(
            config=BertConfig.from_dict(model_hf.config.to_dict()),
            num_labels=num_labels,
            debug=debug,
            num_contributions=num_contributions,
            shapley_include_bias=shapley_include_bias,
            generalized=generalized)
        decomposed_model.model.load_state_dict(new_state_dict)
        decomposed_model.model.eval()
        return decomposed_model


class BertPredictionHeadTransformDecomposed:
    def __init__(self, layer: BertPredictionHeadTransform, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.dense_decomposed = LinearDecomposed(layer.dense, debug=debug)
        self.transform_act_fn = layer.transform_act_fn
        self.LayerNorm_decomposed = BertLayerNormDecomposed(
            layer.LayerNorm, debug=debug)
        self.debug = debug

    def __call__(self, xs, perms):
        if self.debug:
            x = torch.sum(xs, dim=0)

        xs = self.dense_decomposed(xs)
        xs = decomp_activation(xs, self.transform_act_fn,
                               self.debug, perms=perms)
        xs = self.LayerNorm_decomposed(xs)

        if self.debug:
            # verify correctness
            y = self.layer(x)
            assert torch.allclose(y, torch.sum(xs, dim=0), atol=1e-5)

        return xs


class BertLMPredictionHeadDecomposed:
    def __init__(self, layer: BertLMPredictionHead, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.transform_decomposed = BertPredictionHeadTransformDecomposed(
            layer.transform, debug)
        self.decoder_decomposed = LinearDecomposed(self.layer.decoder, debug)
        self.debug = debug

    def __call__(self, xs, perms):
        if self.debug:
            x = torch.sum(xs, dim=0)

        xs = self.transform_decomposed(xs, perms)
        xs = self.decoder_decomposed(xs)

        # add layer's bias to xs' bias component
        xs[-1, :] = xs[-1, :] + self.layer.bias

        if self.debug:
            # verify correctness
            y = self.layer(x)
            assert torch.allclose(y, torch.sum(xs, dim=0), atol=1e-5)

        return xs


class BertOnlyMLMHeadDecomposed:
    def __init__(self, layer: BertOnlyMLMHead, debug=False):
        self.layer = layer.to(torch.torch.double)
        self.predictions_decomposed = BertLMPredictionHeadDecomposed(
            self.layer.predictions, debug=debug)
        self.debug = debug

    def __call__(self, sequence_output, perms):
        prediction_scores = self.predictions_decomposed(sequence_output, perms)
        return prediction_scores


class BertForMaskedLMDecomposed:
    def __init__(self, config, debug=False, shapley_include_bias=False, num_contributions=2, generalized=True):
        self.model = BertForMaskedLM(
            config).to(torch.torch.double)
        self.bert_decomposed = BertModelDecomposed(
            self.model.bert, generalized=generalized)
        self.cls_decomposed = BertOnlyMLMHeadDecomposed(self.model.cls)
        self.debug = debug

        # whether or not to fix bias in permuations when computing shapley values
        if shapley_include_bias:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions + 1)))).to(device)
        else:
            self.permutations = torch.tensor(
                list(itertools.permutations(torch.arange(num_contributions)))).to(device)
        self.num_contributions = num_contributions

    def __call__(self, input_ids, beta_mask, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert_decomposed(
            input_ids=input_ids, beta_mask=beta_mask, token_type_ids=token_type_ids,
            attention_mask=attention_mask, perms=self.permutations,
            num_contributions=self.num_contributions)

        prediction_scores = self.cls_decomposed(
            sequence_output, perms=self.permutations)

        if self.debug:
            # verify correctness
            sequence_output_, _ = self.model.bert(
                input_ids=input_ids, token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_all_encoded_layers=False)

            prediction_scores_ = self.model.cls(sequence_output_)
            predictions_error = torch.mean(torch.abs(
                prediction_scores_ - torch.sum(prediction_scores, dim=0)
            )).item()
            print("\nBert LM predictions error: ", predictions_error)

        return prediction_scores

    @classmethod
    def from_pretrained(cls, pretrained_model, debug=False, shapley_include_bias=False, num_contributions=2, generalized=True, **kwargs):
        """
        load from pretrained model name or file path from huggingface
        """
        model_hf = transformers.BertForMaskedLM.from_pretrained(
            pretrained_model, **kwargs
        )

        new_state_dict = OrderedDict()
        for key, value in model_hf.state_dict().items():
            new_key = key.replace(
                "classifier.dense", "bert.pooler.dense").replace(
                    "classifier.out_proj", "classifier")
            # ref : https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
            new_key = new_key.replace("LayerNorm.weight",
                                      "LayerNorm.gamma").replace("LayerNorm.bias",
                                                                 "LayerNorm.beta")
            new_key = new_key.replace("lm_head.layer_norm.weight",
                                      "cls.predictions.transform.LayerNorm.gamma"
                                      ).replace("lm_head.layer_norm.bias",
                                                "cls.predictions.transform.LayerNorm.beta")

            new_state_dict[new_key] = value

        if "cls.predictions.decoder.bias" in new_state_dict:
            new_state_dict.pop("cls.predictions.decoder.bias")

        # okay to initialize as empty, since these parameters are not used in LM
        hd_size = model_hf.config.hidden_size
        new_state_dict['bert.pooler.dense.weight'] = torch.empty(
            [hd_size, hd_size])
        new_state_dict['bert.pooler.dense.bias'] = torch.empty([hd_size])

        decomposed_model = BertForMaskedLMDecomposed(
            config=BertConfig.from_dict(model_hf.config.to_dict()),
            debug=debug,
            shapley_include_bias=shapley_include_bias,
            num_contributions=num_contributions,
            generalized=generalized)
        decomposed_model.model.load_state_dict(new_state_dict)
        decomposed_model.model.eval()
        return decomposed_model
