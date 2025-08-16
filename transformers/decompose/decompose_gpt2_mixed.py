from gpt2_model import *
from decompose_gpt2 import *
import numpy as np


class GPT2ForSequenceClassificationMixed:
    def __init__(self, config, state_dict, segment_layer, num_labels=2, debug=False):
        self.decomposed_model = GPT2ForSequenceClassificationDecomposed(
            config, num_labels=num_labels, debug=debug)
        self.decomposed_model.model.load_state_dict(state_dict)
        self.decomposed_model.model.eval()

        self.segment_layer = segment_layer
        self.encoder = self.decomposed_model.model.transformer
        self.decomposed_layers = self.decomposed_model.transformer_decomposed.h_decomposed
        self.ln_f = self.decomposed_model.transformer_decomposed.ln_f_decomposed
        self.clf = self.decomposed_model.score_decomposed

        if self.segment_layer > len(self.decomposed_layers) or self.segment_layer < 0:
            raise ValueError(
                f"segment_layer must be within the number of encoder layers ({len(self.decomposed_layers)})")

    def __call__(self, input_ids, beta_mask, num_contributions,
                 attention_mask=None):
        # if splitting components at the input layer, just return decomposed_model's output
        if self.segment_layer == 0:
            return self.decomposed_model(
                input_ids=input_ids,
                num_contributions=num_contributions,
                attention_mask=attention_mask,
                beta_mask=beta_mask
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # get embeddings right before the layer to segment components
        # RobertaModel creates its own extended attention mask
        pre_segment_embeddings = self.encoder(
            input_ids, attention_mask=attention_mask,
            output_all_encoded_layers=True)[0][self.segment_layer - 1]

        # embeddings shape = [batch_size, seq_len, hidden_size]
        # beta_mask shape = [batch_size, num_contributions, seq_len]
        # -> reshape to [num_contributions, batch_size, seq_len, 1]
        beta_mask = beta_mask.permute(1, 0, 2).unsqueeze(-1)
        if num_contributions != beta_mask.shape[0]:
            raise RuntimeError(
                f"num_contributions {num_contributions} does not match beta_mask length at dim 0 ({beta_mask.shape[0]})")

        # add bias values (zero) to beta_mask
        beta_mask = np.concatenate(
            [beta_mask, torch.zeros_like(beta_mask[0]).unsqueeze(0)], axis=0)

        # conduct the split
        new_split = torch.stack([pre_segment_embeddings]
                                * (num_contributions + 1))

        if beta_mask.shape[:-1] != new_split.shape[:-1]:
            raise ValueError(
                f"beta_mask shape ({beta_mask.shape}) must be broadcastable to embeddings shape ({new_split.shape})")

        xs = new_split * beta_mask
        perms = torch.tensor(
            list(itertools.permutations(torch.arange(num_contributions))))

        # keep attention mask the same as GPT2Model
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * \
                torch.finfo(extended_attention_mask.dtype).min

        # run embeddings through decomposed transformer layers
        input_shape = input_ids.size()
        for layer_module in self.decomposed_layers[self.segment_layer:]:
            xs = layer_module(
                xs, attention_mask=extended_attention_mask, perms=perms)[0]
        output_shape = input_shape + (xs.size(-1),)
        hidden_states_masked = self.ln_f(xs)
        cls_embedding = hidden_states_masked.view(
            num_contributions + 1, *output_shape)

        # decomposed classifier layers
        logits = self.clf(cls_embedding)

        # use the attention_mask to determine the sequence lengths
        sequence_lengths = torch.eq(
            attention_mask, 0).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
        batch_size, _ = input_ids.shape[:2]

        pooled_logits = logits[:, torch.arange(
            batch_size, device=device), sequence_lengths]

        return {
            "pre_segment_embeddings": pre_segment_embeddings,
            "final_hidden_embeddings": xs,
            "logits": pooled_logits}
