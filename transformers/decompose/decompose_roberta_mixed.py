from roberta_model import *
from decompose_roberta import *


class RobertaForSequenceClassificationMixed:
    def __init__(self, config, state_dict, segment_layer, num_labels=2, debug=False):
        self.decomposed_model = RobertaForSequenceClassificationDecomposed(
            config, num_labels=num_labels, debug=debug)
        self.decomposed_model.model.load_state_dict(state_dict)
        self.decomposed_model.model.eval()

        self.segment_layer = segment_layer
        self.bert = self.decomposed_model.model.bert
        self.decomposed_layers = self.decomposed_model.bert_decomposed.encoder_decomposed.layer_decomposed
        self.pooler = self.decomposed_model.bert_decomposed.pooler_decomposed
        self.clf = self.decomposed_model.classifier_decomposed

        if self.segment_layer > len(self.decomposed_layers) or self.segment_layer < 0:
            raise ValueError(
                f"segment_layer must be within the number of encoder layers ({len(self.decomposed_layers)})")

    def __call__(self, input_ids, beta_mask, num_contributions,
                 attention_mask=None):
        # if splitting components at the input layer, just return decomposed_model's output
        if self.segment_layer == 0:
            return {
                "logits": self.decomposed_model(
                    input_ids=input_ids,
                    num_contributions=num_contributions,
                    attention_mask=attention_mask,
                    beta_mask=beta_mask
                )}

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # get embeddings right before the layer to segment components
        # RobertaModel creates its own extended attention mask
        pre_segment_embeddings = self.bert(
            input_ids, attention_mask=attention_mask,
            output_all_encoded_layers=True)[0][self.segment_layer - 1]

        # embeddings shape = [batch_size, seq_len, hidden_size]
        # beta_mask shape = [batch_size, num_contributions, seq_len]
        # -> reshape to [num_contributions, batch_size, seq_len, 1]
        beta_mask = beta_mask.permute(1, 0, 2).unsqueeze(-1)

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

        # keep attention mask the same as RobertaModel
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.bert.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # run embeddings through decomposed transformer layers
        for layer_module in self.decomposed_layers[self.segment_layer:]:
            xs = layer_module(xs, extended_attention_mask, perms=perms)

        # decomposed classifier layers
        cls_embedding = self.pooler(xs, perms=perms)
        output = self.clf(cls_embedding)

        return {
            "pre_segment_embeddings": pre_segment_embeddings,
            "final_hidden_embeddings": xs,
            "logits": output}
