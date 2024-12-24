import torch.nn as nn
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LVModel(nn.Module):
    def __init__(self, base_model):
        super(LVModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=0.3)
        self.layer_norm = nn.LayerNorm(self.base_model.config.d_model)
        self.classifier = nn.Linear(self.base_model.config.d_model, 3)

        # Ensure weight sharing
        self.base_model.shared = self.base_model.encoder.embed_tokens
        self.base_model.decoder.embed_tokens = self.base_model.encoder.embed_tokens

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.encoder_last_hidden_state[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.classifier(hidden_states)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    