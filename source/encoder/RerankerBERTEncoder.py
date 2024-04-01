from pytorch_lightning import LightningModule
from transformers import BertModel


class RerankerBERTEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, output_hidden_states, pooling):
        super(RerankerBERTEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states

        )
        self.pooling = pooling

    def forward(self, feature):
        encoder_outputs = self.encoder(
            input_ids=feature["input_ids"],
            attention_mask=feature["attention_mask"],
            token_type_ids=feature["token_type_ids"]
        )

        return self.pooling(
            encoder_outputs,
            feature["attention_mask"]
        )
