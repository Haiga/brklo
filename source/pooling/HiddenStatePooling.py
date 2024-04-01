import torch
from pytorch_lightning import LightningModule


class HiddenStatePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self, query_max_length, passage_max_length):
        super(HiddenStatePooling, self).__init__()
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

    def forward(self, encoder_outputs, attention_mask):

        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        # put 0 over PADs
        a = hidden_states * attention_mask
        return torch.nn.functional.normalize(a, p=2, dim=2)

        #return torch.split(a, (self.text_max_length, self.label_max_length), dim=1)



