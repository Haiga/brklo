import torch
from pytorch_lightning import LightningModule


class NoPooling(LightningModule):


    def __init__(self):
        super(NoPooling, self).__init__()

    def forward(self, encoder_outputs, attention_mask=None):
        return encoder_outputs.pooler_output

