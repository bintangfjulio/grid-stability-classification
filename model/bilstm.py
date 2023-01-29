import torch
import torch.nn as nn
import pytorch_lightning as pl

class BiLSTM(pl.LightningModule):
    def __init__(self, num_classes, input_size, hidden_size=300, num_layers=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        lstm_output, _ = self.lstm(X)
        preds = self.output_layer(self.dropout(lstm_output))

        return preds
