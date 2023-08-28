import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy

class BiLSTM(pl.LightningModule):
    def __init__(self, lr, num_classes, input_size, hidden_size=300, num_layers=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size * 2, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy_metric = BinaryAccuracy()
        self.lr = lr
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        lstm_output, _ = self.lstm(X)
        preds = self.output_layer(self.dropout(lstm_output))

        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X, target = train_batch

        preds = self(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, target = valid_batch

        preds = self(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        X, target = test_batch

        preds = self(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss
