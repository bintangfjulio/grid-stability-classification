import torch
import torch.nn as nn
import pytorch_lightning as pl

from model.bilstm import BiLSTM
from torchmetrics.classification import BinaryAccuracy

class Classifier(pl.LightningModule):
    def __init__(self, lr, num_classes, input_size):
        super(Classifier, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy_metric = BinaryAccuracy()
        self.lr = lr
        self.model = BiLSTM(num_classes=num_classes, input_size=input_size)
        self.sigmoid = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, target = train_batch

        preds = self.model(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, target = valid_batch

        preds = self.model(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        X, target = test_batch

        preds = self.model(X)
        
        preds = preds.squeeze(1)
        loss = self.criterion(preds, target.float())
        
        preds = self.sigmoid(preds)
        accuracy = self.accuracy_metric(preds, target)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss
