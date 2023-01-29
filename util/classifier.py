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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, target = train_batch

        preds = self.model(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, target = valid_batch

        preds = self.model(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        X, target = test_batch

        preds = self.model(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss
