import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchmetrics.classification import BinaryAccuracy

class CNN1D(pl.LightningModule):
    def __init__(self, lr, num_classes=2, in_channels=12, out_channels=96, window_sizes=[1, 2, 3, 4, 5], dropout=0.5):
        super(CNN1D, self).__init__()
        self.convolutional_layers = nn.ModuleList([nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=window_size) for window_size in window_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(out_channels * len(window_sizes), num_classes)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.accuracy_metric = BinaryAccuracy()
        self.lr = lr
        self.tanh = nn.Tanh()

    def forward(self, X):
        pooling_layer = [self.tanh(convolutional_layer(X)) for convolutional_layer in self.convolutional_layers] 
        max_pooling_layer = [F.max_pool1d(filtered_features, filtered_features.size(2)) for filtered_features in pooling_layer]

        flattened = torch.cat(max_pooling_layer, 1)
        flattened = logits.squeeze(-1)

        fully_connected_layer = self.fully_connected(self.dropout(flattened))
        preds = self.sigmoid(fully_connected_layer)

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-8, weight_decay=0.9)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        X, target = train_batch

        preds = self(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        X, target = valid_batch

        preds = self(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        X, target = test_batch

        preds = self(X)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy}, prog_bar=True, on_epoch=True)

        return loss
