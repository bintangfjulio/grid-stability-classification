import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics.classification import BinaryAccuracy

class Classifier(pl.LightningModule):
    def __init__(self, lr, num_classes, input_size, dropout=0.5):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.accuracy_metric = BinaryAccuracy()
        self.lr = lr
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 8)
        self.layer7 = nn.Linear(8, num_classes)

    def forward(self, X):
        hidden_layer1 = self.relu(self.layer1(X))
        hidden_layer2 = self.relu(self.layer2(hidden_layer1))
        hidden_layer3 = self.relu(self.layer3(hidden_layer2))
        hidden_layer4 = self.relu(self.layer4(hidden_layer3))
        hidden_layer5 = self.relu(self.layer5(hidden_layer4))
        hidden_layer6 = self.relu(self.layer6(hidden_layer5))
        preds = self.sigmoid(self.layer7(hidden_layer6))

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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
