import torch
import torch.nn as nn
import pytorch_lightning as pl

class BiLSTM(pl.LightningModule):
    def __init__(self, num_classes, lr, input_size, hidden_size, dropout=0.5, num_layers=2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.output_layer = nn.Linear(hidden_size * 2, num_classes) 
        self.sigmoid = nn.Sigmoid()   
        self.criterion = nn.BCELoss()
        self.accuracy_metric = MulticlassAccuracy(num_classes=num_classes)
        self.lr = lr

    def forward(self, X):        
        _, (lstm_output_layer, _) = self.lstm(X)
        
        sequential_direction_backward = lstm_output_layer[-2]
        sequential_direction_forward = lstm_output_layer[-1]
        lstm_output = torch.cat([sequential_direction_backward, sequential_direction_forward], dim=-1)   

        fully_connected_layer = self.output_layer(self.dropout(lstm_output))
        preds = self.sigmoid(fully_connected_layer)
        
        return preds
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
      
    def training_step(self, train_batch, batch_idx):
        input_ids, target = train_batch

        preds = self.model(input_ids=input_ids)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)
        mcc = self.mcc_metric(max_pred_idx, max_target_idx)

        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, valid_batch, batch_idx):
        input_ids, target = valid_batch

        preds = self.model(input_ids=input_ids)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)
        mcc = self.mcc_metric(max_pred_idx, max_target_idx)

        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        input_ids, target = test_batch

        preds = self.model(input_ids=input_ids)
        loss = self.criterion(preds, target=target.float())

        max_pred_idx = preds.argmax(1)
        max_target_idx = target.argmax(1)
        accuracy = self.accuracy_metric(max_pred_idx, max_target_idx)
        mcc = self.mcc_metric(max_pred_idx, max_target_idx)

        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_mcc': mcc}, prog_bar=True, on_epoch=True)

        return loss
