import os
import torch
import multiprocessing
import pandas as pd
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

class Preprocessor(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(Preprocessor, self).__init__()    
        self.dataset = pd.read_csv('dataset/simulated_electrical_grid.csv')
        self.batch_size = batch_size
        self.oversampling = SMOTE(random_state=42)
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        train_set, valid_set, test_set = self.preprocessor()   
        if stage == "fit":
            self.train_set = train_set
            self.valid_set = valid_set
        elif stage == "test":
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count()
        )

    def preprocessor(self):
        if os.path.exists("dataset/train_set.pt") and os.path.exists("dataset/valid_set.pt") and os.path.exists("dataset/test_set.pt"):
            print("\nLoading Data...")
            train_set = torch.load("dataset/train_set.pt")
            valid_set = torch.load("dataset/valid_set.pt")
            test_set = torch.load("dataset/test_set.pt")
            print('[ Loading Completed ]\n')
        else:
            print("\nPreprocessing Data...")
            train_set, valid_set, test_set = self.preprocessing_data(self.dataset)
            print('[ Preprocessing Completed ]\n')

        return train_set, valid_set, test_set
    
    def preprocessing_data(self, dataset):
        X = dataset[['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']]
        y = dataset['stabf']

        X_train_res, y_train_res = self.oversampling.fit_resample(X, y)

        self.scaler.fit(X_train_res)
        X_train_res = self.scaler.transform(X_train_res)
        X_train_res = pd.DataFrame(X_train_res, columns=X.columns)  

        y_train_res = self.label_encoding(y_train_res)

        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X_train_res, y_train_res, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.1, random_state=42)
    
        X_train_tensor = torch.tensor(X_train.values.tolist())
        y_train_tensor = torch.tensor(y_train.values.tolist())
        
        X_valid_tensor = torch.tensor(X_valid.values.tolist())
        y_valid_tensor = torch.tensor(y_valid.values.tolist())
        
        X_test_tensor = torch.tensor(X_test.values.tolist())
        y_test_tensor = torch.tensor(y_test.values.tolist())

        train_set = TensorDataset(X_train_tensor, y_train_tensor)
        valid_set = TensorDataset(X_valid_tensor, y_valid_tensor)
        test_set = TensorDataset(X_test_tensor, y_test_tensor)
    
        torch.save(train_set, "dataset/train_set.pt")
        torch.save(valid_set, "dataset/valid_set.pt")
        torch.save(test_set, "dataset/test_set.pt")

        return train_set, valid_set, test_set

    def label_encoding(self, y_train):
        encoder = {'unstable': 0, 'stable': 1}
        y_train = y_train.astype('str').map(encoder)

        return y_train

    def get_feature_size(self):
        X = self.dataset[['tau1','tau2','tau3','tau4','p1', 'p2', 'p3', 'p4','g1','g2','g3','g4']]
        
        return 1, len(X.columns.tolist())
