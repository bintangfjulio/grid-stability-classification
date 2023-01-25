import torch
import os
import multiprocessing
import pandas as pd
import pytorch_lightning as pl

from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
from torch.utils.data import TensorDataset, DataLoader

class Preprocessor(pl.LightningDataModule):
    def __init__(self, batch_size):
        super(Preprocessor, self).__init__()    
        self.dataset = pd.read_csv('dataset/simulated_electrical_grid.csv')
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_set, test_set = self.preprocessor(self.dataset)   
        if stage == "fit":
            self.train_set = train_set
        elif stage == "test":
            self.test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=multiprocessing.cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count()
        )

    def preprocessor(self, dataset):
        dataset = self.normalization(dataset)

        X_train, y_train = self.oversampling(dataset)
        y_train = self.label_encoding(y_train)

        self.feature_size = len(X_train.columns.tolist())

    def normalization(self, dataset):
        dataset['tau1 Z'] = zscore(dataset['tau1'])
        dataset['tau2 Z'] = zscore(dataset['tau2'])
        dataset['tau3 Z'] = zscore(dataset['tau3'])
        dataset['tau4 Z'] = zscore(dataset['tau4'])
        dataset['p1 Z'] = zscore(dataset['p1'])
        dataset['p2 Z'] = zscore(dataset['p2'])
        dataset['p3 Z'] = zscore(dataset['p3'])
        dataset['p4 Z'] = zscore(dataset['p4'])
        dataset['g1 Z'] = zscore(dataset['g1'])
        dataset['g2 Z'] = zscore(dataset['g2'])
        dataset['g3 Z'] = zscore(dataset['g3'])
        dataset['g4 Z'] = zscore(dataset['g4'])

        return dataset

    def label_encoding(self, y_train):
        encoder = {'unstable': [1, 0], 'stable': [0, 1]}
        y_train = y_train.astype('str').map(encoder)

        return y_train

    def oversampling(self, dataset):
        X = dataset.drop(['tau1','tau2','tau3','tau4','p1', 'p2', 'p3', 'p4','g1','g2','g3','g4','stab','stabf'], axis=1)
        y = dataset['stabf']     
        oversampling = SMOTE(random_state=42)
        X_train, y_train = oversampling.fit_resample(X, y)

        return X_train, y_train

    def get_feature_size(self):
        return self.feature_size
