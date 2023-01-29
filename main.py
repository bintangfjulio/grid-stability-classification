import argparse
import pytorch_lightning as pl

from util.preprocessor import Preprocessor
from util.classifier import Classifier
from util.regressor import Regressor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mining', choices=['klasifikasi', 'regresi'], required=True, help='Pilihan data mining')
    
    args = parser.parse_args()
    config = vars(args)

    module = Preprocessor(batch_size=64)
    num_classes, input_size = module.get_feature_size()
    
    if config['mining'] == 'klasifikasi':
        model = Classifier(lr=1e-3, num_classes=num_classes, input_size=input_size)
    elif config['mining'] == 'regresi':
        model = Regressor()
    
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{config['mining']}_bilstm_result', monitor='val_loss')
    logger = TensorBoardLogger('log', name=f'{config['mining']}_bilstm_result')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir=f'./checkpoints/{config['mining']}_bilstm_result',
        callbacks = [checkpoint_callback, early_stop_callback],
        deterministic=True,
        logger=logger)
    
    trainer.fit(model=model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
