import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from util.preprocessor import Preprocessor
from util.classifier import Classifier

if __name__ == "__main__":
    pl.seed_everything(42)

    module = Preprocessor(batch_size=64)
    num_classes, input_size = module.get_feature_size()
    
    model = Classifier(lr=1e-3, num_classes=num_classes, input_size=input_size)
    
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/bilstm_result', monitor='val_loss')
    logger = TensorBoardLogger('log', name='bilstm_result')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, check_on_train_epoch_end=1, patience=10)

    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=100,
        default_root_dir='./checkpoints/bilstm_result',
        callbacks = [checkpoint_callback, early_stop_callback],
        deterministic=True,
        logger=logger)
    
    trainer.fit(model=model, datamodule=module)
    trainer.test(model=model, datamodule=module, ckpt_path='best')
