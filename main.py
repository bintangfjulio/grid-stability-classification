import pytorch_lightning as pl

from util.preprocessor import Preprocessor

if __name__ == "__main__":
    pl.seed_everything(42)

    module = Preprocessor(batch_size=100)
    module.preprocessor()