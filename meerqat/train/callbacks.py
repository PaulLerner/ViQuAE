# -*- coding: utf-8 -*-
from typing import Optional
from pytorch_lightning.callbacks import Callback


class TestAfterFit(Callback):   
    """
    Calls trainer.test with 'best' ckpt on fit end 
    (so make sure you configure ModelCheckpoint to save best model).
    
    Parameters
    ----------
    data_update: dict, optional
        Arguments of trainer.datamodule to update before running test
        I.e. differences between your validation and test setups
        E.g. for re-ranking, you might want to pass:
        {
          "M": 100, # re-rank top-100 passages
          "eval_batch_size": 2, # lower batch size to fit in a GPU
          "run_path": "/path/to/test/run.json",
          "qrels_path": "/path/to/test/qrels.json"
        }
    """
    def __init__(self, data_update: Optional[dict] = None):
        super().__init__()
        self.data_update = data_update
        
    def on_fit_end(self, trainer, pl_module):        
        if self.data_update is not None:
            for k, v in self.data_update.items():
                if not hasattr(trainer.datamodule, k):
                    raise AttributeError(f"{trainer.datamodule.__class__.__name__} has no attribute '{k}'")
                setattr(trainer.datamodule, k, v)
        # N. B. dataloader reloading is done in Trainer._run_evaluate
        trainer.test(pl_module, ckpt_path="best")
