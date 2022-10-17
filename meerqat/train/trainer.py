"""
usage: trainer.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
        {fit,validate,test,predict,tune} ...

Main training script based on pytorch lightning. Also holds Trainer and LightningCLI subclasses.

optional arguments:
-h, --help            Show this help message and exit.
-c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
--print_config [={comments,skip_null,skip_default}+]
                        Print configuration and exit.

subcommands:
For more details of each subcommand add it as argument followed by --help.

{fit,validate,test,predict,tune}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
    tune                Runs routines to tune hyperparameters before training.
    split               Splits a BiEncoder in two (e.g. BiEncoder in DPRQuestionEncoder and DPRContextEncoder).
"""
from pathlib import Path
from typing import Optional, Union
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

import pytorch_lightning as pl
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.cli import LightningCLI


class Trainer(pl.Trainer):
    # FIXME: no need to load data to split model. Can we control this from CLI or Trainer ?
    def split(
            self,         
            model: "pl.LightningModule",
            train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
            val_dataloaders: Optional[EVAL_DATALOADERS] = None,
            datamodule: Optional[LightningDataModule] = None,
            ckpt_path: Optional[str] = None,
            bert: Optional[bool] = False
        ):
        """
        Splits a BiEncoder in two (e.g. BiEncoder in DPRQuestionEncoder and DPRContextEncoder).
        
        Parameters
        ----------
        ckpt_path: str
            Path to the pytorch-lightning checkpoint
        bert: bool, optional
            Save Encoder sub-BertModel instead of the full Encoder (which then must have bert_model attribute).
            Defaults to False.       
        """
        # used as root directory
        ckpt_path = Path(ckpt_path).with_suffix('')
        question_model = model.question_model
        context_model = model.context_model
        if bert:
            question_model = question_model.question_encoder.bert_model
            context_model = context_model.ctx_encoder.bert_model
            question_path = ckpt_path/'question_model_bert'
            context_path = ckpt_path/'context_model_bert'
        else:
            question_path = ckpt_path/'question_model'
            context_path = ckpt_path/'context_model'
        question_model.save_pretrained(question_path)
        context_model.save_pretrained(context_path)
        print(f"saved question_model at {question_path}")
        print(f"saved context_model at {context_path}")
        
        
class CLI(LightningCLI):        
    def subcommands(self):
        lightning_commands = super().subcommands()
        lightning_commands["split"] = {"model", "train_dataloaders", "val_dataloaders", "datamodule"}
        return lightning_commands


def main():
    cli = CLI(
        trainer_class=Trainer, 
        # same default as transformers although it is unlikely that the calls are in the exact same order
        seed_everything_default=42, 
        description='Main training script based on pytorch lightning.  Also holds Trainer and LightningCLI subclasses.'
    )
    return cli
    
    
if __name__ == "__main__":
    main()