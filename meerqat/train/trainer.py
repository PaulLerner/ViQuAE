"""
usage: trainer.py [-h] [-c CONFIG]
                  [--print_config [={comments,skip_null,skip_default}+]]
                  {fit,validate,test,predict,tune} ...

Main training script based on pytorch lightning.

optional arguments:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
  --print_config [={comments,skip_null,skip_default}+]
                        Print the configuration after applying all other
                        arguments and exit.

subcommands:
  For more details of each subcommand add it as argument followed by --help.

  {fit,validate,test,predict,tune}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
    tune                Runs routines to tune hyperparameters before training.
"""
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(
        trainer_class=pl.Trainer, 
        # same default as transformers although it is unlikely that the calls are in the exact same order
        # N. B. called with `workers=True` in LightningCLI
        seed_everything_default=42, 
        description='Main training script based on pytorch lightning.'
    )
    return cli
    
    
if __name__ == "__main__":
    main()