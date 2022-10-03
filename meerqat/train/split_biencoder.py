"""
Splits a BiEncoder in two (e.g. DPRBiEncoder in DPRQuestionEncoder and DPRContextEncoder).
The config file should be the same as for ``train.trainer``.

You might get a warning like "weights were not used", this comes from load_pretrained_in_kwargs, don’t worry:
If the checkpoint does not match the model an exception will be raised.

Usage: split_biencoder.py <config> [<checkpoint> --bert]

Positional arguments:
    <config>        Path to the JSON configuration file (passed as kwargs)
    <checkpoint>    Path to the BiEncoder checkpoint, optional. 
                    If not provided, should be in the config file under checkpoint.resume_from_checkpoint,
                    like the eval mode of ``train.trainer`.                    

Options:
    --bert      Save BertModel instead of question_model and context_model
"""
from pathlib import Path
from docopt import docopt
import json

import torch
from transformers.file_utils import WEIGHTS_NAME

from ..data.loading import load_pretrained_in_kwargs


def split_biencoder(trainee, checkpoint, bert=False):
    """
    Utility function to split a BiEncoder in two 
    (e.g. DPRBiEncoder in DPRQuestionEncoder and DPRContextEncoder)
    
    Parameters
    ----------
    trainee: BiEncoder
        must have attributes:
            - question_model: PreTrainedModel
            - context_model: PreTrainedModel
    checkpoint: str
        Path to the directory where the pytorch checkpoint is stored under WEIGHTS_NAME ("pytorch_model.bin")
    bert: bool, optional
        Save BertModel instead of question_model and context_model (which then must have bert_model attribute).
        Defaults to False.        
    """
    checkpoint_path = Path(checkpoint)
    state_dict = torch.load(checkpoint_path/WEIGHTS_NAME, map_location='cpu')
    trainee.load_state_dict(state_dict)
    question_model = trainee.question_model
    context_model = trainee.context_model
    if bert:
        question_model = question_model.question_encoder.bert_model
        context_model = context_model.ctx_encoder.bert_model
        question_path = checkpoint_path/'question_model_bert'
        context_path = checkpoint_path/'context_model_bert'
    else:
        question_path = checkpoint_path/'question_model'
        context_path = checkpoint_path/'context_model'
    question_model.save_pretrained(question_path)
    context_model.save_pretrained(context_path)
    print(f"saved question_model at {question_path}")
    print(f"saved context_model at {context_path}")


if __name__ == '__main__':
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = load_pretrained_in_kwargs(json.load(file))
    
    bert = args['--bert']
    checkpoint = args['<checkpoint>'] if args['<checkpoint>'] is not None else config['checkpoint']['resume_from_checkpoint']
    split_biencoder(config['trainee'], checkpoint, bert=bert)
