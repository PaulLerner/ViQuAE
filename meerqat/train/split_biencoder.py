"""
Usage: split_biencoder.py <config> [--bert]

Options:
--bert              Save BertModel instead of DPRQuestionEncoder|DPRContextEncoder
"""
from pathlib import Path
from docopt import docopt
import json

import torch
from transformers.file_utils import WEIGHTS_NAME

from meerqat.data.loading import load_pretrained_in_kwargs


def split_biencoder(trainee, checkpoint, bert=False):
    """Utility function to split a DPRBiEncoder in DPRQuestionEncoder and DPRContextEncoder"""
    checkpoint_path = Path(checkpoint['resume_from_checkpoint'])
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
    split_biencoder(**config, bert=bert)
