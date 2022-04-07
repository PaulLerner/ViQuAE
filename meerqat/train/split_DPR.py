"""
Usage: split_DPR.py <checkpoint_path> [--bert]

Options:
--bert              Save BertModel instead of DPRQuestionEncoder|DPRContextEncoder
"""
from pathlib import Path
from docopt import docopt

import torch
import transformers


def split_DPRBiEncoder(checkpoint_path, bert=False):
    """Utility function to split a DPRBiEncoder in DPRQuestionEncoder and DPRContextEncoder"""
    state_dict = torch.load(checkpoint_path/'pytorch_model.bin', map_location='cpu')
    question_model_state_dict = {k[len('question_model.'):]: v for k, v in state_dict.items() if k.startswith('question_model.')}
    question_model = transformers.DPRQuestionEncoder(transformers.DPRConfig())
    question_model.load_state_dict(question_model_state_dict)
    context_model = transformers.DPRContextEncoder(transformers.DPRConfig())
    context_model_state_dict = {k[len('context_model.'):]: v for k, v in state_dict.items() if k.startswith('context_model.')}
    context_model.load_state_dict(context_model_state_dict)
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
    bert = args['--bert']
    split_DPRBiEncoder(Path(args['<checkpoint_path>']), bert=bert)
