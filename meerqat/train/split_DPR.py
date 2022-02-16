"""Usage: split_DPR.py <checkpoint_path>"""
from pathlib import Path
from docopt import docopt

import torch
import transformers


def split_DPRBiEncoder(checkpoint_path):
    """Utility function to split a DPRBiEncoder in DPRQuestionEncoder and DPRContextEncoder"""
    state_dict = torch.load(checkpoint_path/'pytorch_model.bin', map_location='cpu')
    question_model_state_dict = {k[len('question_model.'):]: v for k, v in state_dict.items() if k.startswith('question_model.')}
    question_model = transformers.DPRQuestionEncoder(transformers.DPRConfig())
    question_model.load_state_dict(question_model_state_dict)
    question_model.save_pretrained(checkpoint_path/'question_model')
    print(f"saved question_model at {checkpoint_path/'question_model'}")
    context_model = transformers.DPRContextEncoder(transformers.DPRConfig())
    context_model_state_dict = {k[len('context_model.'):]: v for k, v in state_dict.items() if k.startswith('context_model.')}
    context_model.load_state_dict(context_model_state_dict)
    context_model.save_pretrained(checkpoint_path/'context_model')
    print(f"saved context_model at {checkpoint_path/'context_model'}")


if __name__ == '__main__':
    args = docopt(__doc__)
    split_DPRBiEncoder(Path(args['<checkpoint_path>']))
