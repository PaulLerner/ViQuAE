"""Usage: trainer.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import warnings
from tqdm import tqdm

import torch
from torch.autograd import set_detect_anomaly

from transformers import Trainer, TrainingArguments, trainer_callback, logging
from transformers.trainer_callback import TrainerState
from transformers.file_utils import WEIGHTS_NAME
from datasets import load_from_disk

from meerqat.data.loading import load_pretrained_in_kwargs


def get_checkpoint(resume_from_checkpoint: str, *args, **kwargs):
    if args or kwargs:
        warnings.warn(f"ignoring additional arguments:\n{args}\n{kwargs}")
    cpt = Path(resume_from_checkpoint)
    # weird trick to glob using pathlib
    resume_from_checkpoints = list(cpt.parent.glob(cpt.name))
    return resume_from_checkpoints


def instantiate_trainer(trainee, debug=False, train_dataset=None, eval_dataset=None, training_kwargs={}, callbacks_args=[], **kwargs):
    """Additional arguments are passed to Trainer"""
    # debug (see torch.autograd.detect_anomaly)
    set_detect_anomaly(debug)

    # data
    if train_dataset is not None:
        train_dataset = load_from_disk(train_dataset)
    if eval_dataset is not None:
        eval_dataset = load_from_disk(eval_dataset)

    # training
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(model=trainee, args=training_args,
                      train_dataset=train_dataset, eval_dataset=eval_dataset,
                      **kwargs)
    # training callbacks
    for callback in callbacks_args:
        CallbackClass = getattr(trainer_callback, callback.pop("Class"))
        trainer.add_callback(CallbackClass(**callback))

    return trainer, training_args


def write_predictions(predictions, resume_from_checkpoint):
    raise NotImplementedError()


def write_metrics(metrics, resume_from_checkpoint):
    print(metrics)
    with open(resume_from_checkpoint/"metrics.json", "w") as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    # load and parse arguments
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    with open(config_path, "r") as file:
        config = load_pretrained_in_kwargs(json.load(file))

    verbosity = config.get("verbosity")
    if verbosity is not None:
        logging.set_verbosity(verbosity)

    checkpoint = config.pop("checkpoint", {})
    trainer, training_args = instantiate_trainer(**config)

    if training_args.do_train:
        trainer.train(**checkpoint)
    elif training_args.do_eval:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Evaluation"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path)
            trainer.model.load_state_dict(state_dict)

            # optionally load trainer state for better logging
            trainer_state = resume_from_checkpoint/"trainer_state.json"
            if trainer_state.is_file():
                trainer.state = TrainerState.load_from_json(trainer_state)
            else:
                warnings.warn("couldn't load trainer state, TB logging might use an inappropriate step")
            metrics = trainer.evaluate()
            write_metrics(metrics, resume_from_checkpoint)
    elif training_args.do_predict:
        resume_from_checkpoints = get_checkpoint(**checkpoint)
        for resume_from_checkpoint in tqdm(resume_from_checkpoints, desc="Prediction"):
            # load state dict
            state_dict_path = resume_from_checkpoint / WEIGHTS_NAME
            if not state_dict_path.exists():
                continue
            state_dict = torch.load(state_dict_path)
            trainer.model.load_state_dict(state_dict)

            # run model on evaluation dataset
            eval_dataloader = trainer.get_eval_dataloader()
            predictions = trainer.predict(eval_dataloader)
            write_predictions(predictions, resume_from_checkpoint)
    else:
        warnings.warn("Did nothing except instantiate the trainer, "
                      "you probably want to set do_train, do_eval or do_predict to True"
                      f"see {training_args.__doc__}")
