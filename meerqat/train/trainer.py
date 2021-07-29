"""Usage: trainer.py <config>"""
from docopt import docopt
import json
from pathlib import Path
import warnings
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import set_detect_anomaly

from transformers import Trainer, TrainingArguments, trainer_callback, logging
from transformers.trainer_callback import TrainerState
from transformers.file_utils import WEIGHTS_NAME
from datasets import load_from_disk

from meerqat.data.loading import load_pretrained_in_kwargs


class MeerqatTrainer(Trainer):
    """Base class for all trainers. Should be very similar to Trainer"""
    def log(self, logs: Dict[str, float]) -> None:
        """Adds memory usage to the logs"""
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            logs[f"max_memory_{device}"] = torch.cuda.max_memory_allocated(device)
        return super().log(logs)

        
class MultiPassageBERTTrainer(MeerqatTrainer):
    """
    Overrides some methods because we need to create the batch of questions and passages on-the-fly

    Because the inputs should be shaped like (N * M, L), where:
            N - number of distinct questions
            M - number of passages per question in a batch
            L - sequence length

    Parameters
    ----------
    *args, **kwargs: additional arguments are passed to Trainer
    kb: str
        path towards the knowledge base (Dataset) used to get the passages
    M: int, optional
        Number of passages (relevant or irrelevant) per question in a batch
        Defaults to 24
    n_relevant_passages: int, optional
        Defaults to 1
    max_n_answers: int, optional
        The answer might be found several time in the same passage, this is a threshold to enable batching
        Defaults to 10.
    eval_search_key: str, optional
        This column in the dataset should hold the result of information retrieval (e.g. the output of ir.search)
        It is not used during training.
        Defaults to 'search_indices'
    tokenization_kwargs: dict, optional
        To be passed to self.tokenizer
    """
    def __init__(self, *args, kb, M=24, n_relevant_passages=1, max_n_answers=10, eval_search_key='search_indices',
                 tokenization_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.tokenizer is not None
        self.kb = load_from_disk(kb)
        self.M = M
        assert n_relevant_passages <= M
        self.n_relevant_passages = n_relevant_passages
        self.max_n_answers = max_n_answers
        self.eval_search_key = eval_search_key
        default_tokenization_kwargs = dict(return_tensors='pt', padding=True, truncation=True)
        if tokenization_kwargs is None:
            tokenization_kwargs = {}
        default_tokenization_kwargs.update(tokenization_kwargs)
        self.tokenization_kwargs = default_tokenization_kwargs
        self.data_collator = self.collate_fn

        # we need those ‘un-used’ columns to actually create the batch the model will use
        if self.args.remove_unused_columns:
            warnings.warn(f'Setting args.remove_unused_columns to False')
            self.args.remove_unused_columns = False

    def get_training_passages(self, item):
        relevant_passages = []
        n_relevant = min(len(item['provenance_index']), self.n_relevant_passages)
        if n_relevant > 0:
            relevant_indices = np.random.choice(item['provenance_index'], n_relevant, replace=False)
            if len(relevant_indices) > 0:
                relevant_passages = self.kb.select(relevant_indices)['passage']
        irrelevant_passages = []
        n_irrelevant = min(len(item['irrelevant_index']), self.M-self.n_relevant_passages)
        if n_irrelevant > 0:
            irrelevant_indices = np.random.choice(item['irrelevant_index'], n_irrelevant, replace=False)
            if len(irrelevant_indices) > 0:
                irrelevant_passages = self.kb.select(irrelevant_indices)['passage']
        elif n_relevant <= 0:
            warnings.warn(f"Didn't find any passage for question {item['id']}")
        return relevant_passages+irrelevant_passages

    def get_eval_passages(self, item):
        """Keep the top-M passages retrieved by the IR"""
        return item[self.eval_search_key][: self.M]

    def get_passages(self, *args, **kwargs):
        if self.args.do_eval or self.args.do_predict:
            return self.get_eval_passages(*args, **kwargs)
        return self.get_training_passages(*args, **kwargs)

    def get_answer_position(self, batch, answers, answer_mask):
        """Adapted from DPR"""
        start_positions, end_positions = torch.zeros_like(answer_mask), torch.zeros_like(answer_mask)
        for j, (input_ids, answer) in enumerate(zip(batch['input_ids'], answers)):
            L = input_ids.size(-1)
            answer_starts, answer_ends = [], []
            for a in answer:
                answer_len = a.size(0)
                for i in range(L-answer_len+1):
                    if (a == input_ids[i: i+answer_len]).all():
                        start, end = i, i+answer_len-1
                        if start not in answer_starts and end not in answer_ends:
                            answer_starts.append(start)
                            answer_ends.append(end)
                            if len(answer_starts) >= self.max_n_answers:
                                break
                for i, (start, end) in enumerate(zip(answer_starts, answer_ends)):
                    start_positions[j, i] = start
                    end_positions[j, i] = end
                    # un-mask answer
                    answer_mask[j, i] = 1
        start_positions = start_positions.view(-1, self.M, self.max_n_answers)
        end_positions = end_positions.view(-1, self.M, self.max_n_answers)
        answer_mask = answer_mask.view(-1, self.M, self.max_n_answers)
        batch.update(dict(start_positions=start_positions, end_positions=end_positions, answer_mask=answer_mask))
        return batch

    def collate_fn(self, items):
        """
        Collate batch so that each question is associate with n_relevant_passages and M-n irrelevant ones.
        Also tokenizes input strings

        Returns (a dict of)
        -------------------
        input_ids: Tensor[int]
            shape (N * M, L)
        start_positions, end_positions: Tensor[int]
            shape (N, M, max_n_answers)
        answer_mask: Tensor[int]
            shape (N, M, max_n_answers)
        **kwargs: more tensors depending on the tokenizer, e.g. attention_mask
        """
        questions, passages = [], []
        answers = []
        N = len(items)
        answer_mask = torch.zeros((N*self.M, self.max_n_answers), dtype=torch.long)
        for i, item in enumerate(items):
            # N. B. seed is set in Trainer
            questions.extend([item['input']]*self.M)
            passage = self.get_passages(item)
            passages.extend(passage)
            # all passages have at least 1 non-masked answer (set to 0 for irrelevant passages)
            answer_mask[i*self.M: i*self.M+len(passage), 0] = 1
            # except for padding passages
            if len(passage) < self.M:
                passages.extend(['']*(self.M-len(passage)))

            original_answer = item['output']['original_answer']
            # avoid processing the same answer twice
            answer = item['output']['answer']
            if self.tokenizer.do_lower_case:
                answer = list({a.lower() for a in answer} - {original_answer})
            # but ensure the original answer is still the first to be processed
            answer = original_answer + answer
            answer = self.tokenizer(answer,
                                    add_special_tokens=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False)['input_ids']
            answer = [torch.tensor(a, dtype=torch.long) for a in answer]
            answers.extend([answer]*self.M)
        batch = self.tokenizer(*(questions, passages), **self.tokenization_kwargs)
        batch = self.get_answer_position(batch, answers, answer_mask)
        return batch


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
    trainer = MultiPassageBERTTrainer(model=trainee, args=training_args,
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
