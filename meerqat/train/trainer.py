"""
usage: trainer.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
        {fit,validate,test,predict,tune} ...

Main training script based on pytorch lightning.

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
"""
import warnings
import collections

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from ..models.qa import get_best_spans, format_predictions_for_squad


class ICTTrainer:
    """
    Extends the Inverse Cloze Task (ICT, [2]_) to multimodal documents.
    Given a wikipedia section, one sentence is considered as a pseudo-question/query and the nearby sentences as a pseudo-target/relevant passage.
    In this multimodal setting, we also consider the image of the section in the query and the infobox/main image of the article in the target.

    Inherits from MMTrainer/DPRBiEncoderTrainer and overrides:
        - ``get_training_passages``, which implements what’s described above
        - ``collate_fn`` to load and concatenate the image features

    The image_kb, kb and search_key attributes are not used.
    
    Parameters
    ----------
    *args, **kwargs: 
        additional arguments are passed to MMTrainer
    sentences_per_target: int, optional
        Number of sentences in the target passages
    prepend_title: bool, optional
        Whether to preprend the title of the article to the target passage
    text_mask_rate: float, optional
        Rate at which the pseudo-question is masked in the target passage
    image_mask_rate: float, optional
        Rate at which the infobox image is used as target (keep input image otherwise)
        
    References
    ----------
    .. [2] Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent Retrieval for Weakly Supervised Open Domain Question Answering. 
       In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 
       pages 6086–6096, Florence, Italy. Association for Computational Linguistics.
    """
    def __init__(self, *args, sentences_per_target=4, prepend_title=False, 
                 text_mask_rate=1.0, image_mask_rate=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError("Compatible with transformers but not lightning")
        self.kb = None
        self.image_kb = None
        self.sentences_per_target = sentences_per_target
        self.prepend_title = prepend_title
        self.text_mask_rate = text_mask_rate
        self.image_mask_rate = image_mask_rate
        self.data_collator = self.collate_fn

    def get_training_passages(self, item):
        """
        Beware this does not return the same data as the parent classes.

        Returns
        -------
        query: dict
        target: dict
        """
        sentences = item['sentences']

        # pick a random sentence: easy
        i = np.random.randint(len(sentences))
        query = dict(text=sentences[i]['text'])
        # pick n random sentences around it: more tricky
        n = min(self.sentences_per_target, len(sentences)-1)
        max_shift = min(i, n)
        if i+n < len(sentences):
            min_shift = 0
        else:
            min_shift = i + n - len(sentences) + 1
        shift = np.random.randint(min_shift, max_shift+1)
        # standard ICT: remove sentence from its context
        if np.random.rand() < self.text_mask_rate:
            target = [s['text'] for s in sentences[i-shift: i]+sentences[i+1: i+1+n-shift]]
        # robustness trick: keep the sentence in the context so that the model learns lexical overlap
        else:
            target = [s['text'] for s in sentences[i-shift: i+1+n-shift]]
            
        if self.prepend_title:
            target.insert(0, self.tokenizer.sep_token)
            target.insert(0, item['title'])
        target = dict(text=" ".join(target))  
        
        # standard MICT: use the contextual image as target
        if np.random.rand() < self.image_mask_rate:
            context_image_key = "context_"
        # robustness trick: use the same image in query/target so that the model keeps image information
        else:
            context_image_key = ""
        # rename context image features
        for k in ({"face_box", "face_embedding"} | self.image_embeddings_keys):
            target[k] = item.get(f"{context_image_key}{k}")
        return query, target

    def _get_eval_sampler(self, eval_dataset):
        """
        It’s important to shuffle even the evaluation set because the examples 
        get really hard (even false negatives) if they come from the same Wikipedia article.
        """
        if self.args.use_legacy_prediction_loop:
            raise NotImplementedError()
        # Build the sampler.
        if self.args.group_by_length or self.args.world_size > 1:
            raise NotImplementedError()
        return RandomSampler(eval_dataset)

    def collate_fn(self, items):
        """"""
        questions, relevant_passages, labels = [], [], []
        for i, item in enumerate(items):
            query, relevant_passage = self.get_training_passages(item)
            labels.append(i)
            questions.append(query)
            relevant_passages.append(relevant_passage)

        question_inputs_text = self.tokenizer([q['text'] for q in questions], **self.tokenization_kwargs)
        context_inputs_text = self.tokenizer([p['text'] for p in relevant_passages], **self.tokenization_kwargs)
        # get image features, for both questions and passages
        question_inputs = dict(
            text_inputs=question_inputs_text, 
            face_inputs=self.get_face_inputs(items), 
            image_inputs=self.get_image_inputs(items)
        )
        context_inputs = dict(
            text_inputs=context_inputs_text, 
            face_inputs=self.get_face_inputs(relevant_passages), 
            image_inputs=self.get_image_inputs(relevant_passages)
        )

        # make n_irrelevant_passages by shifting the images of relevant passages
        n_irrelevant_passages = self.M-self.n_relevant_passages
        if n_irrelevant_passages > 0:
            # duplicate relevant text
            for k, v in context_inputs["text_inputs"].items():
                context_inputs["text_inputs"][k] = torch.tile(v, (self.M, 1))
            # shift relevant images
            for k, v in context_inputs['image_inputs'].items():
                shifted_input, shifted_mask = [v['input']], [v['attention_mask']]
                for shift in range(n_irrelevant_passages):
                    # shift along axis 0 (batch axis)
                    shifted_input.append(torch.roll(v['input'], shift+1, 0))
                    shifted_mask.append(torch.roll(v['attention_mask'], shift+1, 0))
                # cat along axis 0 (batch axis)
                v['input'] = torch.cat(shifted_input, 0)
                v['attention_mask'] = torch.cat(shifted_mask, 0)
            # shift relevant faces
            shifted_faces, shifted_boxes = [context_inputs['face_inputs']["face"]], [context_inputs['face_inputs']["bbox"]]
            shifted_mask = [context_inputs['face_inputs']['attention_mask']]
            for shift in range(n_irrelevant_passages):
                # shift along axis 0 (batch axis)
                shifted_faces.append(torch.roll(context_inputs['face_inputs']["face"], shift+1, 0))
                shifted_boxes.append(torch.roll(context_inputs['face_inputs']["bbox"], shift+1, 0))
                shifted_mask.append(torch.roll(context_inputs['face_inputs']["attention_mask"], shift+1, 0))
            # cat along axis 0 (batch axis)
            context_inputs['face_inputs']["face"] = torch.cat(shifted_faces, 0)
            context_inputs['face_inputs']["bbox"] = torch.cat(shifted_boxes, 0)
            context_inputs['face_inputs']['attention_mask'] = torch.cat(shifted_mask, 0)

        # wrap it up
        labels = torch.tensor(labels)
        batch = dict(question_inputs=question_inputs, context_inputs=context_inputs, labels=labels)
        return batch


class MultiPassageBERTTrainer:
    """
    Specific for RC, more precisely MultiPassageBERT
    (will I manage to code an extra-level of abstraction, e.g. ReadingComprehensionTrainer?)

    Parameters
    ----------
    *args, **kwargs: 
        additional arguments are passed to QuestionAnsweringTrainer
    max_n_answers: int, optional
        The answer might be found several time in the same passage, this is a threshold to enable batching
        Defaults to 10.
    ignore_keys: List[str], optional
        List of keys to remove from the batch before feeding it to the model
        (data not used by the model but necessary for evaluation)
        Defaults to ['answer_strings']
    train_original_answer_only: bool, optional
        Whether the model should be trained to predict only the original answer (default)
        or all alternative answers (with the only limit of max_n_answers)
        This has no effect on the evaluation (where all alternative answers are always considered)
    oracle: bool, optional
        Whether to use only relevant passages at inference (stored in {search_key}_provenance_indices)
        Will enforce n_relevant_passages=M
        Defaults to False (use IR passages at inference, stored in {search_key}_indices)
    run_path: str, optional
        Path to the ranx run stored in the TREC format that holds the IR results.
        To be used instead of search_key at inference.
        Defaults to None.
    """
    def __init__(self, *args, max_n_answers=10, ignore_keys=['answer_strings'], 
                 train_original_answer_only=True, oracle=False, run_path=None, **kwargs):
        raise NotImplementedError("Compatible with transformers but not lightning")
        super().__init__(*args, **kwargs)
        self.max_n_answers = max_n_answers
        self.ignore_keys = ignore_keys
        self.train_original_answer_only = train_original_answer_only
        self.oracle = oracle
        if self.oracle:
            self.prediction_file_name = "oracle_predictions.json"
            self.metrics_file_name = "oracle_metrics.json"
            if self.n_relevant_passages != self.M:
                warnings.warn(f"Oracle mode. Setting n_relevant_passages={self.M}")
                self.n_relevant_passages = self.M
                
        if run_path is not None:
            self.run = ranx.Run.from_file(run_path, 'trec')
        else:
            self.run = None

        # FIXME isn't there a more robust way of defining data_collator as the method collate_fn ?
        self.data_collator = self.collate_fn

    def get_eval_passages(self, item):
        """Keep the top-M passages retrieved by the IR"""
        if self.run is None:
            indices = item[self.search_key+"_indices"][: self.M]
            scores = item[self.search_key+"_scores"][: self.M]
        else:
            ir_results = self.run.run[item['id']]
            # document ids in ranx are str so we map them back to indices (int)
            indices = list(map(int, ir_results.keys()))[: self.M]
            scores = list(ir_results.values())[: self.M]
            
        return self.kb.select(indices), scores

    def get_answer_position(self, batch, answers, answer_mask):
        """Adapted from DPR"""
        start_positions, end_positions = torch.zeros_like(answer_mask), torch.zeros_like(answer_mask)
        for j, (input_ids, answer) in enumerate(zip(batch['input_ids'], answers)):
            L = input_ids.size(-1)
            answer_starts, answer_ends = [], []
            for a in answer:
                answer_len = a.size(0)
                enough = False
                for i in range(L-answer_len+1):
                    if (a == input_ids[i: i+answer_len]).all():
                        start, end = i, i+answer_len-1
                        if start not in answer_starts and end not in answer_ends:
                            answer_starts.append(start)
                            answer_ends.append(end)
                            if len(answer_starts) >= self.max_n_answers:
                                enough = True
                                break
                if enough:
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
        passage_scores: Tensor[float], optional
            shape (N * M)
            only in evaluation mode
        **kwargs: more tensors depending on the tokenizer, e.g. attention_mask
        """
        questions, passages = [], []
        answers, answer_strings = [], []
        passage_scores = []
        N = len(items)
        answer_mask = torch.zeros((N*self.M, self.max_n_answers), dtype=torch.long)
        for i, item in enumerate(items):
            # N. B. seed is set in Trainer
            questions.extend([item['input']]*self.M)

            # oracle -> use only relevant passages
            if (self.args.do_eval or self.args.do_predict) and not self.oracle:
                passage, score = self.get_eval_passages(item)
                passage = passage['passage']
                passage_scores.extend(score)
                if len(score) < self.M:
                    passage_scores.extend([0]*(self.M-len(score)))
            else:
                relevant_passage, irrelevant_passage = self.get_training_passages(item)
                passage = [p['passage'] for p in relevant_passage] + [p['passage'] for p in irrelevant_passage]

            passages.extend(passage)
            # all passages have at least 1 non-masked answer (set to 0 for irrelevant passages)
            answer_mask[i*self.M: i*self.M+len(passage), 0] = 1
            # except for padding passages
            if len(passage) < self.M:
                passages.extend(['']*(self.M-len(passage)))

            original_answer = item['output']['original_answer']
            # avoid processing the same answer twice
            answer = item['output']['answer']
            answer_strings.extend([answer]*self.M)
            # beware this create a discrepancy between answer_strings and answers (tokens)
            # evaluation should always be done using answer_strings
            if self.train_original_answer_only:
                answer = [original_answer]
            else:
                if self.tokenizer.do_lower_case:
                    original_answer = original_answer.lower()
                    answer = list({a.lower() for a in answer} - {original_answer})
                # but ensure the original answer is still the first to be processed
                answer = [original_answer] + answer
            answer = self.tokenizer(answer,
                                    add_special_tokens=False,
                                    return_token_type_ids=False,
                                    return_attention_mask=False)['input_ids']
            answer = [torch.tensor(a, dtype=torch.long) for a in answer]
            answers.extend([answer]*self.M)
        batch = self.tokenizer(*(questions, passages), **self.tokenization_kwargs)
        batch = self.get_answer_position(batch, answers, answer_mask)
        batch['answer_strings'] = answer_strings
        if passage_scores:
            batch['passage_scores'] = torch.tensor(passage_scores)

        return batch

    def _prepare_inputs(self, inputs: dict) -> dict:
        """remove all keys not used by the model but necessary for evaluation before returning Trainer._prepare_inputs"""
        for k in self.ignore_keys:
            if k not in inputs:
                warnings.warn(f"Didn't find {k} in inputs")
                continue
            inputs.pop(k)
        return super()._prepare_inputs(inputs)

    def log_probs_to_answers(self, predictions, input_ids, **kwargs):
        """""
        1. get span start and end positions from log-probabilities
        2. extract actual tokens (answer) from input_ids
        """
        _, _, start_log_probs, end_log_probs = predictions
        passage_indices, start_indices, end_indices = get_best_spans(start_probs=np.exp(start_log_probs),
                                                                     end_probs=np.exp(end_log_probs),
                                                                     **kwargs)
        answers = []
        for i, (passage_index, start, end) in enumerate(zip(passage_indices, start_indices, end_indices)):
            answers.append(input_ids[i, passage_index, start: end])
        return self.tokenizer.batch_decode(answers, skip_special_tokens=True)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool = None,
        ignore_keys: list = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Same as Trainer.evaluation_loop but does not truncate output to the size of the dataset because
        there is M passages per question so the output is M times the size of the dataset

        Also gather input_ids instead of labels in order to recover the tokens from the model's span start and end probabilities
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        print(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            print(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/input_ids on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        input_ids_host = None
        passage_scores_host = None
        # losses/preds/input_ids on CPU (final containers)
        all_losses = None
        all_preds = None
        all_input_ids = None
        all_passage_scores = None
        all_answers = []

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            answer_strings = inputs.get('answer_strings')
            if answer_strings is not None:
                all_answers.extend(answer_strings)
            passage_score = inputs.get('passage_scores')
            if passage_score is not None:
                passage_scores_host = passage_score if passage_scores_host is None else torch.cat((passage_scores_host, passage_score), dim=0)

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, _ = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            input_ids = self._pad_across_processes(inputs['input_ids'])
            input_ids = self._nested_gather(input_ids)
            input_ids_host = input_ids if input_ids_host is None else nested_concat(input_ids_host, input_ids, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                input_ids = nested_numpify(input_ids_host)
                all_input_ids = (
                    input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
                )
                if passage_scores_host is not None:
                    passage_scores = nested_numpify(passage_scores_host)
                    all_passage_scores = passage_scores if all_passage_scores is None else nested_concat(all_passage_scores, passage_scores, padding_index=0)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, input_ids_host, passage_scores_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if input_ids_host is not None:
            input_ids = nested_numpify(input_ids_host)
            all_input_ids = input_ids if all_input_ids is None else nested_concat(all_input_ids, input_ids, padding_index=-100)
        if passage_scores_host is not None:
            passage_scores = nested_numpify(passage_scores_host)
            all_passage_scores = passage_scores if all_passage_scores is None else nested_concat(all_passage_scores, passage_scores, padding_index=0)

        # reshape like (N, M, L) to ease further processing
        if all_preds is not None:
            all_preds = tuple(pred.reshape(num_samples, self.M, -1) for pred in all_preds)
        if all_input_ids is not None:
            all_input_ids = all_input_ids.reshape(num_samples, self.M, -1)
        if all_passage_scores is not None:
            all_passage_scores = all_passage_scores.reshape(num_samples, self.M)
        if all_answers:
            all_answers = [all_answers[i] for i in range(0, len(all_answers), self.M)]
            assert len(all_answers) == num_samples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_input_ids is not None and all_answers:
            # 1. raw predictions from scores spans
            predictions = self.log_probs_to_answers(all_preds, all_input_ids)
            predictions, references = format_predictions_for_squad(predictions, all_answers)
            metrics = self.compute_metrics(predictions=predictions, references=references)
            # 2. weighted predictions
            if all_passage_scores is not None:
                weighted_predictions = self.log_probs_to_answers(all_preds, all_input_ids, weights=all_passage_scores)
                weighted_predictions, references = format_predictions_for_squad(weighted_predictions, all_answers)
                for k, v in self.compute_metrics(predictions=weighted_predictions, references=references).items():
                    metrics['weighted_'+k] = v
        else:
            metrics = {}
            predictions = all_preds

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=predictions, label_ids=None, metrics=metrics, num_samples=num_samples)
    

def main():
    cli = LightningCLI(
        trainer_class=pl.Trainer, 
        # same default as transformers although it is unlikely that the calls are in the exact same order
        seed_everything_default=42, 
        description='Main training script based on pytorch lightning.'
    )
    return cli
    
    
if __name__ == "__main__":
    main()