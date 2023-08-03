"""Utility functions specific to Question Answering."""
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import (
    BertForQuestionAnswering, ViltPreTrainedModel, ViltModel
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..train.optim import multi_passage_rc_loss
from .outputs import ReaderOutput
from .vilt import ViltEmbeddings, ViltPooler, ViltEncoder
from .mm import ECAEncoder


def get_best_spans(start_probs, end_probs, weights=None, cannot_be_first_token=True):
    """
    Get the best scoring spans from start and end probabilities

    notations:
        - N - number of distinct questions
        - M - number of passages per question in a batch
        - L - sequence length

    Parameters
    ----------
    start_probs, end_probs: Tensor
        shape (N, M, L)
    weights: Tensor, optional
        shape (N, M)
        Used to weigh the spans scores, e.g. might be BM25 scores from the retriever
    cannot_be_first_token: bool, optional
        (Default) null out the scores of start/end in the first token
        (e.g. "[CLS]", used during training for irrelevant passages)

    Returns
    -------
    passage_indices: Tensor
        shape (N, )
    start_indices, end_indices: Tensor
        shape (N, )
        start (inclusive) and end (exclusive) index of each span
    """
    N, M, L = start_probs.shape
    
    # 1. compute pairwise scores -> shape (N, M, L, L)
    pairwise = start_probs.reshape(N, M, L, 1) @ end_probs.reshape(N, M, 1, L)
    # fix scores where end < start
    pairwise = torch.triu(pairwise)
    # null out the scores of start in the first token (and thus end because of the upper triangle)
    # (e.g. [CLS], used during training for irrelevant passages)
    if cannot_be_first_token:
        pairwise[:, :, 0, :] = 0
    # eventually weigh the scores
    if weights is not None:
        minimum = weights.min()
        if minimum < 1:
            warnings.warn("weights should be > 1, adding 1-minimum")
            weights += 1-minimum
        pairwise *= weights.reshape(N, M, 1, 1)

    # 2. find the passages with the maximum score
    pairwise = pairwise.reshape(N, M, L * L)
    max_per_passage = pairwise.max(axis=2).values
    passage_indices = max_per_passage.argmax(axis=1)
    pairwise_best_passages = pairwise[torch.arange(N), passage_indices]

    # 3. finally find the best spans for each question
    flat_argmaxes = pairwise_best_passages.argmax(axis=-1)
    # convert from flat argmax to line index (start) and column index (end)
    start_indices = torch.div(flat_argmaxes, L, rounding_mode='floor')
    # add +1 to make end index exclusive so the spans can easily be used with slices
    end_indices = (flat_argmaxes % L) + 1

    return passage_indices, start_indices, end_indices


class MultiPassageBERT(BertForQuestionAnswering):
    """
    PyTorch/Transformers implementation of Multi-passage BERT [1]_ (based on the global normalization [2]_)
    i.e. groups passages per question before computing the softmax (and the NLL loss)
    so that spans scores are comparable across passages

    Code based on transformers.BertForQuestionAnswering, dpr.models.Reader
    and https://github.com/allenai/document-qa/blob/master/docqa/nn/span_prediction.py

    References
    ----------
    .. [1] Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 
       2019. Multi-passage BERT: A Globally Normalized BERT Model for Open-domain Question Answering. 
       In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing 
       and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 
       pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.

    .. [2] Christopher Clark and Matt Gardner. 2018. Simple and Effective Multi-Paragraph Reading Comprehension. 
       In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 
       pages 845–855, Melbourne, Australia. Association for Computational Linguistics.
    """
    def __init__(self, *args, fuse_ir_score=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_ir_score = fuse_ir_score
        if fuse_ir_score:
            # easier than overriding Bert wieght initialization
            self.score_proj_w = nn.Parameter(torch.ones((1,1)))
            self.score_proj_b = nn.Parameter(torch.zeros(1))
            self.weights_to_log = {
                "score_proj_w": self.score_proj_w, 
                "score_proj_b": self.score_proj_b
            }
        
    def forward(self,
                input_ids, passage_scores=None,
                start_positions=None, end_positions=None, answer_mask=None,
                return_dict=None, **kwargs):
        """
        notations: 
           * N - number of distinct questions
           * M - number of passages per question in a batch
           * L - sequence length

        Parameters
        ----------
        input_ids: Tensor[int]
            shape (N * M, L)
            There should always be a constant number of passages (relevant or not) per question
        passage_scores: FloatTensor, optional
            shape (N * M, )
            If self.fuse_ir_score, will be fused with start_logits and end_logits before computing loss
        start_positions, end_positions: Tensor[int], optional
            shape (N, M, max_n_answers)
            The answer might be found several time in the same passage, maximum ``max_n_answers`` times
            Defaults to None (i.e. don’t compute the loss)
        answer_mask: Tensor[int], optional
            shape (N, M, max_n_answers)
            Used to mask the loss for answers that are not ``max_n_answers`` times in the passage
            Required if start_positions and end_positions are specified
        **kwargs: additional arguments are passed to BERT after being reshape like 
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(input_ids, return_dict=True, **kwargs)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        if self.fuse_ir_score:
            passage_scores = passage_scores.unsqueeze(1) @ self.score_proj_w + self.score_proj_b
            start_logits += passage_scores
            end_logits += passage_scores
        
        # compute loss
        if start_positions is not None and end_positions is not None:
            pack = multi_passage_rc_loss(
                input_ids, 
                start_positions, 
                end_positions, 
                start_logits, 
                end_logits, 
                answer_mask
            )
            # unpack so that the line is not hundreds columns long
            total_loss, start_positions, end_positions, start_logits, end_logits, start_log_probs, end_log_probs = pack
        else:            
            total_loss, start_log_probs, end_log_probs = None, None, None

        if not return_dict:
            output = (start_logits, end_logits, start_log_probs, end_log_probs) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ReaderOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            start_log_probs=start_log_probs,
            end_log_probs=end_log_probs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
class MultiPassageECA(ECAEncoder):
    """Like MultiPassageBERT with a ECA backbone instead of BERT"""
    def __init__(self, config, **kwargs):
        assert not config.no_text, "no_text option is only for IR"
        super().__init__(config, **kwargs)
        self.fuse_ir_score = False
        
        # like BertForQuestionAnswering
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(self, text_inputs, *args,
                start_positions=None, end_positions=None, answer_mask=None,
                return_dict=True, **kwargs):
        input_ids = text_inputs['input_ids']
        outputs = super().forward(text_inputs, *args, return_dict=return_dict, **kwargs)
        
        # truncate to keep only text representations
        # the answer is extracted from text and the sequence length must match start/end positions shape (L)
        sequence_output = outputs.last_hidden_state[:, :input_ids.shape[1]]
        
        # same as MultiPassageBERT
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # compute loss
        if start_positions is not None and end_positions is not None:
            pack = multi_passage_rc_loss(
                input_ids, 
                start_positions, 
                end_positions, 
                start_logits, 
                end_logits, 
                answer_mask
            )
            # unpack so that the line is not hundreds columns long
            total_loss, start_positions, end_positions, start_logits, end_logits, start_log_probs, end_log_probs = pack
        else:            
            total_loss, start_log_probs, end_log_probs = None, None, None

        if not return_dict:
            output = (start_logits, end_logits, start_log_probs, end_log_probs) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ReaderOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            start_log_probs=start_log_probs,
            end_log_probs=end_log_probs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
class ViltMultiImageEmbeddings(ViltEmbeddings):
    """
    Similar to the 'triplet' strategy of UNITER, 
    patches of multiple images are concatenated in the sequence dimension.
    The resulting embedding thus have a sequence length of #tokens + num_patches*num_images
    """    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        passage_pixel_values,
        passage_pixel_mask,
        inputs_embeds
    ):
        """
        Parameters
        ----------
        input_ids (`torch.LongTensor` of shape `(batch_size, #tokens)`):
            Indices of input sequence tokens in the vocabulary. Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. 
    
        attention_mask (`torch.FloatTensor` of shape `(batch_size, #tokens})`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
        token_type_ids (`torch.LongTensor` of shape `(batch_size, #tokens)`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
    
        pixel_values, passage_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViltFeatureExtractor`]. See
            [`ViltFeatureExtractor.__call__`] for details.
    
        pixel_mask, passage_pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
    
            - 1 for pixels that are real (i.e. **not masked**),
            - 0 for pixels that are padding (i.e. **masked**).
        
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, #tokens, hidden_size)`):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        """
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
    
        # PART 2: patch embeddings (with interpolated position encodings)
        image_embeds, image_masks, patch_index = self.visual_embed(
            pixel_values, pixel_mask, max_image_length=self.config.max_image_length
        )
        passage_image_embeds, passage_image_masks, _ = self.visual_embed(
            passage_pixel_values, passage_pixel_mask, max_image_length=self.config.max_image_length
        )
    
        # PART 3: add modality type embeddings
        # 0 indicates text, 1 question image, 2 passage image
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, 1, dtype=torch.long, device=text_embeds.device)
        )
        passage_image_embeds = passage_image_embeds + self.token_type_embeddings(
            torch.full_like(passage_image_masks, 2, dtype=torch.long, device=text_embeds.device)
        )
    
        # PART 4: concatenate
        embeddings = torch.cat([text_embeds, image_embeds, passage_image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks, passage_image_masks], dim=1)
    
        return embeddings, masks


class ViltMultiImageModel(ViltModel):
    """Same as ViltModel with ViltMultiImageEmbeddings instead of ViltEmbeddings"""
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = ViltMultiImageEmbeddings(config)
        self.encoder = ViltEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        passage_pixel_values: Optional[torch.FloatTensor] = None,
        passage_pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        text_batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((text_batch_size, seq_length)), device=device)

        if pixel_values is not None and image_embeds is not None:
            raise ValueError("You cannot specify both pixel_values and image_embeds at the same time")
        elif pixel_values is None and image_embeds is None:
            raise ValueError("You have to specify either pixel_values or image_embeds")

        image_batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeds.shape[0]
        if image_batch_size != text_batch_size:
            raise ValueError("The text inputs and image inputs need to have the same batch size")
        if pixel_mask is None:
            pixel_mask = torch.ones((image_batch_size, self.config.image_size, self.config.image_size), device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, attention_mask = self.embeddings(
            input_ids,
            attention_mask,
            token_type_ids,
            pixel_values,
            pixel_mask,
            passage_pixel_values,
            passage_pixel_mask,
            inputs_embeds
        )

        # broadcast attention mask to all heads. N. B input_shape is only used for decoder models
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# TODO: alternatively, subclass ViltForImagesAndTextClassification and feed it two text-image pairs
class MultiPassageVilt(ViltPreTrainedModel):
    """Like MultiPassageBERT with a ViLT backbone instead of BERT"""
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.vilt = ViltMultiImageModel(config, add_pooling_layer=add_pooling_layer)
        self.fuse_ir_score = False
        
        # like BertForQuestionAnswering
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(self, input_ids, *args,
                start_positions=None, end_positions=None, answer_mask=None,
                return_dict=True, **kwargs):
        outputs = self.vilt(input_ids, *args, return_dict=return_dict, **kwargs)
        
        sequence_output = outputs[0]
        # truncate to keep only text representations
        # the answer is extracted from text and the sequence length must match start/end positions shape (L)
        sequence_output = sequence_output[:, :input_ids.shape[1]]
        
        # same as MultiPassageBERT
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # compute loss
        if start_positions is not None and end_positions is not None:
            pack = multi_passage_rc_loss(
                input_ids, 
                start_positions, 
                end_positions, 
                start_logits, 
                end_logits, 
                answer_mask
            )
            # unpack so that the line is not hundreds columns long
            total_loss, start_positions, end_positions, start_logits, end_logits, start_log_probs, end_log_probs = pack
        else:            
            total_loss, start_log_probs, end_log_probs = None, None, None

        if not return_dict:
            output = (start_logits, end_logits, start_log_probs, end_log_probs) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ReaderOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            start_log_probs=start_log_probs,
            end_log_probs=end_log_probs,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        
