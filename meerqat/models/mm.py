"""Implements the two main architectures presented in the ECIR-submitted paper."""
import warnings

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import (
    PreTrainedModel, BertModel, DPRQuestionEncoder, DPRContextEncoder, 
    ViltPreTrainedModel, ViltModel, CLIPModel, CLIPConfig
)
from transformers.models.bert import BertConfig, BertPreTrainedModel

from .outputs import EncoderOutput, ECAEncoderOutput
from .image import ImageEmbedding, FaceEmbedding
from .utils import TanhGate
from .bert import BertAttention, BertEmbeddings, BertIntermediate, BertOutput, BertPooler, BertLayer


class MMConfig(BertConfig):
    """
    Base configuration class for multimodal models based on BertConfig.
    
    Parameters
    ----------
    *args, **kwargs: 
        additionnal arguments are passed to BertConfig.
    n_images: int, optional
        Number of images to embed alongside with text. 
        Each image can be mapped to multiple face features or image features.
        If greater than 1, will be assigned to a type embedding (analog to BERT).
    n_faces: int, optional
        Number of faces that the multimodal model should take as input. Defaults to 4.
    face_kwargs: dict, optional
        Keyword arguments used for the FaceEmbedding module.
        Defaults to dict(face_dim=512, bbox_dim=7).
    image_kwargs: dict, optional
        Keyword arguments used for as many ImageEmbedding modules (one per key).
        Defaults to {
            "clip-RN50": {"input_dim": 1024},
            "imagenet-RN50": {"input_dim": 2048}
        }
    face_and_image_are_exclusive: bool, optional
        Whether face and full-image representation should be combined (default) or exclusive.
        Handled with attention masks in transformers
    no_text: bool, optional
        Whether to rely only on faces and images. 
        In this case, only the [CLS] token embedding is concatenated to the image features.
        Defaults to False.
    gating: bool, optional
        Whether to use flamingo-style tanh gating (init at 0) [2]_
        Defaults to no gating        
        
     References
    ----------
    .. [2] Jean-Baptiste Alayrac et al. (2022). 
       Flamingo: a Visual Language Model for Few-Shot Learning. ArXiv:2204.14198.
    """
    def __init__(
            self,
             *args,
             n_images=1,
             n_faces=4,
             face_kwargs=None,
             image_kwargs=None,
             face_and_image_are_exclusive=False,
             no_text=False,
             gating=False,
             **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.n_images = n_images
        self.n_faces = n_faces
        if face_kwargs is None:
            self.face_kwargs = dict(face_dim=512, bbox_dim=7)
        else:
            self.face_kwargs = face_kwargs
        if image_kwargs is None:
            self.image_kwargs = {
                "clip-RN50": {"input_dim": 1024},
                "imagenet-RN50": {"input_dim": 2048}
            }
        else:
            self.image_kwargs = image_kwargs
        self.face_and_image_are_exclusive = face_and_image_are_exclusive
        self.no_text = no_text
        self.gating = gating


class FlamantConfig(MMConfig):
    """
    Hyperparameters for multimodal cross-attention layers
    
    Same defaults as BertConfig.
    """
    def __init__(self,
                 *args,
                 multimodal_attention_every=1,
                 image_num_attention_heads=12,
                 image_intermediate_size=3072,
                 image_hidden_dropout_prob=0.1,
                 image_attention_probs_dropout_prob=0.1,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.multimodal_attention_every = multimodal_attention_every
        self.image_num_attention_heads = image_num_attention_heads
        self.image_intermediate_size = image_intermediate_size
        self.image_hidden_dropout_prob = image_hidden_dropout_prob
        self.image_attention_probs_dropout_prob = image_attention_probs_dropout_prob
        
        
def overwrite_bert_config(flamant_config):
    """
    Overwrite BERT parameters in the input flamant_config if they start with "image_".
    See usage in FlamantLayer.
    
    Parameters
    ----------
    flamant_config: FlamantConfig
    
    Returns
    -------
    bert_config: BertConfig
    """
    config_dict = flamant_config.to_dict()
    for k in list(config_dict.keys()):
        if k.startswith("image_"):
            # overwrite BERT parameter with the image version of Flamant
            config_dict[k[len("image_"):]] = config_dict.pop(k)
            
    return BertConfig.from_dict(config_dict)


class FlamantLayer(nn.Module):
    """Adapted from transformers.BertLayer"""
    def __init__(self, config):
        super().__init__()
        if config.chunk_size_feed_forward != 0:
            raise NotImplementedError()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.image_crossattention = BertAttention(overwrite_bert_config(config), position_embedding_type="absolute")
        # like BertIntermediate + BertOutput without residual connection and layer-norm
        # which must happen after gating
        self.image_ffw = nn.Sequential(
            nn.Linear(config.hidden_size, config.image_intermediate_size),
            # FIXME: does not take into account config.hidden_act
            # (because transformers.activations.ACT2FN returns a function and not a Module)
            # Also: Squared-ReLU is used in Flamingo
            nn.GELU(), 
            nn.Linear(config.image_intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        if config.gating:
            self.attn_gate, self.ffw_gate = TanhGate(), TanhGate()
        else:
            self.attn_gate, self.ffw_gate = nn.Identity(), nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,        
        image_embeddings: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False
    ) -> Tuple[torch.Tensor]:
        if past_key_value is not None or output_attentions:            
            raise NotImplementedError()
            
        # Flamingo-style gated cross-attention
        hidden_states = self.attn_gate(
            self.image_crossattention(
                hidden_states, # query
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=image_embeddings, # key and value
                encoder_attention_mask=image_attention_mask,
                past_key_value=None,
                output_attentions=False
            )[0]
        ) + hidden_states
        hidden_states = self.ffw_gate(self.image_ffw(hidden_states)) + hidden_states
        # tough architectural choice: keep BERT-style post layer-norm
        # but it goes against the flamingo spirit of 
        # "output should be the same as the pretrained language model after init"
        hidden_states = self.LayerNorm(hidden_states)
        
        # ========================== #
        # Below: standard BERT layer #
        # ========================== #      
            
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            raise NotImplementedError()
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            raise NotImplementedError()

        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
    
class FlamantEncoder(nn.Module):
    """Like BertEncoder but with FlamantLayer instead of BertLayer every n layers"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if i % config.multimodal_attention_every == 0:
                self.layer.append(FlamantLayer(config))
            else:
                self.layer.append(BertLayer(config))
                
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_embeddings: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        if use_cache:
            raise NotImplementedError()

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            inputs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )

            # feed image embeddings for multimodal cross-attention
            if isinstance(layer_module, FlamantLayer):
                inputs = (
                    hidden_states,        
                    image_embeddings,
                    attention_mask,
                    image_attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            # standard BERT inputs
            else:
                inputs = (
                    hidden_states,        
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), *inputs
                )
            else:
                layer_outputs = layer_module(*inputs)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
    
class FlamantModel(BertPreTrainedModel):
    """
    Fuses modalities with gated cross-attention layers like in Flamingo [2]_
    Adapted from transformers.BertModel
    """    
    config_class = FlamantConfig
    load_tf_weights = None
    
    def __init__(self, config, add_pooling_layer=False):
        if config.n_images > 1:
            raise NotImplementedError()
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = FlamantEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        if self.config.n_faces > 0:
            self.face_embedding = FaceEmbedding(embedding_dim=self.config.hidden_size,
                                                dropout=self.config.hidden_dropout_prob,
                                                layer_norm_eps=self.config.layer_norm_eps,
                                                **self.config.face_kwargs)
        else:
            self.face_embedding = None
        self.image_embeddings, self.image_gates = nn.ModuleDict(), nn.ModuleDict()
        for name, image_kwarg in self.config.image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=self.config.hidden_size,
                                                         dropout=self.config.hidden_dropout_prob,
                                                         **image_kwarg)
        self.weights_to_log = {}
        # add pointers to the gate parameters so that they are logged in trainer
        if self.config.gating:
            for i, layer_module in enumerate(self.encoder.layer):
                if isinstance(layer_module, FlamantLayer):
                    self.weights_to_log[f"attn_gate_{i}"] = layer_module.attn_gate.gate_param
                    self.weights_to_log[f"ffw_gate_{i}"] = layer_module.ffw_gate.gate_param
                    
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, text_inputs, face_inputs, image_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        """
        Arguments
        ---------
        text_inputs: dict[str, torch.LongTensor]
            usual BERT inputs, see transformers.BertModel
        face_inputs: dict[str, torch.FloatTensor]
            {
                "face": (batch_size, n_faces, face_dim),
                "bbox": (batch_size, n_faces, bbox_dim),
                "attention_mask": (batch_size, n_faces)
            }
        image_inputs: dict[str, dict[str, torch.FloatTensor]]
            {
                model:
                {
                    "input": (batch_size, image_dim)
                    "attention_mask": (batch_size, )
                }
            }
        """
        # reshape faces
        faces = face_inputs['face']
        batch_size, n_faces, face_dim = faces.shape
        if n_faces > 0:
            faces = faces.reshape(batch_size * n_faces, face_dim)
            # embed batch of size batch_size*n_faces
            face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size * n_faces, -1))
            face_output = face_output.reshape(batch_size, n_faces, -1)
        else:
            face_output = torch.zeros(batch_size, 0, self.config.hidden_size, device=faces.device)

        # embed images
        if image_inputs:
            image_outputs, image_attention_mask = [], []
            for name, image in image_inputs.items():
                image_output = self.image_embeddings[name](image['input']).unsqueeze(0)
                image_outputs.append(image_output)
                image_attention_mask.append(image['attention_mask'].unsqueeze(0))
            # (n_models, batch_size, embedding_dim) -> (batch_size, n_models, embedding_dim)
            image_outputs = torch.cat(image_outputs, 0).transpose(0, 1)
            image_attention_mask = torch.cat(image_attention_mask, 0).transpose(0, 1)
        else:
            image_outputs = torch.zeros(batch_size, 0, self.config.hidden_size, device=faces.device)
            image_attention_mask = torch.zeros(batch_size, 0, device=faces.device)
        
        if self.config.face_and_image_are_exclusive:
            face_attention_mask = face_inputs["attention_mask"]
            # indices at the batch level: at least one face detected (i.e. not masked)
            where_are_faces = face_attention_mask.nonzero()[:,0].unique()
            # mask images if at least one face was detected
            image_attention_mask[where_are_faces] = 0

        if self.config.no_text:
            raise NotImplementedError()
                
        # embed text: (batch_size, sequence_length, embedding_dim)
        token_type_ids = text_inputs.get('token_type_ids')
        text_embeddings = self.embeddings(input_ids=text_inputs['input_ids'],
                                          token_type_ids=token_type_ids)
        attention_mask = self.get_extended_attention_mask(
            text_inputs['attention_mask'], text_embeddings.shape[:-1], text_embeddings.device)

        # (batch_size, n_faces+n_models, embedding_dim)
        image_embeddings = torch.cat((face_output, image_outputs), dim=1)
        image_attention_mask = torch.cat((face_inputs['attention_mask'], image_attention_mask), dim=1)
        # N. B. looks like this produces the same output as get_extended_attention_mask
        # I stick to what is in BertModel implementation
        image_attention_mask = self.invert_attention_mask(image_attention_mask)
        outputs = self.encoder(
            text_embeddings, image_embeddings, attention_mask=attention_mask,
            image_attention_mask=image_attention_mask, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)

        # same as DPR: extract representation from [CLS]: the first token
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        
        if not return_dict:
            return (pooled_output, ) + outputs[2:]
        
        return ECAEncoderOutput(
                pooler_output=pooled_output,
                hidden_states=outputs.hidden_states, 
                attentions=outputs.attentions)


class ViltForIR(ViltPreTrainedModel):
    """
    Pools ViLT using the representation of the [CLS] token, 
    i.e. DPR-style, *not* with ViltPooler (ITM pre-trained layer),
    except if add_pooling_layer=True
    """
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)
        self.vilt = ViltModel(config, add_pooling_layer=add_pooling_layer)
        # N. B. post_init is called in ViltModel
    
    def forward(self, *args, return_dict=True, **kwargs):
        outputs = self.vilt(*args, return_dict=return_dict, **kwargs)
        # default behavior: pooling from [CLS] instead of ViltPooler (ITM pre-trained layer)
        if outputs.pooler_output is None:
            outputs.pooler_output = outputs.last_hidden_state[:, 0]
        # else keep pooling from ViltPooler
        return outputs


class CLIPForIR(PreTrainedModel):
    """
    Fuses image and text embeddings simply by summing them to be compatible with BiEncoder.
    
    Because BiEncoder uses dot-product similarity, note that this will be equivalent to computing:
        i_q*i_p + i_q*t_p + t_q*t_p + t_q*i_p
    Where i, t stand for image, text and _q and _p suffixes stand for question and passage (or context)
    i.e. computing all mono-modal and cross-modal similarities.
    
    But it might be worth using another trainee than BiEncoder to be able to scale these similarities.
    """
    config_class = CLIPConfig
    base_model_prefix = "clip"

    def __init__(self, config):
        super().__init__(config)
        self.clip = CLIPModel(config)
        # N. B. post_init is called in CLIPModel
    
    def forward(self, *args, return_dict=True, return_loss=False, **kwargs):
        outputs = self.clip(*args, return_dict=return_dict, return_loss=return_loss, **kwargs)
        multimodal_output = outputs.text_embeds + outputs.image_embeds
        return EncoderOutput(pooler_output=multimodal_output)
        
        
class ECAEncoder(PreTrainedModel):
    """
    Text and image are fused by concatenating them at the sequence-level then feeding them to BERT, à la UNITER [1]_
        - one face ≃ one token  
        - one image ≃ one token

    The multimodal representation is obtained from the "[CLS]" token.
    
    When using gating (see MMConfig), it is done before the attention layer, unlike in Flamingo [2]_

    References
    ----------
    .. [1] Chen, Y.C., Li, L., Yu, L., El Kholy, A., Ahmed, F., Gan, Z., Cheng, Y., Liu, J.:
        Uniter: Universal image-text representation learning. In: European Conference on
        Computer Vision. pp. 104–120. https://openreview.net/forum?id=S1eL4kBYwr. Springer (2020)
    """
    config_class = MMConfig
    load_tf_weights = None
    base_model_prefix = "bert_model"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert_model = BertModel(config, add_pooling_layer=False)
        # add pointers to the gate parameters so that they are logged in trainer
        self.weights_to_log = {}
        
        if self.config.n_images > 1:
            self.image_type_embeddings = nn.Embedding(self.config.n_images, self.config.hidden_size)
            image_layer_norm = self.config.layer_norm_eps
        else:
            image_layer_norm = None
            
        if self.config.n_faces > 0:
            self.face_embedding = FaceEmbedding(embedding_dim=self.config.hidden_size,
                                                dropout=self.config.hidden_dropout_prob,
                                                layer_norm_eps=self.config.layer_norm_eps,
                                                **self.config.face_kwargs)
            if self.config.gating:
                self.face_gate = TanhGate()
                self.weights_to_log["face_gate"] = self.face_gate.gate_param
            else:
                self.face_gate = nn.Identity()
        else:
            self.face_embedding = None
        self.image_embeddings, self.image_gates = nn.ModuleDict(), nn.ModuleDict()
        for name, image_kwarg in self.config.image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=self.config.hidden_size,
                                                         dropout=self.config.hidden_dropout_prob,
                                                         layer_norm_eps=image_layer_norm,
                                                         **image_kwarg)
            if self.config.gating:
                self.image_gates[name] = TanhGate()
                self.weights_to_log[f"{name}_gate"] = self.image_gates[name].gate_param
            else:
                self.image_gates[name] = nn.Identity()
                                
    def _init_weights(self, module):
        # same as BERT
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # keep torch defaults for linear layers
    
    def forward(self, text_inputs, face_inputs, image_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        """
        Arguments
        ---------
        text_inputs: dict[str, torch.LongTensor]
            usual BERT inputs, see transformers.BertModel
        face_inputs: dict[str, torch.FloatTensor]
            {
                "face": (batch_size, n_images, n_faces, face_dim),
                "bbox": (batch_size, n_images, n_faces, bbox_dim),
                "attention_mask": (batch_size, n_images, n_faces)
            }
        image_inputs: dict[str, dict[str, torch.FloatTensor]]
            {
                model:
                {
                    "input": (batch_size, n_images, image_dim)
                    "attention_mask": (batch_size, n_images)
                }
            }
        """
        # reshape faces
        faces = face_inputs['face']
        batch_size, n_images, n_faces, face_dim = faces.shape
        assert n_images == self.config.n_images
        if n_faces > 0:
            if n_images > 1:
                image_type_ids = torch.zeros((batch_size, n_images, n_faces), dtype=torch.long, device=faces.device)
                # broadcast arange to the right shape
                image_type_ids += torch.arange(n_images, dtype=torch.long, device=faces.device).reshape(1, n_images, 1)
                image_type_embeddings = self.image_type_embeddings(image_type_ids.reshape(batch_size*n_images*n_faces))
            else:
                image_type_embeddings = None
            faces = faces.reshape(batch_size*n_images*n_faces, face_dim)
            bbox=face_inputs['bbox'].reshape(batch_size*n_images*n_faces, -1)
            face_output = self.face_embedding(face=faces, bbox=bbox, image_type_embeddings=image_type_embeddings)
            face_output = face_output.reshape(batch_size, n_images*n_faces, -1)
            # maybe gate faces
            face_output = self.face_gate(face_output)
        else:
            face_output = torch.zeros(batch_size, 0, self.config.hidden_size, device=faces.device)

        # embed images
        if image_inputs:
            if n_images > 1:
                image_type_ids = torch.zeros((batch_size, n_images), dtype=torch.long, device=faces.device)
                image_type_ids += torch.arange(n_images, dtype=torch.long, device=faces.device)
                image_type_embeddings = self.image_type_embeddings(image_type_ids.reshape(batch_size*n_images))
            else:
                image_type_embeddings = None

            image_outputs, image_attention_mask = [], []
            for name, image in image_inputs.items():
                image_output = self.image_embeddings[name](
                    image['input'].reshape(batch_size*n_images), 
                    image_type_embeddings=image_type_embeddings
                )
                # maybe gate image
                image_output = self.image_gates[name](image_output)
                image_outputs.append(image_output.reshape(1, batch_size, n_images, -1))
                image_attention_mask.append(image['attention_mask'].unsqueeze(0))
            # (n_models, batch_size, n_images, embedding_dim) -> (batch_size, n_images*n_models, embedding_dim)
            image_outputs = torch.cat(image_outputs, 0).transpose(0, 1).reshape(batch_size, len(image_inputs)*n_images, -1)
            image_attention_mask = torch.cat(image_attention_mask, 0).transpose(0, 1).reshape(batch_size, len(image_inputs)*n_images)
        else:
            image_outputs = torch.zeros(batch_size, 0, self.config.hidden_size, device=faces.device)
            image_attention_mask = torch.zeros(batch_size, 0, device=faces.device)
        
        if self.config.face_and_image_are_exclusive:
            face_attention_mask = face_inputs["attention_mask"]
            # indices at the batch level: at least one face detected (i.e. not masked)
            where_are_faces = face_attention_mask.nonzero()[:,0].unique()
            # mask images if at least one face was detected
            image_attention_mask[where_are_faces] = 0

        token_type_ids = text_inputs.get('token_type_ids')
        # keep only keep [CLS] token
        if self.config.no_text:
            text_inputs['input_ids'] = text_inputs['input_ids'][:, :1]
            text_inputs['attention_mask'] = text_inputs['attention_mask'][:, :1]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :1]
                
        # embed text: (batch_size, sequence_length, embedding_dim)
        text_embeddings = self.bert_model.embeddings(input_ids=text_inputs['input_ids'],
                                                     token_type_ids=token_type_ids)

        # (batch_size, sequence_length+(n_faces+n_models)*n_images, embedding_dim)
        multimodal_embeddings = torch.cat((text_embeddings, face_output, image_outputs), dim=1)
        attention_mask = torch.cat((text_inputs['attention_mask'], face_inputs['attention_mask'], image_attention_mask), dim=1)
        extended_attention_mask = self.bert_model.get_extended_attention_mask(
            attention_mask, multimodal_embeddings.shape[:-1], multimodal_embeddings.device
        )
        outputs = self.bert_model.encoder(multimodal_embeddings, attention_mask=extended_attention_mask,
                                          output_attentions=output_attentions,
                                          output_hidden_states=output_hidden_states,
                                          return_dict=return_dict)

        # same as DPR: extract representation from [CLS]: the first token
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        
        if not return_dict:
            return (pooled_output, ) + outputs[2:]
        
        return ECAEncoderOutput(
                pooler_output=pooled_output,
                hidden_states=outputs.hidden_states, 
                attentions=outputs.attentions)


class ILFConfig(MMConfig):
    """
    Same as MMConfig with an extra parameter: 
    question_encoder: bool, optional
        Whether to use DPRQuestionEncoder (default) or DPRContextEncoder.
        This makes no real differences in the architecture, only the name changes.
    """
    def __init__(self,
                 *args,
                 question_encoder=True,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.question_encoder = question_encoder
         
            
class IntermediateLinearFusion(PreTrainedModel):
    """Fuses DPR’s text representation with image embeddings by projecting them linearly in the same space"""
    config_class = ILFConfig
    load_tf_weights = None
    base_model_prefix = "dpr_encoder"

    def __init__(self, config):
        if config.n_images > 1:
            raise NotImplementedError()
        super().__init__(config)
        self.config = config
        if self.config.question_encoder:
            self.dpr_encoder = DPRQuestionEncoder(config)
        else:
            self.dpr_encoder = DPRContextEncoder(config)
        if self.config.n_faces > 0:
            self.face_embedding = FaceEmbedding(embedding_dim=self.config.hidden_size, dropout=self.config.hidden_dropout_prob,
                                                layer_norm_eps=self.config.layer_norm_eps, **self.config.face_kwargs)
        else:
            self.face_embedding = None
        self.image_embeddings = nn.ModuleDict()
        for name, image_kwarg in self.config.image_kwargs.items():
            self.image_embeddings[name] = ImageEmbedding(embedding_dim=self.config.hidden_size, dropout=self.config.hidden_dropout_prob, **image_kwarg)
        self.dpr_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
    def _init_weights(self, module):
        # same as BERT
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # keep torch defaults for linear layers
    
    def forward(self, text_inputs, face_inputs, image_inputs):
        """
        Arguments
        ---------
        text_inputs: dict[str, torch.LongTensor]
            usual BERT inputs, see transformers.DPRQuestionEncoder
        face_inputs: dict[str, torch.FloatTensor]
            {
                "face": (batch_size, n_faces, face_dim),
                "bbox": (batch_size, n_faces, bbox_dim),
                "attention_mask": (batch_size, n_faces)
            }
        image_inputs: dict[str, dict[str, torch.FloatTensor]]
            {
                model:
                {
                    "input": (batch_size, image_dim)
                    "attention_mask": (batch_size, )
                }
            }
        """
        # embed text
        output = self.dpr_encoder(**text_inputs).pooler_output
        output = self.dpr_proj(output)
        
        # reshape faces
        faces = face_inputs['face']
        batch_size, n_faces, face_dim = faces.shape
        if n_faces > 0:
            faces = faces.reshape(batch_size * n_faces, face_dim)
            # embed batch of size batch_size*n_faces
            face_output = self.face_embedding(face=faces, bbox=face_inputs['bbox'].reshape(batch_size * n_faces, -1))
            face_output = face_output.reshape(batch_size, n_faces, -1)
            # sum over all faces
            face_output = face_output.sum(axis=1)
            
            # fuse text and faces
            output += face_output

        # fuse text and image
        if self.config.face_and_image_are_exclusive:
            face_attention_mask = face_inputs["attention_mask"]
            # indices at the batch level: at least one face detected (i.e. not masked)
            where_are_faces = face_attention_mask.nonzero()[:,0].unique()
        for name, image in image_inputs.items():
            # mask images if at least one face was detected
            if self.config.face_and_image_are_exclusive:
                image['input'][where_are_faces] = 0
            output += self.image_embeddings[name](image['input'])
        output = self.LayerNorm(output)
        output = self.dropout(output)
        return EncoderOutput(pooler_output=output)
