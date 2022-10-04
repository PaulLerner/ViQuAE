"""Script to embed dataset and Knowledge Base prior to search.

Usage: embedding.py <dataset> <config> [--disable_caching --kb=<path> --output=<path>]

Positional arguments:
    1. <dataset>   Path to the dataset  
    2. <config>    Path to the JSON configuration file (passed as kwargs)
    
Options:
    --disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
    --kb=<path>             Path to the KB that can be mapped from the passages
    --output=<path>         Optionally save the resulting dataset there instead of overwriting the input dataset.
"""

from docopt import docopt
import json

import torch

from datasets import load_from_disk, set_caching_enabled

from ..models.utils import device, prepare_inputs
from ..models.mm import MMConfig
from ..data.loading import load_pretrained_in_kwargs


def get_face_inputs(batch, n_faces=4, face_dim=512, bbox_dim=7):
    """
    Formats pre-computed face features in nice square tensors similarly to MMTrainer.get_face_inputs
        
    Parameters
    ----------
    batch: dict
    n_faces: int, optional
    face_dim: int, optional
    bbox_dim: int, optional
    
    Returns
    -------
    face_inputs: dict[str, Tensor]
        {
           * face: Tensor(batch_size, n_faces, face_dim)
           * bbox: Tensor(batch_size, n_faces, bbox_dim)
           * attention_mask: Tensor(batch_size, n_faces)
        }
    """
    face_list = batch["face_embedding"]
    batch_size = len(face_list)
    # trim or pad, and convert to tensor
    face_embeddings = torch.zeros((batch_size, n_faces, face_dim))
    face_boxes = torch.zeros((batch_size, n_faces, bbox_dim))
    # 0=masked, 1=not masked
    attention_mask = torch.zeros((batch_size, n_faces), dtype=torch.long)
    if n_faces == 0:
        return {
                "face": face_embeddings,
                "bbox": face_boxes,
                "attention_mask": attention_mask
            }

    for i, (face_embedding, bbox) in enumerate(zip(face_list, batch["face_box"])):
        # no face detected
        if face_embedding is None:
            # keep zero-padding/mask
            continue
        min_n = min(n_faces, len(face_embedding))
        face_embeddings[i, : min_n] = torch.tensor(face_embedding[: min_n])
        face_boxes[i, : min_n] = torch.tensor(bbox[: min_n])
        attention_mask[i, : min_n] = 1

    face_inputs = {
        "face": face_embeddings,
        "bbox": face_boxes,
        "attention_mask": attention_mask
    }
    return face_inputs


def get_image_inputs(batch, image_kwargs):    
    """
    Formats pre-computed full-image features in nice square tensors similarly to MMTrainer.get_image_inputs
    
    Parameters
    ----------
    batch: dict
    image_kwargs: dict
        keys are used to index batch to get precomputed features.
    
    Returns
    -------
    image_inputs: dict[str, dict[str,Tensor]]
        one key per image feature (the same as image_kwargs)
        {
           * input: Tensor(batch_size, ?)
           * attention_mask: Tensor(batch_size, )
             None of the images are masked
        }
        """
    image_inputs = {}
    for name, image_kwarg in image_kwargs.items():
        features = torch.tensor(batch[name])
        # 0=masked, 1=not masked
        attention_mask = torch.ones(features.shape[0], dtype=torch.long)
        image_inputs[name] = dict(input=features, attention_mask=attention_mask)
    return image_inputs  


def map_passage_to_kb(batch, kb, features):
    """
    Parameters
    ----------
    batch: dict
        Should be a batch from the passages KB
        Should be able to map to the KB using the 'index' key
    kb: Dataset
        Should be a dataset with pre-computed features
    features: List[str]
        each feature in features is used to index kb and is then added to the batch
    """
    subset = kb.select(batch['index'])
    for feature in features:
        batch.setdefault(feature, subset[feature])
    return batch

    
def get_inputs(batch, model, tokenizer, tokenization_kwargs={}, key='passage', kb=None):
    """
    Tokenizes input text and optionally gathers image features from the kb depending on model.
    
    Parameters
    ----------
    batch: dict
    model: nn.Module
        If it’s a DMREncoder or IntermediateLinearFusion instance, 
        will gather image features to take as input (from the kb if kb is not None)
    tokenizer: PreTrainedTokenizer
    tokenization_kwargs: dict, optional
        To be passed to tokenizer
    key: str, optional
        Used to index the batch to get the text
    kb: Dataset, optional
        Should hold image features and be mappable from batch['index']
    """
    text_inputs = tokenizer(batch[key], **tokenization_kwargs)
    model_config = getattr(model, "config", None)
    if model_config is not None and isinstance(model_config, MMConfig):
        if kb is not None:
            features = {"face_embedding", "face_box"} | model.config.image_kwargs.keys()
            # /!\ do not modify batch, copy before (else all the features of the KB will be saved). 
            # no need to deepcopy (only modifying batch keys)
            new_batch = map_passage_to_kb(batch.copy(), kb, features)
        else:
            new_batch = batch
        inputs = dict(
            text_inputs=text_inputs, 
            face_inputs=get_face_inputs(new_batch, model.config.n_faces, **model.config.face_kwargs), 
            image_inputs=get_image_inputs(new_batch, model.config.image_kwargs)
        )
    else:
        inputs = text_inputs
    return inputs


def embed(batch, model, tokenizer, tokenization_kwargs={}, key='passage', 
          save_as='text_embedding', output_key=None, forward_kwargs={}, layers=None, kb=None):
    """
    Parameters
    ----------
    batch, model, tokenizer, tokenization_kwargs, key, kb: 
        see ``get_inputs``
    save_as: str, optional
        key to save the resulting embedding in batch
    output_key: str or int, optional
        if model outputs a dict, list, or tuple, used to get THE output Tensor you want
    forward_kwargs: dict, optional
        passed to model.forward
    layers: list[int], optional
        if not None, expects that the output is a List[Tensor] 
        with each Tensor being shaped like (batch_size, sequence_length, hidden_size)
        In this case, it will save in {save_as}_layer_{layer} the representation of the first token (DMR-like), for each layer
    """
    inputs = get_inputs(batch, model, tokenizer, tokenization_kwargs=tokenization_kwargs, key=key, kb=kb)
    # move to device
    inputs = prepare_inputs(inputs)
    with torch.no_grad():
        outputs = model(**inputs, **forward_kwargs)
    # single output
    if isinstance(outputs, torch.Tensor):
        output = outputs
    # multiple outputs
    elif isinstance(outputs, (dict, list, tuple)):
        if output_key is None:
            raise ValueError(f"You should set output_key to choose from the model's outputs (got {output_key})")
        output = outputs[output_key]
    else:
        raise TypeError(f"Invalid type '{type(outputs)}' for model's outputs:\n{outputs}")
    if layers is None:
        batch[save_as] = output.cpu().numpy()
    # extract representation for each layer in layers
    # in this case, output_key should be 'hidden_states' or equivalent
    # i.e. output holds the representation of each token for each layer
    else:
        for layer in layers:
            # FIXME: ad-hoc for DPR: keep only the representation of the [CLS] token
            batch[f"{save_as}_layer_{layer}"] = output[layer][:, 0].cpu().numpy()
    return batch


def dataset_embed(dataset_path, map_kwargs={}, output_path=None, **fn_kwargs):
    """Loads dataset from path, maps it through embed, and saves it to output_path"""
    dataset = load_from_disk(dataset_path)
    # defaults to overwrite the dataset
    if output_path is None:
        output_path = dataset_path
    dataset = dataset.map(embed, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
    dataset.save_to_disk(output_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    if args['--kb']:
        kb = load_from_disk(args['--kb'])
    else:
        kb = None
    config_path = args['<config>']
    with open(config_path, 'rt') as file:
        config = load_pretrained_in_kwargs(json.load(file))

    default_tokenization_kwargs = dict(return_tensors='pt', padding='max_length', truncation=True)
    default_tokenization_kwargs.update(config['tokenization_kwargs'])
    config['tokenization_kwargs'] = default_tokenization_kwargs
    model = config.pop('model')
    model = model.to(device).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    dataset_embed(args['<dataset>'], model=model, kb=kb, output_path=args['--output'], **config)
