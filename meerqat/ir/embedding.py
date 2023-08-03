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

from datasets import load_from_disk, set_caching_enabled, DatasetDict

from ranx import Run

from ..models.utils import device, prepare_inputs
from ..models.mm import MMConfig
from ..data.loading import load_pretrained_in_kwargs


def get_face_inputs(batch, n_faces=4, face_dim=512, bbox_dim=7):
    """
    Formats pre-computed face features in nice square tensors similarly to PreComputedImageFeatures.get_face_inputs
        
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
           * face: Tensor(batch_size, 1, n_faces, face_dim)
           * bbox: Tensor(batch_size, 1,  n_faces, bbox_dim)
           * attention_mask: Tensor(batch_size, 1, n_faces)
        }
    """
    face_list = batch["face_embedding"]
    batch_size = len(face_list)
    # trim or pad, and convert to tensor
    face_embeddings = torch.zeros((batch_size, 1, n_faces, face_dim))
    face_boxes = torch.zeros((batch_size, 1, n_faces, bbox_dim))
    # 0=masked, 1=not masked
    attention_mask = torch.zeros((batch_size, 1, n_faces), dtype=torch.long)
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
        face_embeddings[i, 0, : min_n] = torch.tensor(face_embedding[: min_n])
        face_boxes[i, 0, : min_n] = torch.tensor(bbox[: min_n])
        attention_mask[i, 0, : min_n] = 1

    face_inputs = {
        "face": face_embeddings,
        "bbox": face_boxes,
        "attention_mask": attention_mask
    }
    return face_inputs


def get_image_inputs(batch, image_kwargs):    
    """
    Formats pre-computed full-image features in nice square tensors similarly to PreComputedImageFeatures.get_image_inputs
    
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
           * input: Tensor(batch_size, 1, ?)
           * attention_mask: Tensor(batch_size, 1)
             None of the images are masked
        }
        """
    image_inputs = {}
    for name, image_kwarg in image_kwargs.items():
        features = torch.tensor(batch[name]).unsqueeze(1)
        # 0=masked, 1=not masked
        attention_mask = torch.ones((features.shape[0], 1), dtype=torch.long)
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

    
def expand_query(batch, key='passage', kb=None, run=None, tokenizer=None, 
                 qe_predictions_key=None, doc_name_key='wikidata_label'):
    assert run is None or qe_predictions_key is None
    
    text_inputs = []    
    if run is not None:
        for text_input, q_id in zip(batch[key], batch['id']):
            # get top-1
            doc_id = next(iter(run.run[q_id]))
            doc_name = kb[int(doc_id)][doc_name_key]
            text_inputs.append(f"{text_input} {tokenizer.sep_token} {doc_name}")
    elif qe_predictions_key is not None:
        for text_input, doc_name in zip(batch[key], batch[qe_predictions_key]):
            text_inputs.append(f"{text_input} {tokenizer.sep_token} {doc_name}")
    else:
        text_inputs = batch[key]
    return text_inputs
        

def is_multimodal(model):
    model_config = getattr(model, "config", None)
    # FIXME this does not hold for ViLT and CLIP
    # TODO refactor to use the datamodule of train.data
    # maybe implement in trainer.test ?
    return model_config is not None and isinstance(model_config, MMConfig)


def get_inputs(batch, model, tokenizer, tokenization_kwargs={}, key='passage', kb=None, 
               run=None, qe_predictions_key=None):
    """
    Tokenizes input text and optionally gathers image features from the kb depending on model.
    
    Parameters
    ----------
    batch: dict
    model: nn.Module
        If it’s a ECAEncoder or IntermediateLinearFusion instance, 
        will gather image features to take as input (from the kb if kb is not None)
    tokenizer: PreTrainedTokenizer
    tokenization_kwargs: dict, optional
        To be passed to tokenizer
    key: str, optional
        Used to index the batch to get the text
    kb: Dataset, optional
        Should hold image features and be mappable from batch['index']
    """
    text_inputs = expand_query(batch, key=key, kb=kb, run=run, tokenizer=tokenizer, qe_predictions_key=qe_predictions_key)
    text_inputs = tokenizer(text_inputs, **tokenization_kwargs)
    if is_multimodal(model):
        if kb is not None:
            # FIXME: should the KB be used to expand query or get image features?
            if run is not None:
                raise NotImplementedError("The use of kb is ambiguous when run is provided AND model is multimodal")
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
          save_as='text_embedding', output_key=None, forward_kwargs={}, 
          layers=None, kb=None, call=None, run=None, qe_predictions_key=None):
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
        In this case, it will save in {save_as}_layer_{layer} the representation of the first token (DPR-like), for each layer
    call: str, optional
        Name of the method to call on model. By default, the model should be callable and is called.
    run: Run, optional
        used to expand query with results of visual search
    """
    inputs = get_inputs(batch, model, tokenizer, tokenization_kwargs=tokenization_kwargs, 
                        key=key, kb=kb, run=run, qe_predictions_key=qe_predictions_key)
    # move to device
    inputs = prepare_inputs(inputs)
    method = model if call is None else getattr(model, call)
    with torch.no_grad():        
        outputs = method(**inputs, **forward_kwargs)
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


def dataset_embed(dataset_path, map_kwargs={}, output_path=None, keep_columns=None, 
                  run=None, qe_predictions=None, qe_predictions_key=None, **fn_kwargs):
    """Loads dataset from path, maps it through embed, and saves it to output_path"""
    dataset = load_from_disk(dataset_path)
    # defaults to overwrite the dataset
    if output_path is None:
        output_path = dataset_path
        assert keep_columns is None, f"You probably don't want to overwrite {dataset_path} by keeping only {keep_columns}"
    elif keep_columns is not None:
        keep_columns = set(keep_columns)
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_columns])
    if run is not None:
        run = Run.from_file(run)
    if qe_predictions is not None:
        assert qe_predictions_key is not None
        with open(qe_predictions, 'rt') as file:
            qe_predictions = json.load(file)
        if isinstance(dataset, DatasetDict):
            raise NotImplementedError("The format of predictions saved in trainee are not compatible with a DatasetDict")
        dataset = dataset.add_column(qe_predictions_key, qe_predictions)
    fn_kwargs['run'] = run
    fn_kwargs['qe_predictions_key'] = qe_predictions_key
    dataset = dataset.map(embed, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
    dataset.save_to_disk(output_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
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
    if args['--kb']:
        kb = load_from_disk(args['--kb'])
        if is_multimodal(model):
            keep_columns = {"face_embedding", "face_box"} | model.config.image_kwargs.keys()
        else:
            keep_columns = {'wikidata_label'}
        kb = kb.remove_columns([c for c in kb.column_names if c not in keep_columns])
    else:
        kb = None
    dataset_embed(args['<dataset>'], model=model, kb=kb, output_path=args['--output'], **config)
