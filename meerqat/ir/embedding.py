"""Usage: embedding.py <dataset> <config> [--disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json

import torch

from datasets import load_from_disk, set_caching_enabled

from meerqat.models.utils import device, prepare_inputs
from meerqat.models import mm
from meerqat.data.loading import load_pretrained_in_kwargs


def get_face_inputs(batch, n_faces=4, face_dim=512, bbox_dim=7):
    face_list = batch["face_embedding"]
    batch_size = len(face_list)
    # trim or pad, and convert to tensor
    face_embeddings = torch.zeros((batch_size, n_faces, face_dim))
    face_boxes = torch.zeros((batch_size, n_faces, bbox_dim))
    # 0=masked, 1=not masked
    attention_mask = torch.zeros((batch_size, n_faces), dtype=torch.long)
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
    image_inputs = {}
    for name, image_kwarg in image_kwargs.items():
        features = torch.tensor(batch[name])
        # 0=masked, 1=not masked
        attention_mask = torch.ones(features.shape[0], dtype=torch.long)
        image_inputs[name] = dict(input=features, attention_mask=attention_mask)
    return image_inputs  


def get_inputs(batch, model, tokenizer, tokenization_kwargs={}, key='passage'):
    text_inputs = tokenizer(batch[key], **tokenization_kwargs)
    if isinstance(model, (mm.DMREncoder, mm.IntermediateLinearFusion)):
        inputs = dict(
            text_inputs=text_inputs, 
            face_inputs=get_face_inputs(batch, model.config.n_faces, **model.config.face_kwargs), 
            image_inputs=get_image_inputs(batch, model.config.image_kwargs)
        )
    else:
        inputs = text_inputs
    return inputs


def embed(batch, model, tokenizer, tokenization_kwargs={}, key='passage', 
          save_as='text_embedding', output_key=None, forward_kwargs={}, layers=None):
    inputs = get_inputs(batch, model, tokenizer, tokenization_kwargs=tokenization_kwargs, key=key)
    # move to device
    inputs = prepare_inputs(inputs)
    with torch.no_grad():
        outputs = model(**inputs, **forward_kwargs)
    # single ouput
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
            batch[f"{save_as}_layer_{layer}"] = output[layer][:,0].cpu().numpy()
    return batch


def dataset_embed(dataset_path, map_kwargs={}, **fn_kwargs):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(embed, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
    dataset.save_to_disk(dataset_path)


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
    dataset_embed(args['<dataset>'], model=model, **config)
