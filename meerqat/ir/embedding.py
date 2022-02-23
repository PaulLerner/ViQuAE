"""Usage: embedding.py <dataset> <config> [--disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json

import torch

from datasets import load_from_disk, set_caching_enabled

from meerqat.models.utils import device, prepare_inputs
from meerqat.data.loading import load_pretrained_in_kwargs


def embed(batch, model, tokenizer, tokenization_kwargs={}, key='passage', 
          save_as='text_embedding', output_key=None, forward_kwargs={}, layers=None):
    inputs = tokenizer(batch[key], **tokenization_kwargs)
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
