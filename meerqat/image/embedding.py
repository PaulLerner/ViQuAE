"""Usage: embedding.py <dataset> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json
from collections import OrderedDict
from PIL import Image

import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_from_disk, set_caching_enabled

from meerqat.models.utils import device
from meerqat.data.wiki import COMMONS_PATH as IMAGE_PATH


class ImageEncoder(nn.Module):
    """Simply encode and pool the features"""
    def __init__(self, encoder, pool):
        super().__init__()
        self.encoder = encoder
        self.pool = pool

    def forward(self, x):
        features = self.encoder(x)
        return self.pool(features)


def get_nn_module(Class_name, *args, **kwargs):
    return getattr(nn, Class_name)(*args, **kwargs)


def from_pretrained(model_name='resnet50', **kwargs):
    return getattr(torchvision.models, model_name)(pretrained=True, **kwargs)


def get_encoder(torchvision_model):
    """Keep only convolutional layers (i.e. remove final pooling and classification layers)"""
    if isinstance(torchvision_model, (torchvision.models.ResNet, )):
        cutoff = -2
    else:
        raise NotImplementedError(f"Don't know where the convolutional layers end for {torchvision_model}")

    return nn.Sequential(OrderedDict(list(torchvision_model.named_children())[:cutoff]))


def get_model(pretrained_kwargs={}, pool_kwargs={}, training=False):
    torchvision_model = from_pretrained(**pretrained_kwargs)
    encoder = get_encoder(torchvision_model)
    pool = get_nn_module(**pool_kwargs)
    return ImageEncoder(encoder, pool).to(device).train(training)


def get_transform(resize_kwargs=dict(size=224), crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return Compose([
        Resize(**resize_kwargs),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


def load_image_batch(file_names):
    return [Image.open(IMAGE_PATH / file_name).convert('RGB') for file_name in file_names]


def embed(batch, model, transform):
    images = load_image_batch(batch['image'])
    images = [transform(image).unsqueeze(0) for image in images]
    images = torch.cat(images).to(device)
    with torch.no_grad():
        image_embeddings = model(images)
    batch['image_embedding'] = image_embeddings.cpu().numpy()
    return batch


def dataset_embed(dataset_path, map_kwargs={}, model_kwargs={}, transform_kwargs={}):
    dataset = load_from_disk(dataset_path)
    model = get_model(**model_kwargs)
    transform = get_transform(**transform_kwargs)
    dataset = dataset.map(embed, batched=True, fn_kwargs=dict(model=model, transform=transform), **map_kwargs)
    dataset.save_to_disk(dataset_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    config_path = args['<config>']
    if config_path is not None:
        with open(config_path, 'rt') as file:
            config = json.load(file)
    else:
        config = {}

    dataset_embed(args['<dataset>'], **config)
