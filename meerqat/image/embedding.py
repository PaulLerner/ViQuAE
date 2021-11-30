"""Usage: embedding.py <dataset> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json
from collections import OrderedDict

import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from datasets import load_from_disk, set_caching_enabled

import clip

from meerqat.models.utils import device
from meerqat.data.loading import load_image_batch


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


def get_torchvision_model(pretrained_kwargs={}, pool_kwargs={}):
    """Get model pre-trained on ImageNet"""
    torchvision_model = from_pretrained(**pretrained_kwargs)
    encoder = get_encoder(torchvision_model)
    pool = get_nn_module(**pool_kwargs)
    return ImageEncoder(encoder, pool)


def get_transform(resize_kwargs=dict(size=224), crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return Compose([
        Resize(**resize_kwargs),
        CenterCrop(crop_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


def get_model_and_transform(model_kwargs={}, transform_kwargs={}):
    training = model_kwargs.pop("training", False)
    model_type = model_kwargs.pop("type", "torchvision")
    if model_type == "torchvision":
        model = get_torchvision_model(**model_kwargs)
        transform = get_transform(**transform_kwargs)
    elif model_type == "clip":
        clip_model, transform = clip.load(**model_kwargs, device=device)
        # only interested in the visual bottleneck here (for content-based image retrieval)
        model = clip_model.visual
    else:
        raise ValueError(f"Unexpected model type '{model_type}'")

    model = model.to(device).train(training)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return dict(model=model, transform=transform)


def embed(batch, model, transform, save_as='image_embedding'):
    images = load_image_batch(batch['image'])
    images = [transform(image).unsqueeze(0) for image in images]
    images = torch.cat(images).to(device)
    with torch.no_grad():
        image_embeddings = model(images)
    batch[save_as] = image_embeddings.squeeze().cpu().numpy()
    return batch


def dataset_embed(dataset_path, map_kwargs={}, model_kwargs={}, transform_kwargs={}, **fn_kwargs):
    dataset = load_from_disk(dataset_path)
    fn_kwargs.update(get_model_and_transform(model_kwargs=model_kwargs, transform_kwargs=transform_kwargs))
    dataset = dataset.map(embed, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
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