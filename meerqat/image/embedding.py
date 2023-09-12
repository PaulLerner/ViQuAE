"""Usage: embedding.py <dataset> [<config> --disable_caching --output=<path>]

Options:
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--output=<path>         Optionally save the resulting dataset there instead of overwriting the input dataset.
"""

from docopt import docopt
import json
from collections import OrderedDict

from multiprocessing import Pool

import numpy as np

import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from transformers import FeatureExtractionMixin

from datasets import load_from_disk, set_caching_enabled

from ..models.utils import device
from ..data.loading import load_image_batch, get_pretrained


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


def from_pretrained(model_name='resnet50', imagenet=True, pretrained_model_path=None, **kwargs):
    """
    Notes
    -----
    For models trained on other dataset than imagenet, don’t forget to pass the right num_classes in kwargs

    Examples
    --------
    To load from a Places365 checkpoint, first process the state_dict as this:
    >>> checkpoint = torch.load("resnet50_places365.pth.tar", map_location="cpu")
    >>> state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    >>> torch.save(state_dict, "resnet50_places365_state_dict.pth")
    """
    model = getattr(torchvision.models, model_name)(pretrained=imagenet, **kwargs)
    if not imagenet:
        print(f"Loading pre-trained model from '{pretrained_model_path}'")
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
    else:
        print("Loaded pre-trained model on ImageNet")
    return model


def get_encoder(torchvision_model):
    """Keep only convolutional layers (i.e. remove final pooling and classification layers)"""
    if isinstance(torchvision_model, (torchvision.models.ResNet, )):
        cutoff = -2
    else:
        raise NotImplementedError(f"Don't know where the convolutional layers end for {torchvision_model}")

    return nn.Sequential(OrderedDict(list(torchvision_model.named_children())[:cutoff]))


def get_torchvision_model(pretrained_kwargs={}, pool_kwargs={}):
    """Get model pre-trained on ImageNet or Places365"""
    torchvision_model = from_pretrained(**pretrained_kwargs)
    encoder = get_encoder(torchvision_model)
    pool = get_nn_module(**pool_kwargs)
    return ImageEncoder(encoder, pool)


def get_transform(resize_kwargs=dict(size=224), crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # N. B. default parameters work for both ImageNet provided by pytorch 
    # and places365 https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py#L31
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
    elif model_type == "torchscript":
        model = torch.jit.load(**model_kwargs)
        transform = get_transform(**transform_kwargs)
    elif model_type == "clip":
        import clip
        clip_model, transform = clip.load(**model_kwargs, device=device)
        # only interested in the visual bottleneck here (for content-based image retrieval)
        model = clip_model.visual
    elif model_type == "transformers":
        model = get_pretrained(**model_kwargs)
        transform = get_pretrained(**transform_kwargs)
    else:
        raise ValueError(f"Unexpected model type '{model_type}'")

    model = model.to(device).train(training)
    print(model)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return dict(model=model, transform=transform)


def embed(batch, model, transform, save_as='image_embedding', image_key='image', 
          call=None, pool=None):
    images = load_image_batch(batch[image_key], pool=pool)
    not_None_images, not_None_indices, output = [], [], []
    for i, image in enumerate(images):
        output.append(None)
        if image is not None:
            not_None_images.append(image)
            not_None_indices.append(i)
    if len(not_None_images) == 0:
        return output
    images = not_None_images
    if isinstance(transform, FeatureExtractionMixin):
        if pool is not None:
            image_list = pool.map(transform, images)
            images = {}
            for k in image_list[0].keys():
                images[k] = torch.tensor(
                    np.concatenate([image[k] for image in image_list]), 
                    device=device
                )
        else:
            images = transform(images, return_tensors="pt")
            images = {k: v.to(device) for k, v in images.items()}
    else:
        if pool is not None:
            images = pool.map(transform, images)
            images = torch.stack(images).contiguous().to(device)
        else:
            images = [transform(image).unsqueeze(0) for image in images]
            images = torch.cat(images).to(device)
    method = model if call is None else getattr(model, call)
    with torch.no_grad():
        if isinstance(images, dict):
            image_embeddings = method(**images)
        else:
            image_embeddings = method(images)
    not_None_output = image_embeddings.squeeze().cpu().numpy()
    for i, j in enumerate(not_None_indices):
        output[j] = not_None_output[i]
    batch[save_as] = output
    return batch


def dataset_embed(dataset_path, map_kwargs={}, model_kwargs={}, transform_kwargs={}, 
                  output_path=None, keep_columns=None, processes=None, **fn_kwargs):
    dataset = load_from_disk(dataset_path)
    # defaults to overwrite the dataset
    if output_path is None:
        output_path = dataset_path
        assert keep_columns is None, f"You probably don't want to overwrite {dataset_path} by keeping only {keep_columns}"
    elif keep_columns is not None:
        keep_columns = set(keep_columns)
        dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_columns])
        
    fn_kwargs.update(get_model_and_transform(model_kwargs=model_kwargs, transform_kwargs=transform_kwargs))
    fn_kwargs['pool'] = None if processes is None else Pool(processes=processes)
    dataset = dataset.map(embed, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
    dataset.save_to_disk(output_path)


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    config_path = args['<config>']
    if config_path is not None:
        with open(config_path, 'rt') as file:
            config = json.load(file)
    else:
        config = {}

    dataset_embed(args['<dataset>'], output_path=args['--output'], **config)