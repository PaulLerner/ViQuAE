"""Usage: face_recognition.py <dataset> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json

import torch
from arcface_torch.backbones import get_model
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets import load_from_disk, set_caching_enabled

from meerqat.data.loading import DATA_ROOT_PATH
from meerqat.models.utils import device, map_if_not_None


ARCFACE_PATH = DATA_ROOT_PATH/"arcface"
PRETRAINED_MODELS = {
    "r50": ARCFACE_PATH/"ms1mv3_arcface_r50_fp16"/"backbone.pth"
}


def from_pretrained(model_name='r50', fp16=True, train=False):
    model = get_model(model_name, fp16=fp16)
    weight_path = PRETRAINED_MODELS[model_name]
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.train(train)
    return model.to(device=device)


def get_pil_preprocessor():
    """Use to preprocess PIL image of shape (H x W x C) loaded using Image.open(image_path).convert('RGB')"""
    return Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def preprocess_array(image):
    """
    Used to pre-process an array-like image of shape (C x H x W)

    so basically like the preprocessing of get_pil_preprocessor but without the reshaping
    """
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image.div_(255).sub_(0.5).div_(0.5)
    return image


def preprocess_and_embed(faces, model):
    """Utility function to enable the use of map_if_not_None"""
    return model(preprocess_array(faces))


def compute_face_embedding(batch, model):
    batch['face_embedding'] = map_if_not_None(batch['face'], preprocess_and_embed, model=model)
    return batch


def dataset_compute_face_embedding(dataset_path, map_kwargs={}, pretrained_kwargs={}):
    dataset = load_from_disk(dataset_path)
    model = from_pretrained(**pretrained_kwargs)
    dataset = dataset.map(compute_face_embedding, batched=True, fn_kwargs=dict(model=model), **map_kwargs)
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

    dataset_compute_face_embedding(args['<dataset>'], **config)
