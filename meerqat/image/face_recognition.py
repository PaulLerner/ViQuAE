"""Usage: face_recognition.py <dataset> <face_path> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json
from pathlib import Path

import torch
from arcface_torch.backbones import get_model
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets import load_from_disk, set_caching_enabled

from meerqat.data.loading import DATA_ROOT_PATH, load_faces
from meerqat.models.utils import device


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


def compute_face_embedding(batch, model, preprocessor, face_path, max_n_faces=1):
    if max_n_faces != 1:
        raise NotImplementedError()

    # 1. filter out images without any detected faces
    output = []
    not_None_values, not_None_values_indices = [], []
    for i, image in enumerate(batch['image']):
        # did we detect a face on this image?
        face = load_faces(image, face_path, max_n_faces=max_n_faces)
        # will be overwritten for not_None_values
        output.append(None)
        # if yes then preprocess it
        if face is not None:
            not_None_values.append(preprocessor(face).unsqueeze(0))
            not_None_values_indices.append(i)
    # None of the image had a face detected
    if not not_None_values:
        batch['face_embedding'] = output
        return batch

    # 2. compute face embedding
    not_None_values = torch.cat(not_None_values, axis=0)
    not_None_output = model(not_None_values)

    # 3. return the results in a list of list with proper indices
    for j, i in enumerate(not_None_values_indices):
        output[i] = not_None_output[j]

    batch['face_embedding'] = output
    return batch


def dataset_compute_face_embedding(dataset_path, face_path, map_kwargs={}, pretrained_kwargs={}):
    dataset = load_from_disk(dataset_path)
    model = from_pretrained(**pretrained_kwargs)
    preprocessor = get_pil_preprocessor()
    fn_kwargs = dict(model=model, preprocessor=preprocessor, face_path=face_path)
    dataset = dataset.map(compute_face_embedding, batched=True, fn_kwargs=fn_kwargs, **map_kwargs)
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

    dataset_compute_face_embedding(args['<dataset>'], face_path=Path(args['<face_path>']), **config)
