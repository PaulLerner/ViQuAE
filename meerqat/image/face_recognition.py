"""Usage: face_recognition.py <dataset> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json
from pathlib import Path
import warnings

import numpy as np

import torch
try:
    from arcface_torch.backbones import get_model
except ImportError as e:
    warnings.warn(f"Got the following ImportError: {e}.\n Please install arcface_torch as instructed in README.")
from datasets import load_from_disk, set_caching_enabled

from torchvision.transforms import Compose, ToTensor, Normalize
import cv2
from skimage import transform
from PIL import Image

from ..data.loading import DATA_ROOT_PATH, load_image
from ..models.utils import device


ARCFACE_PATH = DATA_ROOT_PATH/"arcface"
PRETRAINED_MODELS = {
    "r50": ARCFACE_PATH/"ms1mv3_arcface_r50_fp16"/"backbone.pth"
}
# taken from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval_ijbc.py
SRC = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)
SRC[:, 0] += 8.0


def similarity_transform(image, landmarks, src, tform, image_size=112):
    """Adapted from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/eval_ijbc.py"""
    # FIXME is there a way to do this without going from Image to ndarray to Image?
    tform.estimate(landmarks, src)
    M = tform.params[0:2, :]
    transformed_face = cv2.warpAffine(np.array(image, dtype=np.uint8),
                                      M, (image_size, image_size),
                                      borderValue=0.0)
    return Image.fromarray(transformed_face)


def from_pretrained(model_name='r50', fp16=True, train=False):
    model = get_model(model_name, fp16=fp16)
    weight_path = PRETRAINED_MODELS[model_name]
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.train(train)
    return model.to(device=device)


def get_pil_preprocessor():
    """Use to preprocess PIL image of shape (H x W x C) loaded using Image.open(image_path).convert('RGB')"""
    return Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def compute_face_embedding(batch, model, preprocessor, tform, max_n_faces=1, image_key='image'):
    # 1. filter out images without any detected faces
    output = []
    not_None_values, not_None_values_indices = [], []
    for i, (image, landmarks) in enumerate(zip(batch[image_key], batch['face_landmarks'])):
        # will be overwritten for not_None_values
        output.append(None)
        if landmarks is not None:
            image = load_image(image)
            landmarks = np.array(landmarks[:max_n_faces], dtype=np.float32)
            for landmark in landmarks:
                face = similarity_transform(image, landmark, SRC, tform)
                not_None_values.append(preprocessor(face).unsqueeze(0))
            not_None_values_indices.append((i, landmarks.shape[0]))
    # None of the image had a face detected
    if not not_None_values:
        batch['face_embedding'] = output
        return batch

    # 2. compute face embedding
    not_None_values = torch.cat(not_None_values, axis=0).to(device=device)
    not_None_output = model(not_None_values)

    # 3. return the results in a list of list with proper indices
    j = 0
    for i, n_faces in not_None_values_indices:
        output[i] = not_None_output[j: j+n_faces]
        j += n_faces

    batch['face_embedding'] = output
    return batch


def dataset_compute_face_embedding(dataset_path, map_kwargs={}, pretrained_kwargs={}, fn_kwargs={}):
    dataset = load_from_disk(dataset_path)
    model = from_pretrained(**pretrained_kwargs)
    preprocessor = get_pil_preprocessor()
    tform = transform.SimilarityTransform()
    fn_kwargs.update(dict(model=model, preprocessor=preprocessor, tform=tform))
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

    dataset_compute_face_embedding(args['<dataset>'], **config)
