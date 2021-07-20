import torch
from arcface_torch.backbones import get_model
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets import load_from_disk

from meerqat.data.loading import DATA_ROOT_PATH
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


def preprocess_array(image):
    """
    Used to pre-process an array-like image of shape (C x H x W)

    so basically like the preprocessing of get_pil_preprocessor but without the reshaping
    """
    image = torch.tensor(image, dtype=torch.float32, device=device)
    image.div_(255).sub_(0.5).div_(0.5)
    return image


def compute_face_embedding(batch, model):
    faces = preprocess_array(batch['face'])
    embeddings = model(faces)
    batch['face_embedding'] = embeddings
    return batch


def dataset_compute_face_embedding(dataset_path, map_kwargs={}, pretrained_kwargs={}):
    dataset = load_from_disk(dataset_path)
    model = from_pretrained(**pretrained_kwargs)
    dataset = dataset.map(compute_face_embedding, batched=True, fn_kwargs=dict(model=model), **map_kwargs)
    dataset.save_to_disk(dataset_path)



