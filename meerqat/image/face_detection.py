"""Usage: face_detection.py <dataset> [<model_config> --image_key=<image_key> --save=<root_path> --disable_caching]

Options:
--image_key=<image_key>                 Used to index the dataset item [default: image]
--save=<root_path>                      Root path to save the detected face(s).
                                        The face will actually be saved with the same file stem as the original image.
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json
from pathlib import Path

from PIL import Image
from datasets import load_from_disk, set_caching_enabled

from facenet_pytorch import MTCNN

from meerqat.models.utils import device
from meerqat.data.loading import COMMONS_PATH as IMAGE_PATH
from meerqat.data.wiki import VALID_ENCODING


def detect_face(file_name, model, save_root_path=None):
    """TODO group images w.r.t. their size to allow MTCNN batch processing"""

    # TODO: find new image or filter out question/article
    encoding = file_name.split('.')[-1].lower()
    if encoding not in VALID_ENCODING:
        return None, None

    image_path = IMAGE_PATH / file_name
    # TODO idem
    if not image_path.exists():
        return None, None

    image = Image.open(image_path).convert('RGB')
    if save_root_path:
        # if there are multiple faces, the actual save path will be
        # save_root_path/f'{image_path.stem}-{face_index}.jpg'
        # https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/mtcnn.py#L488
        # HACK: make save_path str because of facenet_pytorch/models/mtcnn.py/#L468
        save_path = str((save_root_path/image_path.stem).with_suffix('.jpg'))
    else:
        save_path = None
    face, prob = model(image, save_path=save_path, return_prob=True)
    return face, prob


def dataset_detect_face(item, image_key='image', **kwargs):
    face, prob = detect_face(item[image_key], **kwargs)
    item['face'] = face
    item['face_prob'] = prob
    return item


def dataset_detect_faces(dataset, **kwargs):
    dataset = dataset.map(dataset_detect_face, batched=False, fn_kwargs=kwargs)
    return dataset


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_path = args['<dataset>']
    dataset = load_from_disk(dataset_path)
    model_config_path = args['<model_config>']
    set_caching_enabled(not args['--disable_caching'])

    # default config
    model_config = dict(
        image_size=112,  # for arcface
        post_process=False,  # normalization: will happen in arcface
        select_largest=False,  # select largest face VS select most probable face
        keep_all=True,  # keep all faces
        device=device
    )
    # load specified config
    if model_config_path is not None:
        with open(model_config_path, 'r') as file:
            model_config.update(json.load(file))

    model = MTCNN(**model_config)

    image_key = args['--image_key']
    save_root_path = args['--save']
    if save_root_path:
        save_root_path = Path(save_root_path)
        save_root_path.mkdir(exist_ok=True, parents=True)

    dataset = dataset_detect_faces(dataset, model=model, image_key=image_key, save_root_path=save_root_path)
    dataset.save_to_disk(dataset_path)
