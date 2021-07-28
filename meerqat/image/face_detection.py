"""Usage: face_detection.py <dataset> [<model_config> --image_key=<image_key> --save=<root_path> --disable_caching --batch_size=<n>]

Options:
--image_key=<image_key>                 Used to index the dataset item [default: image]
--save=<root_path>                      Root path to save the detected face(s).
                                        The face will actually be saved with the same file stem as the original image.
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
--batch_size=<n>                        Batch size for Dataset.map. The actual batches processed by the model are first grouped by image size. [default: 64]
"""

from docopt import docopt
import json
from pathlib import Path

from datasets import load_from_disk, set_caching_enabled

from facenet_pytorch import MTCNN

from meerqat.models.utils import device
from meerqat.data.loading import COMMONS_PATH as IMAGE_PATH, load_image_batch
from meerqat.data.wiki import VALID_ENCODING


def detect_face(file_names, model, save_root_path=None):
    images = load_image_batch(file_names)

    # group images by size to allow MTCNN batch-processing
    images_by_size = {}
    for i, (image, file_name) in enumerate(zip(images, file_names)):
        # if there are multiple faces, the actual save path will be
        # save_root_path/f'{file_name}-{face_index}.jpg'
        # https://github.com/timesler/facenet-pytorch/blob/54c869c51e0e3e12f7f92f551cdd2ecd164e2443/models/mtcnn.py#L488

        # HACK: make save_path str because of facenet_pytorch/models/mtcnn.py/#L468
        if save_root_path:
            save_path = str((save_root_path/file_name).with_suffix('.jpg'))
        else:
            save_path = None

        images_by_size.setdefault(image.size, dict(images=[], save_paths=[], indices=[]))
        images_by_size[image.size]['images'].append(image)
        images_by_size[image.size]['save_paths'].append(save_path)
        images_by_size[image.size]['indices'].append(i)

    face_batch, prob_batch = [None for _ in range(len(file_names))], [None for _ in range(len(file_names))]
    for batch in images_by_size.values(): 
        faces, probs = model(batch['images'], save_path=batch['save_paths'], return_prob=True)
        for face, prob, i in zip(faces, probs, batch['indices']):
            face_batch[i] = face
            prob_batch[i] = prob

    return face_batch, prob_batch


def dataset_detect_face(item, image_key='image', **kwargs):
    face, prob = detect_face(item[image_key], **kwargs)
    item['face'] = face
    item['face_prob'] = prob
    return item


def dataset_detect_faces(dataset, batch_size=64, **kwargs):
    dataset = dataset.map(dataset_detect_face, batched=True, fn_kwargs=kwargs, batch_size=batch_size)
    return dataset


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_path = args['<dataset>']
    dataset = load_from_disk(dataset_path)
    model_config_path = args['<model_config>']
    set_caching_enabled(not args['--disable_caching'])
    batch_size = int(args['--batch_size'])
    
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
