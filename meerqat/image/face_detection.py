"""Usage: face_detection.py <dataset> [--save=<root_path> <model_config> --image_key=<image_key> --disable_caching --batch_size=<n>]

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

import numpy as np

from datasets import load_from_disk, set_caching_enabled

from facenet_pytorch import MTCNN as facenet_MTCNN

from ..models.utils import device
from ..data.loading import COMMONS_PATH as IMAGE_PATH, load_image_batch
from ..data.wiki import VALID_ENCODING


class MTCNN(facenet_MTCNN):
    """Simply override forward to allow to return bounding boxes and landmarks"""
    def forward(self, img, save_path=None, return_prob=False, return_box=False, return_landmarks=False):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})
            return_box {bool} -- Whether or not to return the bounding box.
                (default: {False})
            return_landmarks {bool} -- Whether or not to return the facial landmarks.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected, the bounding box coordinates and facial landmarks associated.
                If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.
        Example:
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob, box, landmarks = mtcnn(img, save_path='face.png',
                                                      return_prob=True,
                                                      return_box=True,
                                                      return_landmarks=True)
        """

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        outputs = (faces, )
        if return_prob:
            outputs += (batch_probs, )
        if return_box:
            outputs += (batch_boxes, )
        if return_landmarks:
            outputs += (batch_points, )
        # unpack the tuple if only the face was asked
        if len(outputs) == 1:
            return outputs[0]
        return outputs


def detect_face(file_names, model, save_root_path=None):
    images = load_image_batch(file_names)

    # group images by size to allow MTCNN batch-processing
    images_by_size = {}
    for i, (image, file_name) in enumerate(zip(images, file_names)):
        # trouble when loading the image
        if image is None:
            continue
        
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

    prob_batch = [None for _ in range(len(file_names))]
    box_batch, landmarks_batch = prob_batch.copy(), prob_batch.copy()
    for size, batch in images_by_size.items():
        # avoid exception when image size is smaller than model.min_face_size (keep the None default value)
        # see https://github.com/timesler/facenet-pytorch/issues/176
        if min(size) < model.min_face_size:
            continue
        # extract the face and save them using facenet_pytorch
        if save_root_path is not None:
            faces, probs, boxes, landmarks = model(batch['images'], save_path=batch['save_paths'],
                                                   return_prob=True,
                                                   return_box=True,
                                                   return_landmarks=True)
        # no need to extract the faces
        else:
            boxes, probs, landmarks = model.detect(batch['images'], landmarks=True)
            # Select faces
            if not model.keep_all:
                boxes, probs, landmarks = model.select_boxes(
                    boxes, probs, landmarks, batch['images'], method=model.selection_method
                )
        # save the faces at the right index
        for prob, box, landmark, i in zip(probs, boxes, landmarks, batch['indices']):
            # HACK: convert to list to fix https://github.com/PaulLerner/ViQuAE/issues/1
            prob_batch[i] = prob.tolist() if isinstance(prob, np.ndarray) else prob
            box_batch[i] = box.tolist() if box is not None and isinstance(box, np.ndarray) else box
            landmarks_batch[i] = landmark.tolist() if landmark is not None and isinstance(landmark, np.ndarray) else landmark
    return prob_batch, box_batch, landmarks_batch


def dataset_detect_face(item, image_key='image', **kwargs):
    prob, box, landmarks = detect_face(item[image_key], **kwargs)
    item['face_prob'] = prob
    item['face_box'] = box
    item['face_landmarks'] = landmarks
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
    if save_root_path is not None:
        save_root_path = Path(save_root_path)
        save_root_path.mkdir(exist_ok=True, parents=True)

    dataset = dataset_detect_faces(dataset, model=model, image_key=image_key,
                                   save_root_path=save_root_path, batch_size=batch_size)
    dataset.save_to_disk(dataset_path)
