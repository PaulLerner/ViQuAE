"""Usage: face_box.py <dataset> [--image_key=<image_key> --disable_caching]

Options:
--image_key=<image_key>                 Used to index the dataset item [default: image]
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import numpy as np

from datasets import load_from_disk, set_caching_enabled

from meerqat.data.loading import load_image


def scale_box(item, image_key='image'):
    box = item['face_box']
    if box is None:
        return item
    image = load_image(item[image_key])
    w, h = image.size
    # (n_faces, 4): coordinates are stored as: (x1, y1, x2, y2)
    box = np.array(box)
    # scales x per width
    box[:, 0] /= w
    box[:, 2] /= w
    # scales y per height
    box[:, 1] /= h
    box[:, 3] /= h
    # format bounding boxes as in UNITER (Chen et al.)
    new_w = box[:, 2] - box[:, 0]
    new_h = box[:, 3] - box[:, 1]
    new_area = new_w * new_h
    item['face_box'] = np.concatenate((box, new_w.reshape(-1, 1), new_h.reshape(-1, 1), new_area.reshape(-1, 1)), axis=1)

    landmarks = item.get('face_landmarks')
    if landmarks is not None:
        landmarks = np.array(landmarks)
        landmarks[:, :, 0] /= w
        landmarks[:, :, 1] /= h
        item['face_landmarks'] = landmarks

    return item


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_path = args['<dataset>']
    dataset = load_from_disk(dataset_path)
    set_caching_enabled(not args['--disable_caching'])

    image_key = args['--image_key']
    dataset = dataset.map(scale_box, fn_kwargs=dict(image_key=image_key))
    dataset.save_to_disk(dataset_path)
