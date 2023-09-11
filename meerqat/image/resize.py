"""Usage: resize.py <dataset> <output> [<config> --disable_caching]

Options:
--disable_caching                       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""
from pathlib import Path
from docopt import docopt
import json

from multiprocessing import Pool

from torchvision.transforms import Compose, Resize
from datasets import load_from_disk, set_caching_enabled

from ..data.loading import load_image, save_image


def get_transform(resize_kwargs=dict(size=512)):
    return Compose([
        Resize(**resize_kwargs)
    ])


def resize(file_name, transform, output_root):
    output_path = output_root/file_name
    if output_path.exists():
        return None
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image = load_image(file_name)
    if image is None:
        return None
    image = transform(image)
    save_image(image, output_path)


def batch_resize(file_names, transform, output_root):
    # hacking the way into Pool
    inputs = [(file_name, transform, output_root) for file_name in file_names]
    with Pool() as pool:
        pool.starmap(resize, inputs)
        
        
def dataset_resize(dataset_path, output_path, map_kwargs={}, transform_kwargs={}, image_key='image', **fn_kwargs):
    dataset = load_from_disk(dataset_path)
    transform = get_transform(**transform_kwargs)
    fn_kwargs["transform"] = transform
    fn_kwargs["output_root"] = Path(output_path)
    dataset.map(batch_resize, batched=True, fn_kwargs=fn_kwargs, input_columns=image_key, **map_kwargs)


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    config_path = args['<config>']
    if config_path is not None:
        with open(config_path, 'rt') as file:
            config = json.load(file)
    else:
        config = {}

    dataset_resize(args['<dataset>'], args['<output>'], **config)
