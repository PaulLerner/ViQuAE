# coding: utf-8
"""
WIT for MICT
============

Generates the WIT subset for Multimodal Inverse Cloze Task as described in the ECIR-submitted paper:
    - english-only subset
    - images paired with the sections
    - filtering out images with irrelevant formats (e.g. .svg)
    - splitting in train/validation/test without overlap between the articles
    - splitting sections in sentences (``meerqat.data.loading sentences``)
    - removing sections with a single sentence (DIY after)
    - map the image url to its path (DIY after)
    - images should be resized to have a maximum height or width of 512 pixels using ``meerqat.image.resize`` (DIY after)

You should end up with:
    - 877,635 pairs in train
    - 48,271 pairs in validation
    - 48,815 pairs in test
    
What you should have first
==========================
Downloaded from https://github.com/google-research-datasets/wit

(By any chance, if you have access to Jean Zay, it is available at ``$DSDIR/WIT`` with the right format).:: 
    
    $ tree WIT
    WIT/
    ├── train
    │   ├── 00
    │   │   ├── 000004379cfea6d71f7c47180c2163ee40887b7b23798535435d9b2c0065cea5.png
    │   │   ├── 000004528fa952ab9e2212ff7c749dfb1f28eb0fae2f45bec768e3ba72265420.jpg
    │   │   ├── ...
    │   │   └── 00ffff77789c938b5c2ce004d09246d1d54ef5d325d831adf3611413794d757f.jpg
    │   ├── 01
    │   ├── ...
    │   └── ff
    ├── train_images.tsv
    ├── wit_v1.train.all-00000-of-00010.tsv
    ├── wit_v1.train.all-00001-of-00010.tsv
    ├── wit_v1.train.all-00002-of-00010.tsv
    ├── wit_v1.train.all-00003-of-00010.tsv
    ├── wit_v1.train.all-00004-of-00010.tsv
    ├── wit_v1.train.all-00005-of-00010.tsv
    ├── wit_v1.train.all-00006-of-00010.tsv
    ├── wit_v1.train.all-00007-of-00010.tsv
    ├── wit_v1.train.all-00008-of-00010.tsv
    └── wit_v1.train.all-00009-of-00010.tsv

Instructions for train_images.tsv
---------------------------------

The images from WIT are stored in the "train" directory with the following naming convention:
"train/<xy>/<hash>.<ext>" where
 - <hash> is the SHA256 hash of the image's URL
 - <xy> are the first two characters of the hash (which means there are 256 subfolders named "00" to "ff")
 - <ext> is the extension of the image.

The file "train_images.tsv" contains all the URL of the images with their download status
("True" if the image could be downloaded, "False" otherwise) and the corresponding path.

Once you’ve done this mapping you hsould add it rouself to the dataset.

Sample from "train_images.tsv":::   
    
    url     downloaded      path
    http://upload.wikimedia.org/wikipedia/ca/d/d4/Trobadores.jpeg   True    train/95/953feec3651efda25c166841ec8c0cd8d2064bf59f668c8dcb62dc823963a385.jpg
    http://upload.wikimedia.org/wikipedia/commons/0/00/%2703-%2705_Pontiac_Montana_Taxi.jpg True    train/35/35bcbf0f09424126932707a702b152fac7ebd9c932a877a3f2515d9fe67bb44d.jpg
    http://upload.wikimedia.org/wikipedia/commons/0/00/%2755_Singer_4ADT_Roadster_%28Hudson%29.JPG  True    train/dd/dd10ea054385d8fac82a7bca15202434b7ce0facb01519021980ba07c5e6f626.jpg
    http://upload.wikimedia.org/wikipedia/commons/0/00/%2768_Chevrolet_Biscayne_Coupe_%28Centropolis_Laval_%2710%29.jpg     True    train/44/44a11a487b09c8118e1066491880ad7045513379b5c16cdc9460321db113ad2d.jpg
    http://upload.wikimedia.org/wikipedia/commons/0/00/%2783_Buick_Century_Sedan.JPG        False   HTTP Error 404: Not Found

Docopt
======

Usage:
wit.py <root_path> <output_path>
"""

from tqdm import tqdm
from docopt import docopt
import random

import pandas as pd
from datasets import Dataset, DatasetDict

from pathlib import Path


random.seed(0)
VALID_ENCODING = {'jpeg', 'jpg', 'png'}


def check_encoding(url):
    if url.split('.')[-1].lower() in VALID_ENCODING:
        return True
    return False


def fill_wit_for_mict(wit_for_mict):
    for _,row in tqdm(wit.iterrows(), total=len(wit)):
        wit_for_mict.setdefault(row.page_title, {})
        # filtering out images with irrelevant formats
        if not check_encoding(row.image_url):
            continue
        
        if row.is_main_image:
            wit_for_mict[row.page_title]['main_image_url'] = row.image_url
        else:
            if not isinstance(row.context_section_description, str):
                continue
            
            wit_for_mict[row.page_title].setdefault('sections', {})
            key = str(hash((row.context_section_description, row.image_url)))
            # images paired with the sections
            wit_for_mict[row.page_title]['sections'][key] = {
                "text": row.context_section_description,
                "image_url": row.image_url
            }
            
            
def dict_to_dataset(d):
    table=[]
    for title, article in tqdm(d.items()):
        for section in article['sections'].values():
            section['title'] = title
            section['main_image_url'] = article['main_image_url']
            section['main_image_path'] = article['main_image_path']
            table.append(section)
     
    df = pd.DataFrame(table)
    dataset = Dataset.from_pandas(df)
    return dataset

    
if __name__ == '__main__':
    args = docopt(__doc__)
    root = Path(args['<root_path>'])
    paths = sorted(root.glob('wit_v1.train.all-00*'))
    
    unique_wit_for_mict={}
    for path in tqdm(paths):
        wit = pd.read_csv(path, delimiter='\t')
        # english-only subset
        wit = wit[wit.language=='en']
        fill_wit_for_mict(unique_wit_for_mict)
 
    # split in test/validation/train without overlap between the articles
    titles = list(fill_wit_for_mict)
    random.shuffle(titles)
    # 5% in test and validation, rest in train
    n_in_test = round(len(titles)*0.05)
    superset = {}
    for title in titles[:n_in_test]:
        superset['test'][title] = unique_wit_for_mict.pop(title)
    for title in titles[n_in_test: n_in_test*2]:
        superset['validation'][title] = unique_wit_for_mict.pop(title)
    for title in titles[n_in_test*2: ]:
        superset['train'][title] = unique_wit_for_mict.pop(title)
        
    dataset_dict = DatasetDict()
    for name, subset in superset.values():
        dataset_dict[name] = dict_to_dataset(subset)
        
    dataset_dict.save_to_disk(args['<output_path>'])