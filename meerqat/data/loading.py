# coding: utf-8
"""
Usage:
loading.py passages <input> <output> [<config> --disable_caching]
loading.py map <dataset> <key> <output> [--inverse --one2many --disable_caching]

Options:
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from pathlib import Path
from docopt import docopt
import json

import re
import string

from datasets import load_dataset, Dataset, load_from_disk, set_caching_enabled
import transformers

from meerqat import __file__ as ROOT_PATH

DATA_ROOT_PATH = (Path(ROOT_PATH).parent.parent/"data").resolve()
KVQA_PATH = DATA_ROOT_PATH/"KVQA"
OKVQA_PATH = DATA_ROOT_PATH/"OK-VQA"
MSCOCO_PATH = DATA_ROOT_PATH/"MS-COCO"


def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def answer_preprocess(answer):
    """Adapted from datasets squad metric. Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(answer.lower())))


def get_pretrained(class_name, pretrained_model_name_or_path, **kwargs):
    Class = getattr(transformers, class_name)
    model = Class.from_pretrained(pretrained_model_name_or_path, **kwargs)
    return model


def map_kilt_triviaqa():
    """As instructed by https://github.com/huggingface/datasets/blob/master/datasets/kilt_tasks/README.md"""

    kilt_tasks = load_dataset("kilt_tasks")
    # Most tasks in KILT already have all required data, but KILT-TriviaQA
    # only provides the question IDs, not the questions themselves.
    # Thankfully, we can get the original TriviaQA data with:
    trivia_qa = load_dataset('trivia_qa', 'unfiltered.nocontext')
    # The KILT IDs can then be mapped to the TriviaQA questions with:
    triviaqa_map = {}

    def add_missing_data(x, trivia_qa_subset, triviaqa_map):
        i = triviaqa_map[x['id']]
        x['input'] = trivia_qa_subset[i]['question']
        x['output']['original_answer'] = trivia_qa_subset[i]['answer']['value']
        return x

    for k in ['train', 'validation', 'test']:
        triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa[k]['question_id'])])
        kilt_tasks[k + '_triviaqa'] = kilt_tasks[k + '_triviaqa'].filter(lambda x: x['id'] in triviaqa_map)
        kilt_tasks[k + '_triviaqa'] = kilt_tasks[k + '_triviaqa'].map(
            add_missing_data,
            fn_kwargs=dict(trivia_qa_subset=trivia_qa[k], triviaqa_map=triviaqa_map)
        )

    return kilt_tasks


def make_mapping(value, index, mapping, inverse=False, one2many=False):
    # default to map index to value
    if inverse:
        value, index = index, value

    if one2many:
        mapping.setdefault(index, [])
        mapping[index].append(value)
    else:
        mapping[index] = value


def make_mapping_dataset(dataset_path, key, save_name, **kwargs):
    dataset = load_from_disk(dataset_path)
    mapping = {}
    dataset.map(make_mapping, input_columns=key, with_indices=True, fn_kwargs=dict(mapping=mapping, **kwargs))
    with open(dataset_path/save_name, 'w') as file:
        json.dump(mapping, file)


def remove_special_fields(paragraphs):
    """N. B. this code puts a lot of trust into KILT pre-processing
    https://github.com/facebookresearch/KILT/blob/master/scripts/create_kilt_data_paragraphs.py
    and simply removes the title (1st paragraph), sections titles ("Section::::") and bullet-points ("BULLET::::")
    """
    preprocessed_paragraphs = []
    # drop title (first paragraph)
    for paragraph in paragraphs[1:]:
        # remove sections titles and bullet-points
        if paragraph.startswith("Section::::") or paragraph.startswith("BULLET::::"):
            continue
        # keep as is
        else:
            preprocessed_paragraphs.append(paragraph)
    return preprocessed_paragraphs


def paragraphs_preprocess(paragraphs, method=None, **kwargs):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of paragraphs to preprocess
    method: str, optional
        type of pre-processing, defaults to None (i.e. identity function)
    **kwargs: additional arguments are passed to the appropriate pre-processing function

    Returns
    -------
    paragraphs: List[str]

    """
    methods = {
        None: lambda paragraphs: paragraphs,
        "special_fields": remove_special_fields
    }
    return methods[method](paragraphs, **kwargs)


def uniform_passages(paragraphs, tokenizer, n=100):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of pre-processed paragraphs to split into passages
    tokenizer: PreTrainedTokenizer
    n: int, optional
        Number of tokens in each passage

    Returns
    -------
    passages: List[str]
        Each passage is pre-processed by the tokenizer
        (e.g. lower-cased, added space between punctuation marks, etc.)
    """
    text = ''.join(paragraphs)
    tokens = tokenizer.tokenize(text)
    passages = []
    for i in range(0, len(tokens), n):
        passages.append(tokenizer.convert_tokens_to_string(tokens[i: i + n]))
    return passages


def make_passages(paragraphs, method=None, preprocessing_method=None, preprocessing_kwargs={}, **kwargs):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of paragraphs to preprocess
    method: str, optional
        How to split the text in passages, defaults to keep the original paragraphs
    """
    paragraphs = paragraphs_preprocess(paragraphs, method=preprocessing_method, **preprocessing_kwargs)
    methods = {
        None: lambda paragraphs: paragraphs,
        "uniform": uniform_passages
    }
    return methods[method](paragraphs, **kwargs)


def make_passage_item(item, index, passage_dict, **kwargs):
    passages = make_passages(item['text']['paragraph'], **kwargs)
    total_passages = len(passage_dict['passage'])
    item['passage_index'] = list(range(total_passages, total_passages+len(passages)))
    passage_dict['passage'].extend(passages)
    passage_dict['index'].extend([index]*len(passages))
    return item


def make_passage_dataset(input_path, output_path, **kwargs):
    """Runs through dataset and create a new passage dataset from the paragraphs,
    saving index and reversed-index in both respectively"""
    dataset = load_from_disk(input_path)
    passage_dict = dict(passage=[], index=[])

    dataset = dataset.map(make_passage_item, with_indices=True, fn_kwargs=dict(passage_dict=passage_dict, **kwargs))

    passage_dataset = Dataset.from_dict(passage_dict)
    print(passage_dataset)
    passage_dataset.save_to_disk(output_path)
    dataset.save_to_disk(input_path)


def load_pretrained_in_kwargs(kwargs):
    """Recursively loads pre-trained models/tokenizer in kwargs using get_pretrained"""
    for k, v in kwargs.items():
        # base case: load pre-trained model
        if k == 'pretrained_model_name_or_path':
            return get_pretrained(**kwargs)
        # recursively look in the child arguments
        elif isinstance(v, dict):
            kwargs[k] = load_pretrained_in_kwargs(v)
        # else keep as is
    return kwargs


if __name__ == '__main__':
    args = docopt(__doc__)
    set_caching_enabled(not args['--disable_caching'])
    if args['passages']:
        config_path = args['<config>']
        # load specified config
        if config_path is not None:
            with open(config_path, 'r') as file:
                config = json.load(file)
        else:
            config = {}
        config = load_pretrained_in_kwargs(config)
        make_passage_dataset(args['<input>'], args['<output>'], **config)
    elif args['map']:
        make_mapping_dataset(Path(args['<dataset>']), args['<key>'], args['<output>'],
                             inverse=args['--inverse'], one2many=args['--one2many'])

