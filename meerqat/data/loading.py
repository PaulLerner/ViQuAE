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

from PIL import Image

import re
import string

import spacy
from spacy.lang.en import English
from datasets import load_dataset, Dataset, load_from_disk, set_caching_enabled
import transformers

from meerqat.train import trainee
from meerqat import __file__ as ROOT_PATH

DATA_ROOT_PATH = (Path(ROOT_PATH).parent.parent/"data").resolve()
COMMONS_PATH = DATA_ROOT_PATH / "Commons"
KVQA_PATH = DATA_ROOT_PATH/"KVQA"
OKVQA_PATH = DATA_ROOT_PATH/"OK-VQA"
MSCOCO_PATH = DATA_ROOT_PATH/"MS-COCO"


def load_image_batch(file_names):
    return [Image.open(IMAGE_PATH / file_name).convert('RGB') for file_name in file_names]


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


def get_pretrained(class_name, pretrained_model_name_or_path, trainee_class=None, trainee_kwargs={}, **kwargs):
    Class = None
    modules = [trainee, transformers]
    for module in modules:
        Class = getattr(module, class_name, None)
        if Class is not None:
            break
    if Class is not None:
        if issubclass(Class, trainee.Trainee):
            # first get the wrapped pre-trained model in Trainee
            trainee_model = get_pretrained(trainee_class, pretrained_model_name_or_path, **kwargs)
            # then init the Trainee
            model = Class(trainee_model, **trainee_kwargs)
        # simply use PreTrainedModel.from_pretrained method
        else:
            model = Class.from_pretrained(pretrained_model_name_or_path, **kwargs)
    else:
        raise ValueError(f"Could not find {class_name} in {modules}")
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


def uniform_passages(paragraphs, tokenizer, n=100, title=None):
    """
    Parameters
    ----------
    paragraphs: List[str]
        List of pre-processed paragraphs to split into passages
    tokenizer: PreTrainedTokenizer
    n: int, optional
        Number of tokens in each passage (excluding title)
        Defaults to 100
    title: str, optional
        To prepend at the beginning of each passage like "<title> [SEP] <passage>"
        Defaults to None (only "<passage>")

    Returns
    -------
    passages: List[str]
        Each passage is pre-processed by the tokenizer
        (e.g. lower-cased, added space between punctuation marks, etc.)
    """
    text = ''.join(paragraphs)
    tokens = tokenizer.tokenize(text)
    if title is not None:
        title = tokenizer.convert_tokens_to_string(tokenizer.tokenize(title))
        title = f"{title} {tokenizer.sep_token} "

    passages = []
    for i in range(0, len(tokens), n):
        passage = tokenizer.convert_tokens_to_string(tokens[i: i + n])
        if title is not None:
            passage = title + passage
        passages.append(passage)
    return passages


def uniform_passages_of_sentences(paragraphs, model, n=100, title=None, sep_token='[SEP]'):
    """
    N. B. unlike uniform_passages which is based on transformers PreTrainedTokenizer
    here we're able to get back the un-processed text corresponding to the tokens
    so the output text is not changed (e.g. not lower-cased), 
    only the whitespace between sentences is lost (it is always set to ' ')

    Parameters
    ----------
    paragraphs: List[str]
        List of pre-processed paragraphs to split into passages
    model: spacy model
    n: int, optional
        Maximum number of tokens in each passage (excluding title)
        There can actually be more tokens than this if the passage is a single sentence (with more tokens than n)
        Defaults to 100
    title: str, optional
        To prepend at the beginning of each passage like "<title> [SEP] <passage>"
        Defaults to None (only "<passage>")
    sep_token: str, optional
        To separate title and passages (no effect if title is None)
        Defaults to '[SEP]'

    Returns
    -------
    passages: List[str]
    """
    text = ''.join(paragraphs)
    if title is not None:
        title = f"{title} {sep_token} "

    # 1. segment into sentences
    sentences = model(text).sents

    # 2. group sentences together so that there is maximum n tokens in each passage
    passages = []
    passage = []
    tokens_in_passage = 0
    for sent in sentences:
        # passage would be too long
        if tokens_in_passage + len(sent) > n:
            # so we append the existing passage and start a new one
            if len(passage) > 0:
                passages.append(' '.join(passage))
                passage = [sent.text]
                tokens_in_passage = len(sent)
            # corner case where a single sentence has more tokens than n
            else:
                passages.append(sent.text)
        # add the sentence to the passage
        else:
            passage.append(sent.text)
            tokens_in_passage += len(sent)

    # leftovers        
    if len(passage) > 0:
        passages.append(' '.join(passage))
    
    if title is not None:
        passages = [title + passage for passage in passages]

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
        "uniform": uniform_passages,
        "uniform_sents": uniform_passages_of_sentences
    }
    return methods[method](paragraphs, **kwargs)


def make_passage_item(item, index, passage_dict, prepend_title=False, **kwargs):
    if prepend_title:
        title = item['wikipedia_title']
    else:
        title = None
    passages = make_passages(item['text']['paragraph'], title=title, **kwargs)
    total_passages = len(passage_dict['passage'])
    item['passage_index'] = list(range(total_passages, total_passages+len(passages)))
    passage_dict['passage'].extend(passages)
    passage_dict['index'].extend([index]*len(passages))
    return item


def make_passage_dataset(input_path, output_path, sentencizer=False, **kwargs):
    """Runs through dataset and create a new passage dataset from the paragraphs,
    saving index and reversed-index in both respectively"""
    dataset = load_from_disk(input_path)
    passage_dict = dict(passage=[], index=[])

    # spacy sentence segmentation
    if sentencizer:
        model = English()
        sentencizer = model.create_pipe("sentencizer")
        model.add_pipe(sentencizer)
        kwargs["model"] = model

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

