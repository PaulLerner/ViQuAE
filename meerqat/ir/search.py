"""Both dense and sparse information retrieval is done via HF-Datasets, using FAISS and ElasticSearch, respectively

Usage: search.py <dataset> <es_kb> <faiss_kb> <config> [--k=<k> --disable_caching]

Options:
--k                     Hyperparameter to search for the k nearest neighbors [default: 100].
--disable_caching       Disables Dataset caching (useless when using save_to_disk), see datasets.set_caching_enabled()
"""

from docopt import docopt
import json

from datasets import load_from_disk, set_caching_enabled


def scores2dict(scores_batch, indices_batch):
    scores_dicts = []
    for scores, indices in zip(scores_batch, indices_batch):
        scores_dicts.append(dict(zip(indices, scores)))
    return scores_dicts


def linear_fusion(es_scores, es_indices, faiss_scores, faiss_indices, k=100, alpha=1.1):
    scores_batch, indices_batch = [], []
    es_dicts = scores2dict(es_scores, es_indices)
    faiss_dicts = scores2dict(faiss_scores, faiss_indices)

    for es_dict, faiss_dict in zip(es_dicts, faiss_dicts):
        # fusion = es + alpha * faiss
        for index, score in faiss_dict.items():
            es_dict.setdefault(index, 0.)
            es_dict[index] += alpha * score
        # sort in desc. order and keep top-k
        indices = sorted(es_dict, key=es_dict.get, reverse=True)[:k]
        scores = [es_dict[index] for index in indices]
        scores_batch.append(scores)
        indices_batch.append(indices)

    return scores_batch, indices_batch


def fuse(es_scores, es_indices, faiss_scores, faiss_indices, method='linear', **kwargs):
    """Should return a (scores, indices) tuples the same way as Dataset.search_batch"""

    # easy to fuse when there is only one input
    if es_scores is None and es_indices is None:
        return faiss_scores, faiss_indices
    elif faiss_scores is None and faiss_indices is None:
        return faiss_scores, faiss_indices

    # TODO align es_indices and faiss_indices

    fusions = dict(linear=linear_fusion)

    return fusions[method](es_scores, es_indices, faiss_scores, faiss_indices, **kwargs)


def search(batch, k=100,
           es_kb=None, faiss_kb=None,
           es_index_name=None, faiss_index_name=None,
           es_key=None, faiss_key=None, fusion_method='linear', **fusion_kwargs):
    # TODO compute metrics

    # 1. search using ElasticSearch
    if es_kb is not None:
        es_scores, es_indices = es_kb.search_batch(es_index_name, batch[es_key], k)
        batch['es_scores'], batch['es_indices'] = es_scores, es_indices
    else:
        es_scores, es_indices = None, None

    # 2. search using FAISS
    if faiss_kb is not None:
        faiss_scores, faiss_indices = faiss_kb.search_batch(faiss_index_name, batch[faiss_key], k)
        batch['faiss_scores'], batch['faiss_indices'] = faiss_scores, faiss_indices
    else:
        faiss_scores, faiss_indices = None, None

    # 3. fuse the results of the 2 searches
    if es_kb is not None and faiss_kb is not None:
        scores, indices = fuse(es_scores, es_indices, faiss_scores, faiss_indices, k=k, method=fusion_method, **fusion_kwargs)
        batch['scores'], batch['indices'] = scores, indices

    return batch


def dataset_search(dataset, k=100, es_kb=None, faiss_kb=None,
                   es_kwargs={}, faiss_kwargs={}, map_kwargs={}, fusion_kwargs={}):
    assert (es_kb is not None or faiss_kb is not None), 'Expected at least one KB'

    # add ElasticSearch index
    if es_kb is not None:
        es_key = es_kwargs.pop('key')
        es_kb.add_elasticsearch_index(**es_kwargs)
        es_index_name = es_kwargs.get('index_name', es_kwargs['column'])
    else:
        es_key, es_index_name = None, None
    # add FAISS index
    if faiss_kb is not None:
        faiss_key = faiss_kwargs.pop('key')
        # either load FAISS index or build it
        load = faiss_kwargs.pop('load', False)
        faiss_index_name = faiss_kwargs.get('index_name', faiss_kwargs['column'])
        if load:
            faiss_kb.load_faiss_index(**faiss_kwargs)
        else:
            save_path = faiss_kwargs.pop('file', None)
            faiss_kb.add_faiss_index(**faiss_kwargs)
            # save FAISS index (so it can be loaded later)
            if save_path is not None:
                faiss_kb.save_faiss_index(faiss_index_name, save_path)
    else:
        faiss_index_name, faiss_key = None, None

    # search expects a batch as input
    fn_kwargs = dict(k=k,
                     es_kb=es_kb, faiss_kb=faiss_kb,
                     es_index_name=es_index_name, faiss_index_name=faiss_index_name,
                     es_key=es_key, faiss_key=faiss_key, **fusion_kwargs)
    dataset = dataset.map(search, fn_kwargs=fn_kwargs, batched=True, **map_kwargs)

    return dataset


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset_path = args['<dataset>']
    dataset = load_from_disk(dataset_path)
    es_kb = load_from_disk(args['<es_kb>'])
    faiss_kb = load_from_disk(args['<faiss_kb>'])
    set_caching_enabled(not args['--disable_caching'])
    config_path = args['<config>']
    with open(config_path, 'r') as file:
        config = json.load(file)
    format_kwargs = config.pop('format', {})
    dataset.set_format(**format_kwargs)

    k = int(args['--k'])

    dataset = dataset_search(dataset, k, es_kb, faiss_kb, **config)

    dataset.save_to_disk(dataset_path)
