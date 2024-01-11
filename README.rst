.. image:: ./meerqat_logo_by_hlb.png

``meerqat``
===========

Source code and data used in the papers:
    - `ViQuAE, a Dataset for Knowledge-based Visual Question Answering about Named Entities <https://hal.science/hal-03650618>`__ 
      (Lerner et al., SIGIR’22) 
    - `Multimodal Inverse Cloze Task for Knowledge-based Visual Question Answering <https://hal.science/hal-03933089>`__  
      (Lerner et al., ECIR'23)
    - `Cross-modal Retrieval for Knowledge-based Visual Question Answering <https://hal.science/hal-04384431>`__ (Lerner et al., ECIR'24)

See also `MEERQAT project <https://www.meerqat.fr/>`__.

Getting the ViQuAE dataset and KB
=================================

The data is provided in two formats: HF’s ``datasets`` (based on Apache
Arrow) and plain-text JSONL files (one JSON object per line). Both
formats can be used in the same way as ``datasets`` parses objects into
python ``dict`` (see below), however our code only supports (and is
heavily based upon) ``datasets``. Images are distributed separately, in
standard formats (e.g. jpg).  

The images
----------

Here’s how to get the images grounding the questions of the dataset:

.. code:: sh

   # get the images. TODO integrate this in a single dataset
   git clone https://huggingface.co/datasets/PaulLerner/viquae_images
   # to get ALL images (dataset+KB) use https://huggingface.co/datasets/PaulLerner/viquae_all_images instead 
   cd viquae_images
   # in viquae_all_images, the archive is split into parts of 5GB
   # cat parts/* > images.tar.gz
   tar -xzvf images.tar.gz
   export VIQUAE_IMAGES_PATH=$PWD/images

Alternatively, you can download images from Wikimedia Commons using
``meerqat.data.kilt2vqa download`` (see below).

The ViQuAE dataset
------------------

If you don’t want to use ``datasets`` you can get the data directly from
https://huggingface.co/datasets/PaulLerner/viquae_dataset
(e.g. ``git clone https://huggingface.co/datasets/PaulLerner/viquae_dataset``).

The dataset format largely follows
`KILT <https://huggingface.co/datasets/kilt_tasks>`__. Here I’ll
describe the dataset without pre-computed features. Pre-computed
features are basically the output of each step described in
`EXPERIMENTS.rst <./EXPERIMENTS.rst>`__.

.. code:: py

   In [1]: from datasets import load_dataset
      ...: dataset = load_dataset('PaulLerner/viquae_dataset')
   In [2]: dataset
   Out[2]: 
   DatasetDict({
       train: Dataset({
           features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
           num_rows: 1190
       })
       validation: Dataset({
           features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
           num_rows: 1250
       })
       test: Dataset({
           features: ['image', 'input', 'kilt_id', 'id', 'meta', 'original_question', 'output', 'url', 'wikidata_id'],
           num_rows: 1257
       })
   })
   In [3]: item = dataset['test'][0]

   # this is now a dict, like the JSON object loaded from the JSONL files
   In [4]: type(item)
   Out[4]: dict

   # url of the grounding image
   In [5]: item['url']
   Out[5]: 'http://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Jackie_Wilson.png/512px-Jackie_Wilson.png'

   # file name of the grounding image as stored in $VIQUAE_IMAGES_PATH
   In [6]: item['image']
   Out[6]: '512px-Jackie_Wilson.png'

   # you can thus load the image from $VIQUAE_IMAGES_PATH/item['image']
   # meerqat.data.loading.load_image_batch does that for you
   In [7]: from meerqat.data.loading import load_image_batch
   # fake batch of size 1
   In [8]: image = load_image_batch([item['image']])[0]
   # it returns a PIL Image, all images have been resized to a width of 512
   In [9]: type(image), image.size
   Out[9]: (PIL.Image.Image, (512, 526))

   # question string
   In [10]: item['input']
   Out[10]: "this singer's re-issued song became the UK Christmas number one after helping to advertise what brand?"

   # answer string
   In [11]: item['output']['original_answer']
   Out[11]: "Levi's"

   # processing the data:
   In [12]: dataset.map(my_function)
   # this is almost the same as (see how can you adapt the code if you don’t want to use the `datasets` library)
   In [13]: for item in dataset:
       ...:     my_function(item)

The ViQuAE Knowledge Base (KB)
------------------------------

Again, the format of the KB is very similar to `KILT’s
Wikipedia <https://huggingface.co/datasets/kilt_wikipedia>`__ so I will
not describe all fields exhaustively.


.. code:: py

   # again you can also clone directly from https://huggingface.co/datasets/PaulLerner/viquae_wikipedia to get the raw data
   >>> data_files = dict(
       humans_with_faces='humans_with_faces.jsonl.gz', 
       humans_without_faces='humans_without_faces.jsonl.gz', 
       non_humans='non_humans.jsonl.gz'
   )
   >>> kb = load_dataset('PaulLerner/viquae_wikipedia', data_files=data_files)
   >>> kb
   DatasetDict({
       humans_with_faces: Dataset({
           features: ['anchors', 'categories', 'image', 'kilt_id', 'text', 'url', 'wikidata_info', 'wikipedia_id', 'wikipedia_title'],
           num_rows: 506237
       })
       humans_without_faces: Dataset({
           features: ['anchors', 'categories', 'image', 'kilt_id', 'text', 'url', 'wikidata_info', 'wikipedia_id', 'wikipedia_title'],
           num_rows: 35736
       })
       non_humans: Dataset({
           features: ['anchors', 'categories', 'image', 'kilt_id', 'text', 'url', 'wikidata_info', 'wikipedia_id', 'wikipedia_title'],
           num_rows: 953379
       })
   })
   >>> item = kb['humans_with_faces'][0]
   >>> item['wikidata_info']['wikidata_id'], item['wikidata_info']['wikipedia_title']
   ('Q313590', 'Alain Connes')
   # file name of the reference image as stored in $VIQUAE_IMAGES_PATH
   # you can use meerqat.data.loading.load_image_batch like above
   >>> item['image']
   '512px-Alain_Connes.jpg'
   # the text is stored in a list of string, one per paragraph
   >>> type(item['text']['paragraph']), len(item['text']['paragraph'])
   (list, 25)
   >>> item['text']['paragraph'][1]
   "Alain Connes (; born 1 April 1947) is a French mathematician, \
   currently Professor at the Collège de France, IHÉS, Ohio State University and Vanderbilt University. \
   He was an Invited Professor at the Conservatoire national des arts et métiers (2000).\n"
   # you might want to concatenate these three datasets to get a single dataset (e.g. to split the articles in passages)
   >>> from datasets import concatenate_datasets
   >>> kb['humans_with_faces'] = kb['humans_with_faces'].map(lambda item: {'is_human': True})
   >>> kb['humans_without_faces'] = kb['humans_without_faces'].map(lambda item: {'is_human': True})
   >>> kb['non_humans'] = kb['non_humans'].map(lambda item: {'is_human': False})
   >>> kb_recat = concatenate_datasets([kb['non_humans'], kb['humans_with_faces'], kb['humans_without_faces']])
   >>> kb_recat.save_to_disk('data/viquae_wikipedia_recat')

To format the articles into text passages, follow instructions at
`EXPERIMENTS.rst <./EXPERIMENTS.rst>`__ (Preprocessing passages section).
Alternatively, get them from https://huggingface.co/datasets/PaulLerner/viquae_v4-alpha_passages
(``load_dataset('PaulLerner/viquae_v4-alpha_passages')``).

Formatting WIT for multimodal ICT
=================================

WIT (Srinavasan et al. http://arxiv.org/abs/2103.01913) is available at https://github.com/google-research-datasets/wit.
(By any chance, if you have access to Jean Zay, it is available at ``$DSDIR/WIT``).

Follow instructions at ``meerqat.data.wit`` (see ``meerqat.data.wit.html``) or get it
from https://huggingface.co/datasets/PaulLerner/wit_for_mict (``load_dataset('PaulLerner/wit_for_mict')``)

Annotation of the ViQuAE data
=============================

Please refer to `ANNOTATION.md <./ANNOTATION.md>`__ for the
annotation instructions

Experiments
===========

Please refer to `EXPERIMENTS.rst <./EXPERIMENTS.rst>`__ for instructions
to reproduce our experiments

Reference
=========

If you use the ViQuAE dataset or KB, please cite:
::

   @inproceedings{lerner2022viquae,
      author = {Paul Lerner and Olivier Ferret and Camille Guinaudeau and Le Borgne, Hervé  and Romaric
      Besançon and Moreno, Jose G  and Lovón Melgarejo, Jesús },
      year={2022},
      title={{ViQuAE}, a
      Dataset for Knowledge-based Visual Question Answering about Named
      Entities},
      booktitle = {Proceedings of The 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
       series = {SIGIR’22},
      URL = {https://hal.archives-ouvertes.fr/hal-03650618},
      DOI = {10.1145/3477495.3531753},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA}
   }
   
If you use this code for multimodal information retrieval or early fusion or Inverse Cloze Task pre-training, please cite:
::

    @inproceedings{lerner2023ict,
      TITLE = {{Multimodal Inverse Cloze Task for Knowledge-based Visual Question Answering}},
      AUTHOR = {Lerner, Paul and Ferret, Olivier and Guinaudeau, Camille},
      URL = {https://hal.science/hal-03933089},
      BOOKTITLE = {{European Conference on Information Retrieval (ECIR 2023)}},
      ADDRESS = {Dublin, Ireland},
      YEAR = {2023},
      MONTH = Apr,
      KEYWORDS = {Visual Question Answering ; Pre-training ; Multimodal Fusion},
      PDF = {https://hal.science/hal-03933089v2/file/ecir-2023-vf-authors.pdf},
      HAL_ID = {hal-03933089},
      HAL_VERSION = {v2},
    }



If you use this code for mono- or cross-modal information retrieval with CLIP or fine-tuning CLIP, please cite:
::

    @unpublished{lerner2024cross,
      TITLE = {{Cross-modal Retrieval for Knowledge-based Visual Question Answering}},
      AUTHOR = {Lerner, Paul and Ferret, Olivier and Guinaudeau, Camille},
      URL = {https://hal.science/hal-04384431},
      NOTE = {Accepted at ECIR 2024},
      YEAR = {2024},
      MONTH = Jan,
      KEYWORDS = {Visual Question Answering ; Multimodal ; Cross-modal Retrieval ; Named Entities},
      PDF = {https://hal.science/hal-04384431/file/camera_ecir_2024_cross_modal_arXiv.pdf},
      HAL_ID = {hal-04384431},
      HAL_VERSION = {v1},
    }



Installation
============

Install PyTorch 1.9.0 following `the official document wrt to your
distribution <https://pytorch.org/get-started/locally/>`__ (preferably
in a virtual environment)

Also install
`ElasticSearch <https://www.elastic.co/fr/downloads/elasticsearch>`__
(and run it) or `pyserini <https://github.com/castorini/pyserini>`__ if you want to do sparse retrieval.

The rest should be installed using ``pip``:

.. code:: sh

   $ git clone https://github.com/PaulLerner/ViQuAE.git
   $ pip install -e ViQuAE
   $ python
   >>> import meerqat

Docs
====

`Read the docs! <https://paullerner.github.io/ViQuAE/meerqat.ir.search.html>`__

To build the docs locally, run ``sphinx-apidoc -o source_docs/ -f -e -M meerqat`` then ``sphinx-build -b html source_docs/ docs/``