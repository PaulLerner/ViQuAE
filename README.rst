``meerqat``
===========

Source code and data used in the papers:
    - `ViQuAE, a Dataset for Knowledge-based Visual Question Answering about Named Entities <https://hal.archives-ouvertes.fr/hal-03650618>`__ 
      (Lerner et al., SIGIR’22) 
    - "Multimodal Inverse Cloze Task for Knowledge-based Visual Question Answering" 
      (Lerner et al., submitted to ECIR 2023) TODO: add link or preprint.

See also `MEERQAT project <https://www.meerqat.fr/>`__.

Getting the ViQuAE dataset and KB
=================================

The data is provided in two formats: HF’s ``datasets`` (based on Apache
Arrow) and plain-text JSONL files (one JSON object per line). Both
formats can be used in the same way as ``datasets`` parses objects into
python ``dict`` (see below), however our code only supports (and is
heavily based upon) ``datasets``. Images are distributed separately, in
standard formats (e.g. jpg). Both dataset formats are distributed in two
versions, with (TODO) and without pre-computed features. The
pre-computed feature version allows you to skip one or several step
described in `EXPERIMENTS.rst <./EXPERIMENTS.rst>`__ (e.g. face
detection).

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

To format the articles into text passages, follow instructions at
`EXPERIMENTS.rst <./EXPERIMENTS.rst>`__ (Preprocessing passages section).
Alternatively, get them from
https://huggingface.co/datasets/PaulLerner/viquae_passages
(``load_dataset('PaulLerner/viquae_passages')``). 
FIXME: passages of 'PaulLerner/viquae_passages' contain one extra article (less than 10 passages)
compared to 'PaulLerner/viquae_wikipedia'. Experiments in MICT fixed this but indices of the provided
ViQuAE runs correspond to 'PaulLerner/viquae_passages' so they won’t match the new version.

Formatting WIT for multimodal ICT
=================================

WIT (Srinavasan et al. http://arxiv.org/abs/2103.01913) is available at https://github.com/google-research-datasets/wit.
(By any chance, if you have access to Jean Zay, it is available at ``$DSDIR/WIT``).

TODO add instructions to format WIT or upload the dataset to huggingface.

Images should be resized to have a maximum height or width of 512 pixels using ``meerqat.image.resize``.

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

If you use this code or the ViQuAE dataset, please cite our paper:

::

   @inproceedings{lerner2022,
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

Installation
============

Install PyTorch 1.9.0 following `the official document wrt to your
distribution <https://pytorch.org/get-started/locally/>`__ (preferably
in a virtual environment)

Also install
`ElasticSearch <https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html#install-elasticsearch>`__
(and run it) if you want to do sparse retrieval.

The rest should be installed using ``pip``:

.. code:: sh

   $ git clone https://github.com/PaulLerner/ViQuAE.git
   $ pip install -e ViQuAE
   $ python
   >>> import meerqat

Docs
====

TODO add readthedocs. Until then, have a look at ``docs/build/meerqat.html``

Building the docs: ``sphinx-build -b html docs/source/ docs/build/``
