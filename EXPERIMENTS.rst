Experiments
===========

All commands assume that the working directory is the root of the git
repo (e.g. same level as this file) and that the data is stored in the
``data`` folder, at the root of this repo (except for images for which
you can specify the ``VIQUAE_IMAGES_PATH`` environment variable).
Alternatively, you can change the paths in the config files.

Relevant configuration files can be found in the `experiments
directory <./experiments>`__. Expected output can be found in the
relevant subdirectory of ``experiments``.

We train the models based on lightning, itself based
on ``torch``. Even when not training models, all of our code is based on
``torch``.

Instructions specific to the ECIR-2023 Multimodal ICT paper are marked with "(MICT)",
while the instructions specific to the SIGIR ViQuAE dataset paper are marqued with "(ViQuAE)"
and those for the ECIR-2024 CLIP cross-modal paper are marked with "(Cross-modal)".
Note that, while face detection (MTCNN) and recognition (ArcFace) are not specific to ViQuAE,
they did not give promising results with MICT.


Preprocessing passages
----------------------

Splitting articles in passages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Articles are stripped of semi-structured data, such as tables and lists.
Each article is then split into disjoint passages of 100 words for text
retrieval, while preserving sentence boundaries, and the title of the
article is appended to the beginning of each passage.

.. code:: sh

   python -m meerqat.data.loading passages data/viquae_wikipedia_recat data/viquae_passages experiments/passages/config.json --disable_caching


Note that this will not match the ordering of https://huggingface.co/datasets/PaulLerner/viquae_v4-alpha_passages
which have been processed from a wikipedia version before splitting w.r.t. entity type
(such as ``kilt_wikipedia``).

Then you can extract some columns from the dataset to allow quick (and
string) indexing:

.. code:: sh

   python -m meerqat.data.loading map data/viquae_wikipedia_recat wikipedia_title title2index.json --inverse --disable_caching
   python -m meerqat.data.loading map data/viquae_wikipedia_recat passage_index article2passage.json --disable_caching

Find relevant passages in the linked wikipedia article
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This allows us to find the relevant passages for the question
(i.e. those than contain the answer or the alternative answers):

.. code:: sh

   python -m meerqat.ir.metrics relevant data/viquae_dataset data/viquae_passages data/viquae_wikipedia_recat/title2index.json data/viquae_wikipedia_recat/article2passage.json --disable_caching

Find relevant passages in the IR results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our clue that the passage is relevant for the answer is quite weak: it
contains the answer. That’s it. When scanning for the wikipedia article
of the entity (in ``meerqat.ir.metrics relevant``) you might find some
passages that contain the answer but have nothing to do with the
question. In order to tackle this, we use relevant passages that come
from the IR step in priority. Moreover, in this step (and it has no
impact on the evaluation) we only check for the *original answer* not
all alternative answers (which come from wikipedia aliases). Since this
step does not really fit in any of the modules and I cannot think of a
way of making it robust, I’ll just let you run it yourself from this
code snippet:

.. code:: py

   from datasets import load_from_disk, set_caching_enabled
   from meerqat.ir.metrics import find_relevant
   from ranx import Run
   
   set_caching_enabled(False)
   kb = load_from_disk('data/viquae_passages/')
   dataset = load_from_disk('data/viquae_dataset/train')
   # to reproduce the results of the papers:
   # - use DPR+Image as IR to train the reader or fine-tune ECA/ILF
   # - use BM25 as IR to train DPR (then save in 'BM25_provenance_indices'/'BM25_irrelevant_indices')
   run = Run.from_file('/path/to/bm25/or/multimodal_ir_train.trec')

   def keep_relevant_search_wrt_original_in_priority(item, kb):
       indices = list(map(int, run[item['id']]))
       relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
       if relevant_indices:
           item['search_provenance_indices'] = relevant_indices
       else:
           item['search_provenance_indices'] = item['original_answer_provenance_indices']
       item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
       return item
       
   dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
   dataset.save_to_disk('data/viquae_dataset/train')

Image
-----

This will be applied on both the QA dataset and the KB.

Global image embedding
~~~~~~~~~~~~~~~~~~~~~~

Obtained using ResNet-50:
 - one pre-trained on ImageNet, pooled with
   max-pooling. You can tweak the pooling layer and the backbone in the
   config file, as long as it is a ``nn.Module`` and
   ``torchvision.models``, respectively.
 - the other trained using
   `CLIP <https://github.com/openai/CLIP>`__ (install it from their repo)

The ViT version of CLIP is implemented with transformers.

Obviously you can also tweak the batch size.

.. code:: sh

   # embed dataset images with ImageNet-ResNet50
   python -m meerqat.image.embedding data/viquae_dataset experiments/image_embedding/imagenet/config.json --disable_caching
   # embed KB images with ImageNet-ResNet50
   python -m meerqat.image.embedding data/viquae_wikipedia experiments/image_embedding/imagenet/config.json --disable_caching
   # embed dataset images with CLIP-ResNet50
   python -m meerqat.image.embedding data/viquae_dataset experiments/image_embedding/clip/config.json --disable_caching
   # embed KB images with CLIP-ResNet50
   python -m meerqat.image.embedding data/viquae_wikipedia experiments/image_embedding/clip/config.json --disable_caching
   # embed dataset images with CLIP-ViT   
   python -m meerqat.image.embedding data/viquae_dataset experiments/image_embedding/clip/vit_config.json --disable_caching
   # embed KB images with CLIP-ViT
   python -m meerqat.image.embedding data/viquae_wikipedia experiments/image_embedding/clip/vit_config.json --disable_caching


To get a better sense of the representations the these model provide,
you can have a look at an interactive UMAP visualization, on 1% of the
KB images and the whole dataset images, w.r.t. the entity type,
`here <http://meerqat.fr/imagenet-viquae.html>`__ for ImageNet-ResNet50,
and `there <http://meerqat.fr/clip-viquae.html>`__ for CLIP-RN50 (takes a
while to load).

For WIT, you should change "save_as" and "image_key" in the config file by prepreding "context_"
so that it matches the data format and works with the trainer.


Text embedding (Cross-modal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of embedding the images of the knowledge base with CLIP, you can also embed its text,
e.g. the title of each article, to be able to then perform cross-modal retrieval, to reproduce
the results of the Cross-modal paper (ECIR 2024).

.. code:: sh

  python -m meerqat.ir.embedding data/viquae_wikipedia experiments/ir/viquae/clip/config.json


See below for an interactive visualization of (a subset of) the Wikipedia articles’ titles’ space
represented through CLIP (ViT-base, zero-shot) and reduced to 2D via UMAP.

.. raw:: html
   :file: ./source_docs/umap/title_clip-vit-base-patch32.html

The image is shown only for visualization purposes but the representation is text-only!

Face detection
~~~~~~~~~~~~~~

Things get a little more complicated here, first, you will want to split
your KB in humans and non-humans, since we assume that faces are not
relevant for non-human entities. I guess there’s no need to provide code
for that since it’s quite trivial and we will provide KB already split
in humans and non-humans.

Face detection uses MTCNN (Zhang et al., 2016) via the
``facenet_pytorch`` library. Feel free to tweak the hyperparameters (we
haven’t), you can also set whether to order faces by size or probability
(we do the latter)

Probabilities, bounding boxes and landmarks are saved directly in the
dataset, face croping happens as a pre-processing of Face recognition
(next section).

.. code:: sh

   python -m meerqat.image.face_detection data/viquae_dataset --disable_caching --batch_size=256
   python -m meerqat.image.face_detection data/viquae_wikipedia/humans --disable_caching --batch_size=256

After this you will also want to split the humans KB into humans with
detected faces and without.

Face recognition
~~~~~~~~~~~~~~~~

| Face recognition uses ArcFace (Deng et al., 2019) pre-trained on
  MS-Celeb (Guo et al., 2016) via the insightface ``arcface_torch``
  library. To be able to use ``arcface_torch`` as a library you will
  need to add an ``__init__`` and ``setup`` file in
  ``recognition/arcface_torch`` and ``recognition`` directories,
  respectively, like I did here:
  https://github.com/PaulLerner/insightface/commit/f159d90ce1dc620730c99e8a81991a7c5981dc3e
| Alternatively install it from my fork (or let me know how we are
  supposed to this cleanly :)

.. code:: sh

   git clone https://github.com/PaulLerner/insightface.git
   cd insightface
   git checkout chore/arcface_torch
   cd recognition
   pip install -e .

The pretrained ResNet-50 can be downloaded `from
here <https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC>`__
and the path to the backbone should be
``data/arcface/ms1mv3_arcface_r50_fp16/backbone.pth``

The 5 face landmarks (two eyes, nose and two mouth corners) are adopted
to perform similarity transformation so that they are always at the same
position in the image, regardless of the original pose of the person.
This is done with the ``similarity_transform`` function using
``skimage`` and ``cv2``.

You can tweak the backbone and the batch size, we only tried with
ResNet-50 (note there’s an extra layer compared to the ImageNet one
which pools the embedding dimension down to 512).

Finally we can run it!

.. code:: sh

   python -m meerqat.image.face_recognition data/viquae_dataset experiments/face_recognition/config.json --disable_caching
   python -m meerqat.image.face_recognition data/viquae_wikipedia/humans_with_faces experiments/face_recognition/config.json --disable_caching

You can tweak the number of faces in the config file. We used 4 for MICT experiments.
To reproduce ViQuAE experiments, you will want to consider only the most probable face so do something like:

.. code:: py

    d = load_from_disk('data/viquae_dataset')
    d = d.map(lambda item: {'first_face_embedding': item['face_embedding'][0] if item['face_embedding'] is not None else None})
    d.save_to_disk('data/viquae_dataset')

Again, you can have a look at an `interactive UMAP
visualization <http://meerqat.fr/arcface-viquae.html>`__ (takes a while
to load), trained on the whole KB faces (but displaying only 10K to get
a reasonable HTML size).

Bounding box engineering (MICT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Again, this is provided for the sake of archival but does not provide better results
than MICT models based on CLIP only (no faces).

We follow UNITER (Chen et al.) and represent bounding box features like:
:math:`(x_1, y_1, x_2, y_2, w, h, a)`, where :math:`(x_1, y_1)` and :math:`(x_2, y_2)`
are the top-left and bottom-right coordinates, respectively, both scaled between [0, 1],
:math:`w = x_2-x_1` is the width,  :math:`h = y_2-y_1` is the height, and :math:`a = w \times h` is the area.

To achieve this, simply run: ``meerqat.image.face_box <dataset>``.
Be sure to run it **after** ``meerqat.image.face_recognition`` since it scales bounding boxes and landmarks to [0, 1].

Training dual encoders (e.g. DPR)
---------------------------------
DPR
~~~

We use the same hyperparameters as `Karpukinh et
al. <https://github.com/facebookresearch/DPR>`__. We train DPR using 4
V100 GPUs of 32GB, allowing a total batch size of 256 (32 questions \* 2
passages each \* 4 GPUs). This is crucial because each question uses all
passages paired with other questions in the batch as negative examples.
Each question is paired with 1 relevant passage and 1 irrelevant passage
mined with BM25.

Both the question and passage encoder are initialized from
``"bert-base-uncased"``.


Pre-training on TriviaQA
^^^^^^^^^^^^^^^^^^^^^^^^

You can skip this step and use our pre-trained models: 
    - question model: https://huggingface.co/PaulLerner/dpr_question_encoder_triviaqa_without_viquae
    - context/passage model: https://huggingface.co/PaulLerner/dpr_context_encoder_triviaqa_without_viquae

To be used with ``transformers``'s ``DPRQuestionEncoder`` and
``DPRContextEncoder``, respectively.

Given the small size of ViQuAE, DPR is pre-trained on TriviaQA: 
    - filtered out of all questions used for ViQuAE for training 
    - on questions used to generate ViQuAE’s validation set for validation

Get TriviaQA with these splits from:
https://huggingface.co/datasets/PaulLerner/triviaqa_for_viquae (or
``load_dataset("PaulLerner/triviaqa_for_viquae")``)

In this step we use the complete ``kilt_wikipedia`` instead of
``viquae_wikipedia``.

``python -m meerqat.train.trainer fit --config=experiments/dpr/triviaqa/config.yaml``

The best checkpoint should be at step 13984.

Fine-tuning on ViQuAE
^^^^^^^^^^^^^^^^^^^^^

We use exactly the same hyperparameters as for pre-training.

Once you’ve decided on a TriviaQA checkpoint, (step 13984 in our case) 
you need to split it in two with ``python -m meerqat.train.save_ptm experiments/dpr/triviaqa/config.yaml experiments/dpr/triviaqa/lightning_logs/version_0/step=13984.ckpt``, 
then set the path as in the provided config file.
**Do not** simply set "--ckpt_path=/path/to/triviaqa/pretraing" else
the trainer will also load the optimizer and other training stuffs.

Alternatively, if you want to start training from our pre-trained model,
set "PaulLerner/dpr_question_encoder_triviaqa_without_viquae" and "PaulLerner/dpr_context_encoder_triviaqa_without_viquae"
in the config file.

``python -m meerqat.train.trainer fit --config=experiments/dpr/viquae/config.yaml``

The best checkpoint should be at step 40. Run
``python -m meerqat.train.save_ptm experiments/dpr/viquae/config.yaml experiments/dpr/viquae/lightning_logs/version_0/step=40.ckpt``
to split DPR in a DPRQuestionEncoder and DPRContextEncoder. We’ll use
both to embed questions and passages below.


Multimodal Inverse Cloze Task (MICT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starting from DPR training on TriviaQA, we will train ECA and ILF for MICT on WIT.

Unlike the above DPR pre-training, here we use a single NVIDIA V100 GPU with 32 GB of RAM,
but using gradient checkpointing.

Alternatively, use the provided pre-trained models following instructions below.

ILF
^^^
Notice how ILF fully freezes BERT during this stage with the regex ``".*dpr_encoder.*"``
``python -m meerqat.train.trainer fit --config=experiments/ict/ilf/config.yaml``

Pre-trained models available:
 - https://huggingface.co/PaulLerner/question_ilf_l12_wit_mict
 - https://huggingface.co/PaulLerner/context_ilf_l12_wit_mict


ECA
^^^
ECA uses internally ``BertModel`` instead of ``DPR*Encoder`` so you need to run
``meerqat.train.save_ptm`` again, this time with the ``--bert`` option.

Again, notice how the last six layers of BERT are frozen thanks to the regex.

``python -m meerqat.train.trainer fit --config=experiments/ict/eca/config.yaml``

Pre-trained models available:
 - https://huggingface.co/PaulLerner/question_eca_l6_wit_mict
 - https://huggingface.co/PaulLerner/context_eca_l6_wit_mict


As a sanity check, you can check the performance of the models on WIT’s test set.

``python -m meerqat.train.trainer test --config=experiments/ict/ilf/config.yaml``
``python -m meerqat.train.trainer test --config=experiments/ict/eca/config.yaml``


Fine-tuning multimodal models on ViQuAE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Almost the same as for DPR although some hyperparameters change, notably the model used
to mine negative passage is here set as the late fusion of arcface, imagenet, clip, and dpr.
We have tried to fine-tune DPR with the same hyperparameters and found no significant difference.
Notice also that now we need a second KB that holds the pre-computed image features (viquae_wikipedia_recat)

You can use the provided test config to split the BiEncoder:
``python -m meerqat.train.save_ptm experiments/ict/ilf/config.yaml experiments/ict/ilf/lightning_logs/version_0/step=15600.ckpt``

``python -m meerqat.train.save_ptm experiments/ict/eca/config.yaml experiments/ict/eca/lightning_logs/version_0/step=8200.ckpt``

If you want to start from the pre-trained models we provide, use ``"PaulLerner/<model>"`` in the config files,
e.g. ``"question_model_name_or_path": "PaulLerner/question_eca_l6_wit_mict"``

Notice that all layers of the model are trainable during this stage.

``python -m meerqat.train.trainer fit --config=experiments/mm/ilf/config.yaml``
``python -m meerqat.train.trainer fit --config=experiments/mm/eca/config.yaml``

Once fine-tuning is done, save the PreTrainedModel using the same command as above.


Fine-tuning CLIP for image retrieval (Cross-modal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reproduce the results of the Cross-modal paper (ECIR 24), fine-tune CLIP so that images of ViQuAE
are closer to the name of the depicted entity! 

``python -m meerqat.train.trainer fit --config=experiments/jcm/config.yaml``

TODO for mono-modal fine-tuning, you should add the corresponding image of the entity in the KB as "wikipedia_image" in the dataset.

IR
--

Now that we have a bunch of dense representations, let’s see how to
retrieve information! Dense IR is done with ``faiss`` and sparse IR is
done with ``elasticsearch``, both via HF ``datasets``. We’ll use IR on
both TriviaQA along with the complete Wikipedia (BM25 only) and ViQuAE
along with the multimodal Wikipedia.

Hyperparameter tuning is done using grid search via ``ranx`` on the
dev set to maximize MRR.

Note that the indices/identifiers of the provided runs and qrels match https://huggingface.co/datasets/PaulLerner/viquae_v4-alpha_passages


BM25 (ViQuAE)
~~~~~~~~~~~~~

Before running any of the commands below you should `launch the Elastic
Search
server <https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html#install-elasticsearch>`__.
Alternatively, if you're using pyserini instead of elasticsearch, follow those instructions: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation

First you might want to optimize BM25 hyperparameters, ``b`` and
``k_1``. We did this with a grid-search using ``optuna``: the ``--k``
option asks for the top-K search results.

.. code:: sh

   python -m meerqat.ir.hp bm25 data/viquae_dataset/validation experiments/ir/viquae/hp/bm25/config.json --k=100 --disable_caching --test=data/viquae_dataset/test --metrics=experiments/ir/viquae/hp/bm25/metrics

Alternatively, you can use the parameters we optimized: ``b=0.3`` and
``k_1=0.5``:

.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/bm25/config.json --k=100 --metrics=experiments/ir/viquae/bm25/metrics --disable_caching

Note that, in this case, we set ``index_kwargs.BM25.load=True`` to
re-use the index computed in the previous step.

DPR
~~~

Embedding questions and passages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: sh

   python -m meerqat.ir.embedding data/viquae_dataset experiments/ir/viquae/dpr/questions/config.json --disable_caching
   python -m meerqat.ir.embedding data/viquae_passages experiments/ir/viquae/dpr/passages/config.json --disable_caching

Searching
^^^^^^^^^

Like with BM25:

.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/dpr/search/config.json --k=100 --metrics=experiments/ir/viquae/dpr/search/metrics --disable_caching

ImageNet-ResNet and CLIP vs ArcFace-MS-Celeb (ViQuAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*Do not do this for MICT, we want all representations for all images, 
or use the ``face_and_image_are_exclusive`` option in the config file of the model*

We trust the face detector, if it detects a face then: 
 - the search is done on the human faces KB (``data/viquae_wikipedia/humans_with_faces``)

else:
 - the search is done on the non-human global images KB (``data/viquae_wikipedia/non_humans``)

To implement that we simply set the global image embedding to None when
a face was detected:

.. code:: py

   from datasets import load_from_disk, set_caching_enabled
   set_caching_enabled(False)
   dataset = load_from_disk('data/viquae_dataset/')
   dataset = dataset.rename_column('imagenet-RN50', 'keep_imagenet-RN50')
   dataset = dataset.rename_column('clip-RN50', 'keep_clip-RN50')
   dataset = dataset.map(lambda item: {'imagenet-RN50': item['keep_imagenet-RN50'] if item['face_embedding'] is None else None})
   dataset = dataset.map(lambda item: {'clip-RN50': item['keep_clip-RN50'] if item['face_embedding'] is None else None})
   dataset.save_to_disk('data/viquae_dataset/')

Search is done using cosine distance, hence the ``"L2norm,Flat"`` for
``string_factory`` and ``metric_type=0`` (this does first
L2-normalization then dot product).

The results, corresponding to a KB entity/article are then mapped to the
corresponding passages to allow fusion with BM25/DPR (next §)

Late fusion
~~~~~~~~~~~

Now in order to combine the text results of text and the image results
we do two things: 
1. normalize the scores so that they have zero-mean and unit variance 
2. combine text and image score through a weighted sum for each passage before
re-ordering, note that if only the text finds a given passage then its
image score is set to the minimum of the image results (and vice-versa)

The results are then re-ordered before evaluation. Interpolation hyperparameters are tuned using ranx.

BM25 + ArcFace + CLIP + ImageNet (ViQuAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tune hyperparameters
''''''''''''''''''''

``python -m meerqat.ir.search data/viquae_dataset/validation experiments/ir/viquae/bm25+arcface+clip+imagenet/config_fit.json --k=100 --disable_caching``

Run with the best hyperparameters
'''''''''''''''''''''''''''''''''


.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/bm25+arcface+clip+imagenet/config_test.json --k=100 --metrics=experiments/ir/viquae/bm25+arcface+clip+imagenet/metrics

DPR + ArcFace + CLIP + ImageNet (ViQuAE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Same script, different config.

.. _tune-hyperparameters-1:

Tune hyperparameters
''''''''''''''''''''

``python -m meerqat.ir.search data/viquae_dataset/validation experiments/ir/viquae/dpr+arcface+clip+imagenet/config_fit.json --k=100 --disable_caching``

.. _run-with-the-best-hyperparameters-1:

Run with the best hyperparameters
'''''''''''''''''''''''''''''''''

.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/dpr+arcface+clip+imagenet/config_test.json --k=100 --metrics=experiments/ir/viquae/dpr+arcface+clip+imagenet/metrics


Once search is done and results are saved in a Ranx Run, you can experiment more fusion techniques
(on the validation set first!) using ``meerqat.ir.fuse``


DPR + CLIP (MICT)
^^^^^^^^^^^^^^^^^
For the late fusion baseline based only on DPR and CLIP, be sure to use CLIP on all images
and do **not** run what’s above that sets CLIP=None when a face is detected.

Then, you can do the same as above using ``experiments/ir/viquae/dpr+clip/config.json``

Early Fusion (MICT)
~~~~~~~~~~~~~~~~~~~
Embedding visual questions and visual passages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Much like for DPR, you first need to split the BiEncoder in two once you picked a checkpoint using
``meerqat.train.save_ptm``. Then, set its path like in the provided config file.

The important difference with DPR here, is again that you need to pass viquae_wikipedia
which holds pre-computed image features of the visual passages.


.. code:: sh

   python -m meerqat.ir.embedding data/viquae_dataset experiments/ir/viquae/ilf/embedding/dataset_config.json
   python -m meerqat.ir.embedding data/viquae_passages experiments/ir/viquae/ilf/embedding/kb_config.json --kb=data/viquae_wikipedia_recat
   python -m meerqat.ir.embedding data/viquae_dataset experiments/ir/viquae/eca/embedding/dataset_config.json
   python -m meerqat.ir.embedding data/viquae_passages experiments/ir/viquae/eca/embedding/kb_config.json --kb=data/viquae_wikipedia_recat

Searching
^^^^^^^^^
This is exactly the same as for DPR, simply change "key" and "column" to "ILF_few_shot" or "ECA_few_shot".


Cross-modal CLIP
~~~~~~~~~~~~~~~~
Again using ``meerqat.ir.search`` but this time, using also the cross-modal search of CLIP,
and not only the monomodal search! CLIP can be optionally fine-tuned as explained above.

.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/dpr+clip-cross-modal/config_test.json --k=100 --metrics=experiments/ir/viquae/dpr+clip-cross-modal/


Metrics
~~~~~~~

We use `ranx <https://github.com/AmenRa/ranx>`__ to compute the metrics.
I advise against using any kind of metric that uses recall (mAP,
R-Precision, …) since we estimate relevant document on the go so the
number of relevant documents will *depend on the systemS* you use.

To compare different models (e.g. BM25+Image and DPR+Image), you should:
    - fuse the qrels (since relevant passages are estimated based on the
      model’s output):
      ``python -m meerqat.ir.metrics qrels <qrels>... --output=experiments/ir/all_qrels.json``
    - ``python -m meerqat.ir.metrics ranx <run>... --qrels=experiments/ir/all_qrels.json --output=experiments/ir/comparison``

ViQuAE results
^^^^^^^^^^^^^^
Beware that the ImageNet-ResNet and ArcFace results cannot be compared,
neither between them nor with BM25/DPR because:
 - they are exclusive, roughly **half** the questions have a face -> ArcFace, other don’t ->
   ResNet, while BM25/DPR is applied to **all** questions
 - the mapping from image/document to passage is arbitrary, so the ordering of image
   results is not so meaningful until it is re-ordered with BM25/DPR

If you’re interested in comparing only image representations, leaving
downstream performance aside (e.g. comparing ImageNet-Resnet with
another representation for the full image), you should:
 - ``filter`` the dataset so that you don’t evaluate on irrelevant questions (e.g. those
   were the search is done with ArcFace because a face was detected)
 - evaluate at the *document-level* instead of passage-level as in the Cross-modal paper (ECIR 24)
 
See the following instructions.

Cross-modal results 
^^^^^^^^^^^^^^^^^^^
To reproduce the article-level results of the Cross-modal paper (ECIR 24), you can use a config very similar to
``experiments/ir/viquae/dpr+clip-cross-modal/config_test.json`` although the results 
will **not** be mapped to corresponding passage indices, and the relevance of the article
will be evaluated directly through the "document" ``reference_key``:

.. code:: sh

   python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/clip/article_config.json --k=100 --metrics=experiments/ir/viquae/clip/


You can use the same method to evaluate other article-level representations, 
e.g. ArcFace, ImageNet-ResNet, BM25…


Reading Comprehension
---------------------


Now we have retrieved candidate passages, it’s time to train a Reading
Comprehension system (reader). We first pre-train the reader on TriviaQA
before fine-tuning it on ViQuAE. Our model is based on Multi-Passage
BERT (Wang et al., 2019), it simply extends the BERT fine-tuning for QA
(Devlin et al., 2019) with the global normalization by Clark et. al
(2018), i.e. all passages are processed independently but share the same
softmax normalization so that scores can be compared across passages.
The model is implemented in ``meerqat.train.qa`` it inherits from
HF ``transformers.BertForQuestionAnswering`` and the implementation is
based on DPR (Karpukhin et al., 2020)

We convert the model start and end answer position probabilities to
answer spans in ``meerqat.models.qa.get_best_spans``. The answer span
probabilities can be weighed with the retrieval score, which is ensured
to be > 1. We also enforce that the start starts before the end and that
the first token (``[CLS]``) cannot be the answer since it’s the
objective for irrelevant passages (this is the default behavior but can
be changed with the ``cannot_be_first_token`` flag).

.. _pre-training-on-triviaqa-1:

Pre-training on TriviaQA (ViQuAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to skip this step you can get our pretrained model at
https://huggingface.co/PaulLerner/multi_passage_bert_triviaqa_without_viquae_mean_pool_loss

Our training set consists of questions that were not used to generate
any ViQuAE questions, even those that were discarded or remain to be
annotated. Our validation set consists of the questions that were used
to generate ViQuAE validation set. Get TriviaQA with these splits from:
https://huggingface.co/datasets/PaulLerner/triviaqa_for_viquae (or
``load_dataset("triviaqa_for_viquae")``)

We used the same hyperparameters as Karpukhin et al. except for the
ratio of relevant passages: We use 8 relevant and 16 irrelevant passages
(so 24 in total) per question (the intuition was to get a realistic
precision@24 score w.r.t. the search results, we haven’t tried any other
setting). The model is trained to predict the first token (``[CLS]``) as
answer for irrelevant passages.

-  ``max_n_answers``: the model is trained to predict all off the
   positions of the answer in the passage up to this threshold
-  ``train_original_answer_only``: use in conjunction with the above
   preprocessing, defaults to True

As with DPR, IR is then carried out with BM25 on the full 5.9M articles
of KILT’s Wikipedia instead of our multimodal KB.

.. code:: sh

   python -m meerqat.train.trainer fit --config=experiments/rc/triviaqa/config.json

The best checkpoint should be at step 21000.

.. _fine-tuning-on-viquae-1:

Fine-tuning on ViQuAE (ViQuAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, use ``meerqat.train.save_ptm`` on the best checkpoint and set it
as pre-trained model instead of ``bert-base-uncased``
(``PaulLerner/multi_passage_bert_triviaqa_without_viquae_mean_pool_loss`` to use ours).

Then you can fine-tune the model:

.. code:: sh

   python -m meerqat.train.trainer fit --config=experiments/rc/viquae/config.yaml

The best checkpoint should be at step 894. This run uses the
default seed in ``transformers``: 42. To have multiple runs, like in the
paper, set ``seed_everything: <int>`` in the config. We used
seeds ``[0, 1, 2, 3, 42]``. The expected output provided is with
``seed=1``.


.. code:: sh

   python -m meerqat.train.trainer test --config=experiments/rc/viquae/config.yaml --ckpt_path=experiments/rc/viquae/version_1/checkpoints/step=894.ckpt


To reproduce the oracle results: 

- for “full-oracle”, simply add the ``oracle: true`` flag in the config file and set
  ``n_relevant_passages: 24`` 

- for “semi-oracle”, in addition you should
  filter ``search_provenance_indices`` like above but setting
  ``item['search_provenance_indices'] = []`` when no relevant passages
  where retrieved by the IR system.

Switching IR inputs at inference (MICT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simply set ``run_path:"/path/to/run.trec"`` in experiments/rc/viquae/config.yaml
and run ``meerqat.train.trainer test`` again.


Note on the loss function
~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-passage BERT is trained to independently predict the start and the end of the answer span in the passages. 
At inference, the probability of the answer span being [i:j] 
is the product of the start being i and the end being j. 

Crucially, in Multi-passage BERT, all K passages related to a question share the same softmax normalization. 
The answer can appear up to R times in the same passage. 
Therefore, the objective should be, given a_kl the score predicted for the answer to start 
(the reasoning is analogous for the end of the answer span) at the l-th token of the k-th passage:

.. math::

    -\sum_{r=1}^R \sum_{k=1}^{K}\sum_{l=1}^{L} y_{rkl} \log{\frac{\exp{(a_{kl})}}{\sum_{k'=1}^{K}\sum_{l'=1}^{L}\exp{(a_{k'l'})}}}
    
Where :math:`y_{rkl} \in \{0,1\}` denotes the ground truth.

However, our initial implementation, therefore the results of the ViQuAE and MICT papers,
was based on Karpukhin's DPR who implemented:


.. math::

    -\sum_{r=1}^R \max_{k=1}^{K}\sum_{l=1}^{L} y_{rkl} \log{\frac{\exp{(a_{kl})}}{\sum_{k'=1}^{K}\sum_{l'=1}^{L}\exp{(a_{k'l'})}}}
    
I've opened `an issue <https://github.com/facebookresearch/DPR/issues/244>`__ on Karpukhin's DPR repo
but did not get an answer. This initial max-pooling is still mysterious to me.

Anyway, that explains the difference between v3-alpha and v4-alpha, and, as a consequence,
between the ViQuAE/MICT papers and the cross-modal paper (ECIR 2024).




References
==========
TODO use links between main text and references

Chen, Y.C., Li, L., Yu, L., El Kholy, A., Ahmed, F., Gan, Z., Cheng, Y., Liu, J.:
Uniter: Universal image-text representation learning. In: European Conference on
Computer Vision. pp. 104–120. https://openreview.net/forum?id=S1eL4kBYwr. Springer (2020)
        
Christopher Clark and Matt Gardner. 2018. Simple and Effective
Multi-Paragraph Reading Comprehension. In Proceedings of the 56th Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 845–855, Melbourne, Australia. Association for
Computational Linguistics.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. 2019.
ArcFace: Additive Angular Margin Loss for Deep Face Recognition. pages
4690–4699. 

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding. arXiv:1810.04805 [cs]. ArXiv: 1810.04805.

Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao. 2016.
MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition.
In Computer Vision – ECCV 2016, Lecture Notes in Computer Science, pages
87–102, Cham. Springer International Publishing.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu,
Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage
Retrieval for Open-Domain Question Answering. In Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP),
pages 6769-6781. Https://github.com/facebookresearch/DPR.

Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallap- ati, and Bing Xiang.
2019. Multi-passage BERT: A Globally Normalized BERT Model for Open-
domain Question Answering. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 5878–5882, Hong Kong, China. Association for
Computational Linguistics.

Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and Yu Qiao. 2016. Joint Face
Detection and Alignment Using Multitask Cascaded Convolutional Networks.
IEEE Signal Processing Letters, 23(10):1499–1503. Conference Name: IEEE
Signal Processing Letters.
