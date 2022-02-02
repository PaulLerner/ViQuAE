# Experiments

All of the intermediate outputs of the pipeline should be provided along with the data 
so that people are free to skip one or few steps. (TODO add instructions here or in the README data section)

All commands assume that the working directory is the root of the github repo (e.g. same level as this file).
Relevant configuration files can be found in the [experiments directory](./experiments).

**Table of contents**
* [Image](#image)
    + [Global image embedding](#global-image-embedding)
    + [Face detection](#face-detection)
    + [Face recognition](#face-recognition)
* [IR](#ir)
    + [Preprocessing passages](#preprocessing-passages)
    + [BM25](#bm25)
    + [ResNet-ImageNet + ArcFace-MS-Celeb](#resnet-imagenet---arcface-ms-celeb)
    + [BM25 + Image](#bm25---image)
    + [Metrics](#metrics)
* [Reading Comprehension](#reading-comprehension)
    + [Pre-processing](#pre-processing)
    + [Pre-training on TriviaQA](#pre-training-on-triviaqa)
    + [Fine-tuning on MEERQAT](#fine-tuning-on-meerqat)
* [References](#references)


## Image
This will be applied on both the QA dataset and the KB.  

### Global image embedding

Obtained using a ResNet-50 pre-trained on ImageNet, pooled with max-pooling.
You can tweak the pooling layer and the backbone in the config file, as long as it is a `nn.Module` and `torchvision.models`, respectively.
Obviously you can also tweak the batch size.
```sh
python -m meerqat.image.embedding data/meerqat_dataset experiments/image_embedding/config.json --disable_caching
python -m meerqat.image.embedding data/meerqat_wikipedia experiments/image_embedding/config.json --disable_caching
```

### Face detection

Things get a little more complicated here, first, you will want to split your KB in humans and non-humans,
since we assume that faces are not relevant for non-human entities.
I guess there’s no need to provide code for that since it’s quite trivial and we will provide KB already split in humans and non-humans.

Face detection uses MTCNN (Zhang et al., 2016) via the `facenet_pytorch` library.
Feel free to tweak the hyperparameters (we haven’t), you can also set whether to order faces by size or probability (we do the latter)
 
Probabilities, bounding boxes and landmarks are saved directly in the dataset, face croping happens as a pre-processing of Face recognition (next section).

```sh
python -m meerqat.image.face_detection data/meerqat_dataset --disable_caching --batch_size=256
python -m meerqat.image.face_detection data/meerqat_wikipedia/humans --disable_caching --batch_size=256
```

After this you will also want to split the humans KB into humans with detected faces and without.

### Face recognition

Face recognition uses ArcFace (Deng et al., 2019) pre-trained on MS-Celeb (Guo et al., 2016) via the insightface `arcface_torch` library.
To be able to use `arcface_torch` as a library you will need to add an `__init__` and `setup` file in `recognition/arcface_torch` and `recognition` directories, respectively, 
like I did here: https://github.com/PaulLerner/insightface/commit/f159d90ce1dc620730c99e8a81991a7c5981dc3e  
Alternatively install it from my fork (or let me know how we are supposed to this cleanly :)
```sh
git clone https://github.com/PaulLerner/insightface.git
cd insightface
git checkout chore/arcface_torch
cd recognition
pip install -e .
``` 

The pretrained ResNet-50 can be downloaded [from here](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215583&cid=4A83B6B633B029CC)
and the path to the backbone should be `data/arcface/ms1mv3_arcface_r50_fp16/backbone.pth` 

The 5 face landmarks (two eyes, nose and two mouth corners) are adopted to perform similarity transformation 
so that they are always at the same position in the image, regardless of the original pose of the person.
This is done with the `similarity_transform` function using `skimage` and `cv2`.

You can tweak the backbone and the batch size, we only tried with ResNet-50 
(note there’s an extra layer compared to the ImageNet one which pools the embedding dimension down to 512).

Finally we can run it!
```sh
python -m meerqat.image.face_recognition data/meerqat_dataset experiments/face_detection/meerqat_dataset experiments/face_recognition/config.json --disable_caching
python -m meerqat.image.face_recognition data/meerqat_wikipedia/humans_with_faces experiments/face_detection/meerqat_wikipedia experiments/face_recognition/config.json --disable_caching
```

## IR

Now that we have a bunch of dense representations, let’s see how to retrieve information!
Dense IR is done with `faiss` and sparse IR is done with `elasticsearch`, both via HF `datasets`.
We’ll use IR on both TriviaQA along with the complete Wikipedia (BM25 only) and MEERQAT along with the multimodal Wikipedia.

### Preprocessing passages
You can probably skip this step as we will provide passages dataset along with provenance.

Articles are stripped of semi-structured data, such as tables and lists. 
Each article is then split into disjoint passages of 100 words for text retrieval, while preserving sentence boundaries, 
and the title of the article is appended to the beginning of each passage.

```sh
python -m meerqat.data.loading passages data/meerqat_wikipedia data/meerqat_passages experiments/passages/config.json --disable_caching
```

Then you can extract some columns from the dataset to allow quick (and string) indexing:
```sh
python -m meerqat.data.loading map data/meerqat_wikipedia wikipedia_title title2index.json --inverse --disable_caching
python -m meerqat.data.loading map data/meerqat_wikipedia passage_index article2passage.json --disable_caching
```

This allows us to find the relevant passages for the question (i.e. those than contain the answer or the alternative answers):
```sh
python -m meerqat.ir.metrics relevant data/meerqat_dataset data/meerqat_passages data/meerqat_wikipedia/title2index.json data/meerqat_wikipedia/article2passage.json --disable_caching
```

### BM25
Before running any of the commands below you should [launch the Elastic Search server](https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html#install-elasticsearch).

First you might want to optimize BM25 hyperparameters, `b` and `k_1`.
We did this with a grid-search using `optuna`:
the `--k` option asks for the top-K search results.  
```sh
python -m meerqat.ir.search hp bm25 data/meerqat_dataset/validation experiments/ir/meerqat/bm25/config.json --k=100 --metrics=experiments/ir/meerqat/bm25/metrics.json --test=data/meerqat_dataset/test --disable_caching
```

Alternatively, you can use the parameters we optimized: `b=0.3` and `k_1=0.5` using the same config file but without the `study_name` and `storage` fields:
```sh
python -m meerqat.ir.search data/meerqat_dataset/test experiments/ir/meerqat/bm25/config.json --k=100 --metrics=experiments/ir/meerqat/bm25/metrics.json --disable_caching
```

You can also use the `--save_irrelevant` option if you want to save the irrelevant search results 
along with the union of relevant search results and the output of `meerqat.ir.metrics relevant`.
However in this case we take into account all alternative answers so this is useless if you want to exactly reproduce our reader experiments (see §[Reading Comprehension](#Reading Comprehension))

### ResNet-ImageNet + ArcFace-MS-Celeb
We trust the face detector, if it detects a face then:
- the search is done on the human faces KB (`data/meerqat_wikipedia/humans_with_faces`)

else:
- the search is done on the non-human global images KB (`data/meerqat_wikipedia/non_humans`)

To implement that we simply set the global image embedding to None when a face was detected:
```py
from datasets import load_from_disk, set_caching_enabled
set_caching_enabled(False)
dataset = load_from_disk('data/meerqat_dataset/')
dataset = dataset.rename_column('image_embedding', 'keep_image_embedding')
dataset = dataset.map(lambda item: {'image_embedding': item['keep_image_embedding'] if item['face_embedding'] is None else None})
dataset.save_to_disk('data/meerqat_dataset/')
```

Search is done using cosine distance, hence the `"L2norm,Flat"` for `string_factory` and `metric_type=0`
(this does first L2-normalization then dot product).

The results, corresponding to a KB entity/article are then mapped to the corresponding passages to allow fusion with BM25 (next §)
The image results is simply the union of both ResNet and ArcFace scores since they are exclusive 
(you can tweak that with the `weight` parameter in the config file)

Beware that these results cannot be well interpreted because they do not have a reference KB which would allow looking if the answer is in the passage (TODO refactor to allow that). See other caveats in the next section.

### BM25 + Image

Now in order to combine the text results of BM25 and the image results we do two things:
1. normalize the scores so that they have zero-mean and unit variance, 
   the mean and the variance is computed over the whole subset so you might want to do a dry run first or use ours
2. sum the text and image score for each passage before re-ordering, note that
   if only the text finds a given passage then its image score is set to the minimum of the image results (and vice-versa)

The results are then re-ordered before evaluation.

You can tweak the fusion settings: `alpha` weighs on the image score when doing `fusion = text + alpha*image`.
We’ve tried optimizing it but it converged to `alpha=1` so I won’t spend too much time on it,
if you want to optimize it you can do `meerqat.ir.search hp fusion` but you will have to change the config file
(and run the search at least once before so that you save BM25 and image results)

```sh
python -m meerqat.ir.search data/meerqat_dataset/test experiments/ir/meerqat/bm25+image/config.json --k=100 --metrics=experiments/ir/meerqat/bm25+image/metrics.json
```

Beware that the ImageNet-ResNet and ArcFace results cannot be compared, neither between them nor with BM25 because:
- they are exclusive, roughly **half** the questions have a face -> ArcFace, other don't -> ResNet, while BM25 is applied to **all** questions
- the mapping from image/document to passage is arbitrary, so the ordering of image results is not so meaningful until it is re-ordered with BM25

If you’re interested in comparing only image representations, leaving downstream performance aside (e.g. comparing ImageNet-Resnet with another representation for the full image), you should:
- `filter` the dataset so that you don’t evaluate on irrelevant questions (e.g. those were the search is done with ArcFace because a face was detected)
- evaluate at the *document-level* instead of passage-level. To do so, maybe `checkout` the `document` branch (TODO merge in `main`).

### Metrics
We compute metrics for different top-K results, noted `<metric>@<K>`.

Given `relret@K = |retrieved@K & relevant|` and `R = |relevant|`:
- `precision@K = relret@K/K`
- `recall@K = relret@K/R`
- `hits@K = min(1, relret@K)`, 
   equivalent to `precision@K` when `K=1` and to `recall@K` when `R=1` (i.e. there is only one relevant passage)
- `MRR = 1/rank` where `rank` is the rank of the *first* relevant passage retrieved
- `R-precision = relret@R/R`

Results are then averaged over all queries, we do not take into account queries without any relevant passages

A few notes:
- We min out `R = min(R, k)`, where `k` is the number of results you asked with `--k`
- Recall and R-precision should be interpreted carefully since we do not have a complete coverage of the relevant passages
- MRR is a bit underestimated since we cut off at `k` results

## Reading Comprehension

Now we have retrieved candidate passages, it’s time to train a Reading Comprehension system (reader).
We first pre-train the reader on TriviaQA before fine-tuning it on MEERQAT.
Our model is based on Multi-Passage BERT (Wang et al., 2019), it simply extends the BERT fine-tuning for QA (Devlin et al., 2019)
with the global normalization by Clark et. al (2018), 
i.e. all passages are processed independently but share the same softmax normalization
so that scores can be compared across passages.
The model is implemented in `meerqat.train.trainee` it inherits from HF `transformers.BertForQuestionAnswering`
and the implementation is based on DPR (Karpukhin et al., 2020)

We also implemented the DPR Reader model from Karpukhin et al. (2020), which doesn’t use this global normalization trick
but does re-ranking. However we did not test it (our intuition is that re-ranking with text only will only deteriorate the retriever results)

We train the models based on HF `transformers.Trainer`, itself based on `torch`.

We convert the model start and end answer position probabilities to answer spans in
`meerqat.models.qa.get_best_spans`.
The answer span probabilities can be weighed with the retrieval score, which is ensured to be > 1.
We also enforce that the start starts before the end and 
that the first token (`[CLS]`) cannot be the answer since it’s the objective for irrelevant passages
(this is the default behavior but can be changed with the `cannot_be_first_token` flag).


### Pre-processing
Our clue that the passage is relevant for the answer is quite weak:
it contains the answer. That’s it. 
When scanning for the wikipedia article of the entity (in `meerqat.ir.metrics relevant`)
you might find some passages that contain the answer but have nothing to do with the question.
In order to tackle this, we use relevant passages that come from the IR step in priority.
Moreover, in this step (and it has no impact on the evaluation) we only check for the *original answer*
not all alternative answers (which come from wikipedia aliases).
Since this step does not really fit in any of the modules and I cannot think of a way of making it robust,
I’ll just let you run it yourself from this code snippet:
```py
from datasets import load_from_disk, set_caching_enabled
from meerqat.ir.metrics import find_relevant

set_caching_enabled(False)
kb = load_from_disk('data/meerqat_passages/')
dataset = load_from_disk('data/meerqat_dataset/')

def keep_relevant_search_wrt_original_in_priority(item, kb):
    indices = item['search_indices']
    relevant_indices, _ = find_relevant(indices, item['output']['original_answer'], [], kb)
    if relevant_indices:
        item['search_provenance_indices'] = relevant_indices
    else:
        item['search_provenance_indices'] = item['original_answer_provenance_indices']
    item['search_irrelevant_indices'] = list(set(indices) - set(relevant_indices))
    return item
    
dataset = dataset.map(keep_relevant_search_wrt_original_in_priority, fn_kwargs=dict(kb=kb))
dataset.save_to_disk('data/meerqat_dataset/')
``` 

### Pre-training on TriviaQA
We should provide this model so that you’re able to skip this step.

Our training set consists of questions that were not used to generate any MEERQAT questions, 
even those that were discarded or remain to be annotated.
Our validation set consists of the questions that were used to generate MEERQAT validation set.

We used the same hyperparameters as Karpukhin et al. except for the ratio of relevant passages:
We use 8 relevant and 16 irrelevant passages (so 24 in total) per question 
(the intuition was to get a realistic precision@24 score w.r.t. the search results, 
we haven’t tried any other setting).
The model is trained to predict the first token (`[CLS]`) as answer for irrelevant passages.

- `max_n_answers`: the model is trained to predict all off the positions of the answer in the passage up to this threshold 
- `train_original_answer_only`: use in conjunction with the above preprocessing, defaults to True

```sh
python -m meerqat.train.trainer experiments/rc/triviaqa/train/config.json
```



### Fine-tuning on MEERQAT

This is kind of a hack but once you’ve decided on a TriviaQA checkpoint (step 46000 in our case)
you want to be sure that HF won’t load the optimizer or any other training stuff except the model:
```sh
cd experiments/rc/triviaqa/train/checkpoint-46000
mkdir .keep
mv optimizer.pt .keep
mv scheduler.pt .keep
mv trainer_state.pt .keep
mv training_args.pt .keep
```
Then you can fine-tune the model:
```sh
python -m meerqat.train.trainer experiments/rc/meerqat/train/config.json
```

Note that the validation is done using the same ratio of relevant and irrelevant passages (8:16) as training
while test is done using the top-24 IR results. That is why you should expect a performance gap between validation and test.

The test is configured to save the prediction (without IR weighing) along with the metrics,
if you don’t want this, set `do_eval=True` and `do_predict=False`.

```sh
python -m meerqat.train.trainer experiments/rc/meerqat/test/config.json
```

# References
```bib
@inproceedings{clark_simple_2018,
        address = {Melbourne, Australia},
        title = {Simple and {Effective} {Multi}-{Paragraph} {Reading} {Comprehension}},
        url = {https://aclanthology.org/P18-1078},
        doi = {10.18653/v1/P18-1078},
        urldate = {2021-07-08},
        booktitle = {Proceedings of the 56th {Annual} {Meeting} of the {Association} for {Computational} {Linguistics} ({Volume} 1: {Long} {Papers})},
        publisher = {Association for Computational Linguistics},
        author = {Clark, Christopher and Gardner, Matt},
        month = jul,
        year = {2018},
        pages = {845--855},
    }

@inproceedings{deng_arcface_2019,
	title = {{ArcFace}: {Additive} {Angular} {Margin} {Loss} for {Deep} {Face} {Recognition}},
	shorttitle = {{ArcFace}},
	url = {https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html},
	urldate = {2020-11-27},
	author = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
	year = {2019},
	pages = {4690--4699},
}

@article{devlin_bert_2019,
	title = {{BERT}: {Pre}-training of {Deep} {Bidirectional} {Transformers} for {Language} {Understanding}},
	shorttitle = {{BERT}},
	url = {http://arxiv.org/abs/1810.04805},
	abstract = {We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be ﬁnetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspeciﬁc architecture modiﬁcations.},
	language = {en},
	urldate = {2020-10-09},
	journal = {arXiv:1810.04805 [cs]},
	author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
	month = may,
	year = {2019},
	note = {arXiv: 1810.04805},
}

@inproceedings{guo_ms-celeb-1m_2016,
	address = {Cham},
	series = {Lecture {Notes} in {Computer} {Science}},
	title = {{MS}-{Celeb}-{1M}: {A} {Dataset} and {Benchmark} for {Large}-{Scale} {Face} {Recognition}},
	isbn = {978-3-319-46487-9},
	shorttitle = {{MS}-{Celeb}-{1M}},
	doi = {10.1007/978-3-319-46487-9_6},
	language = {en},
	booktitle = {Computer {Vision} – {ECCV} 2016},
	publisher = {Springer International Publishing},
	author = {Guo, Yandong and Zhang, Lei and Hu, Yuxiao and He, Xiaodong and Gao, Jianfeng},
	editor = {Leibe, Bastian and Matas, Jiri and Sebe, Nicu and Welling, Max},
	year = {2016},
	pages = {87--102},
}

@inproceedings{karpukhin_dense_2020,
	title = {Dense {Passage} {Retrieval} for {Open}-{Domain} {Question} {Answering}},
	url = {https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf},
	booktitle = {Proceedings of the 2020 {Conference} on {Empirical} {Methods} in {Natural} {Language} {Processing} ({EMNLP})},
	author = {Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau},
	year = {2020},
	note = {https://github.com/facebookresearch/DPR},
	pages = {6769--6781},
}

@inproceedings{wang_multi-passage_2019,
        address = {Hong Kong, China},
        title = {Multi-passage {BERT}: {A} {Globally} {Normalized} {BERT} {Model} for {Open}-domain {Question} {Answering}},
        shorttitle = {Multi-passage {BERT}},
        url = {https://www.aclweb.org/anthology/D19-1599},
        doi = {10.18653/v1/D19-1599},
        urldate = {2021-06-14},
        booktitle = {Proceedings of the 2019 {Conference} on {Empirical} {Methods} in {Natural} {Language} {Processing} and the 9th {International} {Joint} {Conference} on {Natural} {Language} {Processing} ({EMNLP}-{IJCNLP})},
        publisher = {Association for Computational Linguistics},
        author = {Wang, Zhiguo and Ng, Patrick and Ma, Xiaofei and Nallapati, Ramesh and Xiang, Bing},
        month = nov,
        year = {2019},
        pages = {5878--5882}
    }

@article{zhang_joint_2016,
	title = {Joint {Face} {Detection} and {Alignment} {Using} {Multitask} {Cascaded} {Convolutional} {Networks}},
	volume = {23},
	issn = {1558-2361},
	doi = {10.1109/LSP.2016.2603342},
	abstract = {Face detection and alignment in unconstrained environment are challenging due to various poses, illuminations, and occlusions. Recent studies show that deep learning approaches can achieve impressive performance on these two tasks. In this letter, we propose a deep cascaded multitask framework that exploits the inherent correlation between detection and alignment to boost up their performance. In particular, our framework leverages a cascaded architecture with three stages of carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. In addition, we propose a new online hard sample mining strategy that further improves the performance in practice. Our method achieves superior accuracy over the state-of-the-art techniques on the challenging face detection dataset and benchmark and WIDER FACE benchmarks for face detection, and annotated facial landmarks in the wild benchmark for face alignment, while keeps real-time performance.},
	number = {10},
	journal = {IEEE Signal Processing Letters},
	author = {Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Qiao, Yu},
	month = oct,
	year = {2016},
	note = {Conference Name: IEEE Signal Processing Letters},
	pages = {1499--1503},
}
```