# Experiments

All of the intermediate outputs of the pipeline should be provided along with the data 
so that people are free to skip one or few steps. (TODO add instructions here or in the README data section)

All commands assume that the working directory is the root of the git repo (e.g. same level as this file) 
and that the data is stored in the `data` folder, at the root of this repo (except for images for which you can specify the `VIQUAE_IMAGES_PATH` environment variable).
Alternatively, you can the paths in the config files.

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
    + [Fine-tuning on ViQuAE](#fine-tuning-on-viquae)
* [References](#references)


## Image
This will be applied on both the QA dataset and the KB.  

### Global image embedding

Obtained using ResNet-50:
- one pre-trained on ImageNet, pooled with max-pooling. 
  You can tweak the pooling layer and the backbone in the config file, 
  as long as it is a `nn.Module` and `torchvision.models`, respectively.
- the other trained using [CLIP](https://github.com/openai/CLIP)

Obviously you can also tweak the batch size.
```sh
# embed dataset images with ImageNet-ResNet50
python -m meerqat.image.embedding data/viquae_dataset experiments/image_embedding/imagenet/config.json --disable_caching
# embed KB images with ImageNet-ResNet50
python -m meerqat.image.embedding data/viquae_wikipedia experiments/image_embedding/imagenet/config.json --disable_caching
# embed dataset images with CLIP-ResNet50
python -m meerqat.image.embedding data/viquae_dataset experiments/image_embedding/clip/config.json --disable_caching
# embed KB images with CLIP-ResNet50
python -m meerqat.image.embedding data/viquae_wikipedia experiments/image_embedding/clip/config.json --disable_caching
```

### Face detection

Things get a little more complicated here, first, you will want to split your KB in humans and non-humans,
since we assume that faces are not relevant for non-human entities.
I guess there’s no need to provide code for that since it’s quite trivial and we will provide KB already split in humans and non-humans.

Face detection uses MTCNN (Zhang et al., 2016) via the `facenet_pytorch` library.
Feel free to tweak the hyperparameters (we haven’t), you can also set whether to order faces by size or probability (we do the latter)
 
Probabilities, bounding boxes and landmarks are saved directly in the dataset, face croping happens as a pre-processing of Face recognition (next section).

```sh
python -m meerqat.image.face_detection data/viquae_dataset --disable_caching --batch_size=256
python -m meerqat.image.face_detection data/viquae_wikipedia/humans --disable_caching --batch_size=256
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
python -m meerqat.image.face_recognition data/viquae_dataset experiments/face_detection/viquae_dataset experiments/face_recognition/config.json --disable_caching
python -m meerqat.image.face_recognition data/viquae_wikipedia/humans_with_faces experiments/face_detection/viquae_wikipedia experiments/face_recognition/config.json --disable_caching
```

## IR

Now that we have a bunch of dense representations, let’s see how to retrieve information!
Dense IR is done with `faiss` and sparse IR is done with `elasticsearch`, both via HF `datasets`.
We’ll use IR on both TriviaQA along with the complete Wikipedia (BM25 only) and ViQuAE along with the multimodal Wikipedia.

### Preprocessing passages
You can probably skip this step as we will provide passages dataset along with provenance.

Articles are stripped of semi-structured data, such as tables and lists. 
Each article is then split into disjoint passages of 100 words for text retrieval, while preserving sentence boundaries, 
and the title of the article is appended to the beginning of each passage.

```sh
python -m meerqat.data.loading passages data/viquae_wikipedia data/viquae_passages experiments/passages/config.json --disable_caching
```

Then you can extract some columns from the dataset to allow quick (and string) indexing:
```sh
python -m meerqat.data.loading map data/viquae_wikipedia wikipedia_title title2index.json --inverse --disable_caching
python -m meerqat.data.loading map data/viquae_wikipedia passage_index article2passage.json --disable_caching
```

This allows us to find the relevant passages for the question (i.e. those than contain the answer or the alternative answers):
```sh
python -m meerqat.ir.metrics relevant data/viquae_dataset data/viquae_passages data/viquae_wikipedia/title2index.json data/viquae_wikipedia/article2passage.json --disable_caching
```

### BM25
Before running any of the commands below you should [launch the Elastic Search server](https://www.elastic.co/guide/en/elastic-stack-get-started/current/get-started-elastic-stack.html#install-elasticsearch).

First you might want to optimize BM25 hyperparameters, `b` and `k_1`.
We did this with a grid-search using `optuna`:
the `--k` option asks for the top-K search results.  
```sh
TODO
```

Alternatively, you can use the parameters we optimized: `b=0.3` and `k_1=0.5`:
```sh
python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/bm25/config.json --k=100 --metrics=experiments/ir/viquae/bm25/metrics --disable_caching
```

### DPR
We use the same hyperparameters as [Karpukinh et al.](https://github.com/facebookresearch/DPR).
We train DPR using 4 V100 GPUs of 32Go, allowing a total batch size of 256 (32 questions * 2 passages each * 4 GPUs). 
This is crucial because each question uses all passages paired with other questions in the batch as negative examples. 
Each question is paired with 1 relevant passage and 1 irrelevant passage mined with BM25.

Both the question and passage encoder are initialized from `"bert-base-uncased"`.

To launch the script with multiple GPUs you should you use `torch.distributed.launch --nproc_per_node=<number of GPUs>`. This is omitted in the following commands.


#### Pre-training on TriviaQA
TODO provide pre-trained model.

Given the small size of ViQuAE, DPR is pre-trained on TriviaQA:
- filtered out of all questions used for ViQuAE for training 
- on questions used to generate ViQuAE’s validation set for validation

In this step we use the complete `kilt_wikipedia` instead of `viquae_wikipedia`.

`python -m meerqat.train.trainer experiments/dpr/kilt/config.json`

The best checkpoint should be `checkpoint-13984`.

#### Fine-tuning on ViQuAE
TODO provide pre-trained model.

We use exactly the same hyperparameters as for pre-training.

This is kind of a hack but once you’ve decided on a TriviaQA checkpoint (step 13984 in our case)
you want to be sure that HF won’t load the optimizer or any other training stuff except the model:
```sh
mkdir experiments/dpr/kilt/checkpoint-13984/.keep
mv experiments/dpr/kilt/checkpoint-13984/optimizer.pt experiments/dpr/kilt/checkpoint-13984/scheduler.pt experiments/dpr/kilt/checkpoint-13984/training_args.pt experiments/dpr/kilt/checkpoint-13984/trainer_state.pt experiments/dpr/kilt/checkpoint-13984/.keep
```

`python -m meerqat.train.trainer experiments/dpr/viquae/config.json`

The best checkpoint should be `checkpoint-40`.
Run `python -m meerqat.train.split_DPR experiments/dpr/viquae/checkpoint-40` to split DPR in a a DPRQuestionEncoder and DPRContextEncoder.
We’ll use both to embed questions and passages below.

#### Embedding questions and passages
```sh
python -m meerqat.ir.embedding data/viquae_dataset experiments/ir/viquae/dpr/questions/config.json --disable_caching
python -m meerqat.ir.embedding data/viquae_passages experiments/ir/viquae/dpr/passages/config.json --disable_caching
```

#### Searching

Like with BM25:

```sh
python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/dpr/search/config.json --k=100 --metrics=experiments/ir/viquae/dpr/search/metrics --disable_caching
```

### ImageNet-ResNet and CLIP vs ArcFace-MS-Celeb
We trust the face detector, if it detects a face then:
- the search is done on the human faces KB (`data/viquae_wikipedia/humans_with_faces`)

else:
- the search is done on the non-human global images KB (`data/viquae_wikipedia/non_humans`)

To implement that we simply set the global image embedding to None when a face was detected:
```py
from datasets import load_from_disk, set_caching_enabled
set_caching_enabled(False)
dataset = load_from_disk('data/viquae_dataset/')
dataset = dataset.rename_column('imagenet-RN50', 'keep_imagenet-RN50')
dataset = dataset.rename_column('clip-RN50', 'keep_clip-RN50')
dataset = dataset.map(lambda item: {'imagenet-RN50': item['keep_imagenet-RN50'] if item['face_embedding'] is None else None})
dataset = dataset.map(lambda item: {'clip-RN50': item['keep_clip-RN50'] if item['face_embedding'] is None else None})
dataset.save_to_disk('data/viquae_dataset/')
```

Search is done using cosine distance, hence the `"L2norm,Flat"` for `string_factory` and `metric_type=0`
(this does first L2-normalization then dot product).

The results, corresponding to a KB entity/article are then mapped to the corresponding passages to allow fusion with BM25 (next §)
The image results is simply the union of both ResNet and ArcFace scores since they are exclusive 
(you can tweak that with the `weight` parameter in the config file)

Beware that these results cannot be well interpreted because they do not have a reference KB which would allow looking if the answer is in the passage (TODO refactor to allow that). See other caveats in the next section.

### Text + Image

Now in order to combine the text results of text and the image results we do two things:
1. normalize the scores so that they have zero-mean and unit variance, 
   **the mean and the variance is computed over the whole subset** so you might want to do a dry run first **or use ours** 
   (this corresponds to the mysterious "normalization" parameter in the config files)
2. sum the text and image score for each passage before re-ordering, note that
   if only the text finds a given passage then its image score is set to the minimum of the image results (and vice-versa)

The results are then re-ordered before evaluation.
Each model has an interpolation hyperparameter. You can either tune-it on the dev set or use ours (more details below).

#### BM25 + Image

##### Tune hyperparameters
`python -m meerqat.ir.hp fusion data/viquae_dataset/validation experiments/ir/viquae/hp/bm25+image/config.json --k=100 --disable_caching --test=data/viquae_dataset/test --metrics=experiments/ir/viquae/hp/bm25+image/metrics`

##### Run with the best hyperparameters
If you don’t use the `--test` option above.

```sh
python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/bm25+image/config.json --k=100 --metrics=experiments/ir/viquae/bm25+image/metrics
```


#### DPR + Image
Same script, different config.

`python -m meerqat.ir.hp fusion data/viquae_dataset/validation experiments/ir/viquae/hp/dpr+image/config.json --k=100 --disable_caching --test=data/viquae_dataset/test --metrics=experiments/ir/viquae/hp/dpr+image/metrics`

##### Run with the best hyperparameters
If you don’t use the `--test` option above.

```sh
python -m meerqat.ir.search data/viquae_dataset/test experiments/ir/viquae/dpr+image/config.json --k=100 --metrics=experiments/ir/viquae/dpr+image/metrics
```

### Metrics
We use [ranx](https://github.com/AmenRa/ranx) to compute the metrics. TODO explain how to compare models.
I advise against using any kind of metric that uses recall (mAP, R-Precision, …) 
since we estimate relevant document on the go so the number of relevant documents will *depend on the systemS* you use.

Beware that the ImageNet-ResNet and ArcFace results cannot be compared, neither between them nor with BM25/DPR because:
- they are exclusive, roughly **half** the questions have a face -> ArcFace, other don't -> ResNet, while BM25/DPR is applied to **all** questions
- the mapping from image/document to passage is arbitrary, so the ordering of image results is not so meaningful until it is re-ordered with BM25/DPR

If you’re interested in comparing only image representations, leaving downstream performance aside (e.g. comparing ImageNet-Resnet with another representation for the full image), you should:
- `filter` the dataset so that you don’t evaluate on irrelevant questions (e.g. those were the search is done with ArcFace because a face was detected)
- evaluate at the *document-level* instead of passage-level. To do so, maybe `checkout` the `document` branch (TODO merge in `main`).


## Reading Comprehension

Now we have retrieved candidate passages, it’s time to train a Reading Comprehension system (reader).
We first pre-train the reader on TriviaQA before fine-tuning it on ViQuAE.
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
kb = load_from_disk('data/viquae_passages/')
dataset = load_from_disk('data/viquae_dataset/')

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
dataset.save_to_disk('data/viquae_dataset/')
``` 

### Pre-training on TriviaQA
We should provide this model so that you’re able to skip this step.

Our training set consists of questions that were not used to generate any ViQuAE questions, 
even those that were discarded or remain to be annotated.
Our validation set consists of the questions that were used to generate ViQuAE validation set.

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



### Fine-tuning on ViQuAE

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
python -m meerqat.train.trainer experiments/rc/viquae/train/config.json
```

Note that the validation is done using the same ratio of relevant and irrelevant passages (8:16) as training
while test is done using the top-24 IR results. That is why you should expect a performance gap between validation and test.

The test is configured to save the prediction (without IR weighing) along with the metrics,
if you don’t want this, set `do_eval=True` and `do_predict=False`.

```sh
python -m meerqat.train.trainer experiments/rc/viquae/test/config.json
```

# References
Christopher Clark and Matt Gardner. 2018. Simple and
Effective Multi-Paragraph Reading Comprehension.
In Proceedings of the 56th Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 845–855, Melbourne, Australia.
Association for Computational Linguistics.

Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos
Zafeiriou. 2019. ArcFace: Additive Angular Margin
Loss for Deep Face Recognition. pages 4690–4699.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. BERT: Pre-training
of Deep Bidirectional Transformers for Language
Understanding. arXiv:1810.04805 [cs]. ArXiv:
1810.04805.

Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He,
and Jianfeng Gao. 2016. MS-Celeb-1M: A Dataset
and Benchmark for Large-Scale Face Recognition.
In Computer Vision – ECCV 2016, Lecture Notes in
Computer Science, pages 87–102, Cham. Springer
International Publishing.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense Passage Retrieval for
Open-Domain Question Answering. In Proceedings
of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 6769-6781. Https://github.com/facebookresearch/DPR.

Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallap-
ati, and Bing Xiang. 2019. Multi-passage BERT:
A Globally Normalized BERT Model for Open-
domain Question Answering. In Proceedings of
the 2019 Conference on Empirical Methods in Natu-
ral Language Processing and the 9th International
Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pages 5878–5882, Hong Kong,
China. Association for Computational Linguistics.

Kaipeng Zhang, Zhanpeng Zhang, Zhifeng Li, and
Yu Qiao. 2016. Joint Face Detection and Alignment
Using Multitask Cascaded Convolutional Networks.
IEEE Signal Processing Letters, 23(10):1499–1503.
Conference Name: IEEE Signal Processing Letters.