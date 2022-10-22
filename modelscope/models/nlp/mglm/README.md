# GLM

GLM is a General Language Model pretrained with an autoregressive blank-filling objective and can be finetuned on
various natural language understanding and generation tasks.

Please refer to our paper for a detailed description of GLM:

[All NLP Tasks Are Generation Tasks: A General Pretraining Framework](https://arxiv.org/abs/2103.10360)

Zhengxiao Du*, Yujie Qian*, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, Jie Tang (*: equal contribution)

Part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [PET](https://github.com/timoschick/pet).

## Pretrained Models
You can download the pretrained models used in the paper [here](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/Eg8MZe62MlVFs_mK2tHaH-sBC-UC01jpGPZop08pID7sOw?e=MsevNR).

| Name | Params | File | Config
|  -----  | ----  | ---- | ----
| GLM-Base | 110M | glm-base-blank.tar.bz2 | model_blocklm_base.sh
| GLM-Large  | 335M | glm-large-blank.tar.bz2 | model_blocklm_large.sh
| GLM-Large (multi-task) | 335M | glm-large-generation.tar.bz2 | model_blocklm_large_generation.sh
| GLM-410M (multi-task) | 410M | glm-1.25-generation.tar.bz2 | model_blocklm_1.25_generation.sh
| GLM-515M (multi-task) | 515M | glm-1.5-generation.tar.bz2 | model_blocklm_1.5_generation.sh
| GLM-RoBERTa | 335M | glm-roberta-large-blank.tar.bz2 | model_blocklm_roberta_large.sh
| GLM-XXLarge | 10B | [apply here](https://resource.wudaoai.cn/home?ind=2&name=WuDao%20WenHui&id=1399364355975327744) | model_blocklm_10B.sh
| GLM-XXLarge-Chinese | 10B | - | model_blocklm_10B_chinese.sh

## Results

### [SuperGLUE](https://super.gluebenchmark.com)
dev set, single model, single-task finetuning

|  Model | COPA | WSC | RTE | WiC | CB | MultiRC | BoolQ | ReCoRD |
|  ----  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| GLM-XXLarge  | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7 | 94.4/94.0 |
| [RoBERTa-Large](https://github.com/pytorch/fairseq/tree/master/examples/roberta) | 94.0 | 91.3 | 86.6 | 75.6 | 98.2/- | 85.7/- | 86.9 |89.5/89.0|
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | - | 93.5 | - | - | 87.8/63.6 | 88.3 | 94.1/93.7 |

### Seq2Seq
[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) (test set, no additional data used)

|  Model  | ROUGE-1 | ROUGE-2 | ROUGE-L |
|  ----  | ---- | ---- | ---- |
| GLM-XXLarge | **44.7** | 21.4 | **41.4** |
| T5-11B | 43.5 | **21.6** | 40.7 |
| PEGASUS-Large | 44.2 | 21.5 | **41.4** |
| BART-Large | 44.2 | 21.3 | 40.9 |

[XSum](https://github.com/EdinburghNLP/XSum) (test set, no additional data used)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ---- | ---- | ---- | ---- |
| GLM-XXLarge | **48.9** | **25.7** | **40.4** |
| PEGASUS-Large | 47.2 | 24.6 | 39.3 |
| BART-Large | 45.1 | 22.3 | 37.3 |

### Language Modeling
test set, zero-shot

| Model | LAMBADA (accuracy) | Wikitext103 (perplexity) |
| ---- | ---- | ---- |
| GLM-XXLarge (bi) | 72.35 | 11.33 |
| GLM-XXLarge (uni) | 67.18 | 12.22 |
| GPT-2 | 52.66 | 17.48 |
| Megatron-LM (8.3B) | 66.51 | 10.81 |
| Turing-NLG | 67.98 | 10.21 |

## Get Started
### Docker Image
We prepare two docker images based on CUDA 10.2 and CUDA 11.2. You can pull the pre-built images from Docker Hub and run with docker v19.03+
  ```shell
  docker run --gpus all --rm -it --ipc=host zxdu20/glm-cuda102
  ```
  or replace `glm-cuda102` with `glm-cuda112`.

  You can also modify the image according to your requirements in [docker/cuda102.dockerfile](docker/cuda102.dockerfile) and build the image yourself
  ```shell
    docker build -f cuda102.dockerfile . -t glm-cuda102
  ```
### Manual Installation
Please first install PyTorch (we use 1.7.0) and [apex](https://github.com/NVIDIA/apex), and then install other dependencies by `pip install -r requirements.txt`
### Clone this repo
  ```shell
  git clone https://github.com/THUDM/GLM
  cd GLM
  ```

## Usage
We provide scripts for finetuning GLM on some downstream tasks.

### SuperGLUE

- Download the [SuperGlue](https://super.gluebenchmark.com/tasks) data and check the experiment setup in
  [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh). Note that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH`
  need to be changed to your local path. You may also change the `batch-size` and `nproc_per_node` according to your
  available hardware.

- Run the following script (use the COPA dataset as an example)

```
bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```
- We also implement [P-Tuning](https://arxiv.org/abs/2103.10385) in our code. Run the following script to integrate p-tuning:
```shell
bash scripts/ds_finetune_superglue_prompt.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- To apply GLM to a new NLU dataset with cloze-filling finetuning, implement a `DataProcessor` in
  [tasks/superglue/dataset.py](tasks/superglue/dataset.py) for data loading and add a `PVP` in
  [tasks/superglue/pvp.py](tasks/superglue/pvp.py) for the cloze question. More details can be found
  [here](tasks/superglue/README.md).

### Text Summarization

- Download the [Gigaword](https://github.com/harvardnlp/sent-summary), [CNN/Daily Mail](https://github.com/artmatsak/cnn-dailymail) or [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset) dataset and check the experiment setup in
  [scripts/ds_finetune_seq2seq.sh](scripts/ds_finetune_seq2seq.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your
  local path.

- Run the following script (use the CNN/Daily Mail dataset as an example)

  ```
  bash scripts/ds_finetune_seq2seq.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/seq_cnndm_org.sh
  ```
- The summaries are written into `./runs/experiment_name/test.jsonl.hyps`. The references are written into `test.jsonl.refs` in the same directory. For calculating rouge, install [file2rouge](https://github.com/pltrdy/files2rouge) and download Stanford CoreNLP from [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip). Run  the following script
  ```
  bash scripts/evaluate_seq2seq.sh \
   ./runs/experiment_name/test.jsonl.hyps ./runs/experiment_name/test.jsonl.refs
  ```

### Language Modeling
#### LAMBADA Cloze Accuracy
* Download the [LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) data and change
  `DATA_ROOT, CHECKPOINT_PATH` in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
```shell
bash scripts/evaluate_lm.sh \
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh
```
#### LM Perplexity
* Download our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh) or [Wikitext103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) dataset and change `DATA_ROOT, CHECKPOINT_PATH`
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_wikitext.sh
  ```

### Blank Language Model
- Download the [Yahoo](https://github.com/Varal7/blank_language_model) dataset and check the experiment setup in
  [scripts/finetune_blank.sh](scripts/finetune_blank.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your
  local path.

- Run the following script

```
bash scripts/finetune_blank.sh \
     config_tasks/model_blocklm_large.sh \
     config_tasks/seq_blank.sh
```

### Blank Filling (Interactive)
* Change `CHECKPOINT_PATH` to your local path. Run the following script
```
bash scripts/generate_block.sh \
     config_tasks/model_blocklm_10B_chinese.sh
```
#### Example1 (Entity Prediction):

Context: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。

GLM:拿破仑军队攻克米兰城

#### Example2 (Sentence Prediction)
Context: 工业互联网（Industrial Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM: 工业互联网是制造业技术、管理、模式的重大变革,是推动互联网、大数据、人工智能和实体经济深度融合的重要载体,是建设制造强国和网络强国的重要基础。

#### Example3 (Long Text Generation)
Context: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]

GLM:海南三亚,三亚的冬天是三亚的旺季,因为冬天国内大部分地方都有雾霾,而三亚却是没有雾霾的地方,所以三亚是冬天最惬意不过的地方了。在东北长大的我觉得三亚简直就是一个天堂,冬天去海口三亚旅游,享受冬天,享受阳光沙滩。但是海口却是很干燥,需要多喝水。 三亚冬天最热门的玩法就是晒太阳,在海边晒晒太阳,在沙滩上晒晒太阳,感觉整个人都得到了解放。三亚还有一个特色项目,就是海上冲浪,在三亚的沙滩上冲浪也是一件非常刺激的事情。 海口,海口冬季的阳光十分温暖,海南的冬季也是属于冬季旅游的旺季。冬季的海口最棒的是去海南的热带野生动植物园,那里有数之不尽的热带小动物,在这里可以近距离的和它们接触,海南的热带野生动植物园也是海南的天然氧吧。还可以在海口观澜湖公园里感受海口美丽的海景。 贵阳,贵州的冬天也是十分温暖的,贵阳也是冬季避寒很好的城市之一。冬季去贵阳玩一定要去黔灵山,黔灵山是贵州香火很旺盛的一个寺庙,寺庙的冬季香火鼎盛,在冬季去寺庙游玩也是一个很好的体验。除了黔灵山,贵阳在冬季还有花溪公园可以去玩,花溪公园也是去当地公园玩最好的选择。 青岛,青岛的冬天是青岛最舒服的时候,青岛有很多海滨浴场,冬天去海边泡一泡温泉,然后晒晒太阳是一件十分惬意的事情。青岛也有沙滩,冬天在沙滩上晒晒太阳,看看海,再玩玩沙滩游戏,感觉十分快乐的事。

## Pretrain
Run the following script to pre-train the GLM-Large model
```shell
bash scripts/ds_pretrain_nvidia.sh config/ds_block_large.sh
```

The script [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) launches the training program with DeepSpeed. You should change `NUM_WORKERS` and `NUM_GPUS_PER_WORKER` to the number of workers and the number of gpus per worker. Also change `HOST_FILE_PATH` to the path to an OpenMPI-style hostfile. More details about DeepSpeed launcher can be found [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

The file [config/ds_block_large.sh](config/ds_block_large.sh) defines the hyperparameters for pretraining. Most of the arguments are fairly self-explanatory. Specifically, `--train-data` can be multiple keywords defined in `NAMED_CORPORA` in [data_utils/corpora.py](data_utils/corpora.py). The hyperparameters of the optimizer are defined in the corresponding json file under `config`. The semantics of the json file can be found [here](https://www.deepspeed.ai/docs/config-json).

## Citation
Please cite our paper if you find this code useful for your research:
```
@article{DBLP:journals/corr/abs-2103-10360,
  author    = {Zhengxiao Du and
               Yujie Qian and
               Xiao Liu and
               Ming Ding and
               Jiezhong Qiu and
               Zhilin Yang and
               Jie Tang},
  title     = {All {NLP} Tasks Are Generation Tasks: {A} General Pretraining Framework},
  journal   = {CoRR},
  volume    = {abs/2103.10360},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.10360}
}
```
