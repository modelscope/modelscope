
<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/modelscope)](https://pypi.org/project/modelscope/)
<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/modelscope)](https://GitHub.com/modelscope/modelscope/commit/)
[![Leaderboard](https://img.shields.io/badge/ModelScope-Check%20Your%20Contribution-orange)](https://opensource.alibaba.com/contribution_leaderboard/details?projectValue=modelscope)

<!-- [![GitHub contributors](https://img.shields.io/github/contributors/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/graphs/contributors/) -->
<!-- [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) -->

<h4 align="center">
    <p>
        <a href="https://github.com/modelscope/modelscope/blob/master/README.md">English</a> |
        <a href="https://github.com/modelscope/modelscope/blob/master/README_zh.md">中文</a> |
        日本語
    <p>
</h4>


</div>

# はじめに

[ModelScope](https://www.modelscope.cn) は、"Model-as-a-Service"(MaaS) の概念に基づいて構築されています。AI コミュニティから最も先進的な機械学習モデルを集め、実世界のアプリケーションで AI モデルを活用するプロセスを合理化することを目指しています。このリポジトリでオープンソース化されている中核となる ModelScope ライブラリは、開発者がモデルの推論、トレーニング、評価を実行するためのインターフェースと実装を提供します。


特に、API 抽象化の豊富なレイヤーにより、ModelScope ライブラリは、CV、NLP、音声、マルチモダリティ、科学計算などのドメインにまたがる最先端のモデルを探索するための統一された体験を提供します。様々な分野のモデル貢献者は、レイヤー化された API を通じてモデルを ModelScope エコシステムに統合することができ、モデルへの容易で統一されたアクセスを可能にします。一旦統合されると、モデルの推論、微調整、および評価は、わずか数行のコードで行うことができます。一方、モデルアプリケーションの様々なコンポーネントを必要に応じてカスタマイズできるように、柔軟性も提供されています。

ModelScope ライブラリは、様々なモデルの実装を保持するだけでなく、ModelScope のバックエンドサービス、特に Model-Hub と Dataset-Hub との必要な相互作用も可能にします。このような相互作用により、エンティティの検索、バージョン管理、キャッシュ管理など、様々なエンティティ（モデルやデータセット）の管理をアンダーザフードでシームレスに実行することができます。

# モデルとオンラインアクセシビリティ

[ModelScope](https://www.modelscope.cn) では、NLP、CV、オーディオ、マルチモダリティ、科学のための AI などの分野の最新開発を網羅した、何百ものモデルが一般公開されています（700 以上、カウント中）。これらのモデルの多くは、特定の分野における SOTA を代表するものであり、ModelScope でオープンソースとしてデビューしました。ユーザーは、ModelScope([modelscope.cn](http://www.modelscope.cn)) にアクセスし、数回クリックするだけで、オンライン体験を通じて、これらのモデルがどのように機能するかを直接体験することができます。また、[ModelScope](https://www.modelscope.cn) をワンクリックするだけで、クラウド上のすぐに使える CPU/GPU 開発環境に支えられた ModelScope ノートブックを通じて、すぐに開発者体験が可能です。


<p align="center">
    <br>
    <img src="data/resource/inference.gif" width="1024"/>
    <br>
<p>

代表的な例をいくつか挙げると:

NLP:

* [nlp_gpt3_text-generation_2.7B](https://modelscope.cn/models/damo/nlp_gpt3_text-generation_2.7B)

* [ChatYuan-large](https://modelscope.cn/models/ClueAI/ChatYuan-large)

* [mengzi-t5-base](https://modelscope.cn/models/langboat/mengzi-t5-base)

* [nlp_csanmt_translation_en2zh](https://modelscope.cn/models/damo/nlp_csanmt_translation_en2zh)

* [nlp_raner_named-entity-recognition_chinese-base-news](https://modelscope.cn/models/damo/nlp_raner_named-entity-recognition_chinese-base-news)

* [nlp_structbert_word-segmentation_chinese-base](https://modelscope.cn/models/damo/nlp_structbert_word-segmentation_chinese-base)

* [Erlangshen-RoBERTa-330M-Sentiment](https://modelscope.cn/models/fengshenbang/Erlangshen-RoBERTa-330M-Sentiment)

* [nlp_convai_text2sql_pretrain_cn](https://modelscope.cn/models/damo/nlp_convai_text2sql_pretrain_cn)

マルチモーダル:

* [multi-modal_clip-vit-base-patch16_zh](https://modelscope.cn/models/damo/multi-modal_clip-vit-base-patch16_zh)

* [ofa_pretrain_base_zh](https://modelscope.cn/models/damo/ofa_pretrain_base_zh)

* [Taiyi-Stable-Diffusion-1B-Chinese-v0.1](https://modelscope.cn/models/fengshenbang/Taiyi-Stable-Diffusion-1B-Chinese-v0.1)

* [mplug_visual-question-answering_coco_large_en](https://modelscope.cn/models/damo/mplug_visual-question-answering_coco_large_en)

CV:

* [cv_controlnet_controllable-image-generation_nine-annotators](https://modelscope.cn/models/dienstag/cv_controlnet_controllable-image-generation_nine-annotators/summary)

* [cv_tinynas_object-detection_damoyolo](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo)

* [cv_unet_person-image-cartoon_compound-models](https://modelscope.cn/models/damo/cv_unet_person-image-cartoon_compound-models)

* [cv_convnextTiny_ocr-recognition-general_damo](https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-general_damo)

* [cv_resnet18_human-detection](https://modelscope.cn/models/damo/cv_resnet18_human-detection)

* [cv_resnet50_face-detection_retinaface](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface)

* [cv_unet_image-matting](https://modelscope.cn/models/damo/cv_unet_image-matting)

* [cv_F3Net_product-segmentation](https://modelscope.cn/models/damo/cv_F3Net_product-segmentation)

* [cv_resnest101_general_recognition](https://modelscope.cn/models/damo/cv_resnest101_general_recognition)


音声:

* [speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)

* [speech_sambert-hifigan_tts_zh-cn_16k](https://modelscope.cn/models/damo/speech_sambert-hifigan_tts_zh-cn_16k)

* [speech_charctc_kws_phone-xiaoyun](https://modelscope.cn/models/damo/speech_charctc_kws_phone-xiaoyun)

* [u2pp_conformer-asr-cn-16k-online](https://modelscope.cn/models/wenet/u2pp_conformer-asr-cn-16k-online)

* [speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary)

* [punc_ct-transformer_zh-cn-common-vocab272727-pytorch](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary)

* [speech_frcrn_ans_cirm_16k](https://modelscope.cn/models/damo/speech_frcrn_ans_cirm_16k)

* [speech_dfsmn_aec_psm_16k](https://modelscope.cn/models/damo/speech_dfsmn_aec_psm_16k)



科学用 AI:

* [uni-fold-monomer](https://modelscope.cn/models/DPTech/uni-fold-monomer/summary)

* [uni-fold-multimer](https://modelscope.cn/models/DPTech/uni-fold-multimer/summary)

**注:** ModelScope のほとんどのモデルは公開されており、アカウント登録なしで modelscope のウェブサイト([www.modelscope.cn](www.modelscope.cn))からダウンロードすることができます。modelscope のライブラリや git が提供する api を使用してモデルをダウンロードするには、[モデルのダウンロード](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E4%B8%8B%E8%BD%BD)の説明を参照してください。

# クイックツアー

様々なタスクに対して、`pipeline` による推論、`Trainer` による微調整と評価のための統一されたインターフェースを提供します。

入力の種類（画像、テキスト、音声、動画...）を問わず、推論パイプラインはわずか数行のコードで実装することができます。:

```python
>>> from modelscope.pipelines import pipeline
>>> word_segmentation = pipeline('word-segmentation',model='damo/nlp_structbert_word-segmentation_chinese-base')
>>> word_segmentation('今天天气不错，适合出去游玩')
{'output': '今天 天气 不错 ， 适合 出去 游玩'}
```

画像があれば、ポートレート・マット（別名、背景除去）は次のコード・スニペットで実現できます:

![image](data/resource/portrait_input.png)

```python
>>> import cv2
>>> from modelscope.pipelines import pipeline

>>> portrait_matting = pipeline('portrait-matting')
>>> result = portrait_matting('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matting.png')
>>> cv2.imwrite('result.png', result['output_img'])
```

背景を除去した出力画像は次のようになります:
![image](data/resource/portrait_output.png)


ファインチューニングと評価も、トレーニングデータセットとトレーナーをセットアップする数行のコードで行うことができ、モデルのトレーニングと評価の重い作業は `traner.train()` と `trainer.evaluate()` インターフェースの実装に
カプセル化されています。

例えば、gpt3 の基本モデル（1.3B）を中国語詩のデータセットでファインチューニングすることで、中国語詩の生成に使用できるモデルを得ることができる。

```python
>>> from modelscope.metainfo import Trainers
>>> from modelscope.msdatasets import MsDataset
>>> from modelscope.trainers import build_trainer

>>> train_dataset = MsDataset.load('chinese-poetry-collection', split='train'). remap_columns({'text1': 'src_txt'})
>>> eval_dataset = MsDataset.load('chinese-poetry-collection', split='test').remap_columns({'text1': 'src_txt'})
>>> max_epochs = 10
>>> tmp_dir = './gpt3_poetry'

>>> kwargs = dict(
     model='damo/nlp_gpt3_text-generation_1.3B',
     train_dataset=train_dataset,
     eval_dataset=eval_dataset,
     max_epochs=max_epochs,
     work_dir=tmp_dir)

>>> trainer = build_trainer(name=Trainers.gpt3_trainer, default_args=kwargs)
>>> trainer.train()
```

# ModelScope ライブラリを使用する理由

1. 統一された簡潔なユーザーインターフェースは、異なるタスクや異なるモデル用に抽象化されている。モデルの推論とトレーニングは、それぞれわずか 3 行と 10 行のコードで実装できる。ModelScope コミュニティで異なる分野のモデルを探索するのに便利です。ModelScope に統合されたモデルはすべてすぐに使用できるため、教育現場でも産業現場でも、AI を簡単に使い始めることができます。

2. ModelScope は、モデル中心の開発とアプリケーション体験を提供します。モデルのトレーニング、推論、エクスポート、デプロイメントのサポートを合理化し、ユーザーが ModelScope エコシステムに基づいて独自の MLO を構築することを容易にします。

3. モデルの推論とトレーニングのプロセスでは、モジュール設計が導入され、豊富な機能モジュールの実装が提供され、ユーザーが独自のモデルの推論、トレーニング、その他のプロセスをカスタマイズするのに便利です。

4. 分散モデル学習、特に大規模モデルに対しては、データ並列、モデル並列、ハイブリッド並列など、豊富な学習ストラテジーサポートを提供する。

# インストール

## Docker

ModelScope ライブラリは現在、PyTorch、TensorFlow、ONNX を含む、モデルの学習と推論のための一般的なディープラーニングフレームワークをサポートしています。すべてのリリースは、Python 3.7+、Pytorch 1.8+、Tensorflow1.15、または Tensorflow2.0+ でテストされ、実行されます。

ModelScope のすべてのモデルをすぐに使えるようにするため、すべてのリリースで公式の docker イメージが提供されています。開発者はこの docker イメージをベースに、環境のインストールや設定をすべて省略して直接使用することができます。現在、CPU イメージと GPU イメージの最新バージョンは以下から入手できます:

CPU docker イメージ
```shell
# py37
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-1.6.1

# py38
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-py38-torch2.0.1-tf2.13.0-1.9.5
```

GPU docker イメージ
```shell
# py37
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.6.1

# py38
registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.5
```

## ローカル Python 環境のセットアップ

pip と conda を使って、ModelScope のローカル環境を構築することもできます。 ローカルの Python 環境を構築するには [anaconda](https://docs.anaconda.com/anaconda/install/) をお勧めします:

```shell
conda create -n modelscope python=3.7
conda activate modelscope
```

PyTorch または TensorFlow は、それぞれのモデルの要件に応じて個別にインストールすることができます。
* pytorch のインストール [doc](https://pytorch.org/get-started/locally/)
* Tensorflow のインストール [doc](https://www.tensorflow.org/install/pip)

必要な機械学習フレームワークをインストールした後、以下のように modelscope ライブラリをインストールします:

モデル／データセットのダウンロードを試したり、modelscope フレームワークで遊びたいだけなら、modelscope のコア・コンポーネントをインストールすることができます:
```shell
pip install modelscope
```

マルチモーダルモデルを使いたい場合:
```shell
pip install modelscope[multi-modal]
```

nlp モデルを使いたい場合:
```shell
pip install modelscope[nlp] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

CV モデルを使いたい場合:
```shell
pip install modelscope[cv] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

オーディオモデルを使用したい場合:
```shell
pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

科学モデルを使いたい場合:
```shell
pip install modelscope[science] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

`備考`:
1. 現在、一部のオーディオタスクモデルは python3.7、tensorflow1.15.4 の Linux 環境のみに対応しています。他のほとんどのモデルは Windows と Mac(x86) にインストールして使うことができます。

2. オーディオ分野では、wav ファイルの処理にサードパーティ製のライブラリ SoundFile を使用している機種がある。Linux では、SoundFile の libsndfile([doc link](https://github.com/bastibe/python-soundfile#installation)) を手動でインストールする必要があります。Windows や MacOS では、ユーザーが操作しなくても自動的にインストールされる。例えば、Ubuntu の場合、以下のコマンドでインストールできます:
    ```shell
    sudo apt-get update
    sudo apt-get install libsndfile1
    ```

3. コンピュータビジョンのモデルによっては mmcv-full が必要です。mmcv [インストールガイド](https://github.com/open-mmlab/mmcv#installation)を参照してください。最小限のインストールは以下の通りです:

    ```shell
    pip uninstall mmcv # mmcv をインストールしている場合は、アンインストールしてください
    pip install -U openmim
    mim install mmcv-full
    ```



# 詳細

私たちは、以下のような追加書類を提供します:
* [より詳細なインストールガイド](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85)
* [タスクの紹介](https://modelscope.cn/docs/%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%8B%E7%BB%8D)
* [モデル推論にパイプラインを使う](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline)
* [ファインチューニング例](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train)
* [データの前処理](https://modelscope.cn/docs/%E6%95%B0%E6%8D%AE%E7%9A%84%E9%A2%84%E5%A4%84%E7%90%86)
* [評価](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AF%84%E4%BC%B0)
* [ModelScope に自分のモデルを投稿する](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# ライセンス

このプロジェクトのライセンスは [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE) です。
