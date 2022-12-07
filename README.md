
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/modelscope)](https://pypi.org/project/modelscope/)
<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/issues)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/pull/)
[![GitHub latest commit](https://badgen.net/github/last-commit/modelscope/modelscope)](https://GitHub.com/modelscope/modelscope/commit/)
<!-- [![GitHub contributors](https://img.shields.io/github/contributors/modelscope/modelscope.svg)](https://GitHub.com/modelscope/modelscope/graphs/contributors/) -->
<!-- [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com) -->


</div>

# Introduction

[ModelScope]( https://www.modelscope.cn) is a “Model-as-a-Service” (MaaS) platform that seeks to bring together most advanced machine learning models from the AI community, and to streamline the process of leveraging AI models in real applications. The core ModelScope library enables developers to perform inference, training and evaluation, through rich layers of API designs that facilitate a unified experience across state-of-the-art models from different AI domains.

The Python library offers the layered-APIs necessary for model contributors to integrate models from CV, NLP, Speech, Multi-Modality, as well as Scientific-computation, into the ModelScope ecosystem. Implementations for all these different models are encapsulated within the library in a way that allows easy and unified access. With such integration, model inference, finetuning, and evaluations can be done with only a few lines of codes. In the meantime, flexibilities are provided so that different components in the model applications can be customized as well, where necessary.

Apart from harboring implementations of various models, ModelScope library also enables the necessary interactions with ModelScope backend services, particularly with the Model-Hub and Dataset-Hub. Such interactions facilitate management of  various entities (models and datasets) to be performed seamlessly under-the-hood, including entity lookup, version control, cache management, and many others.

# Installation

Please refer to [installation](https://modelscope.cn/docs/%E7%8E%AF%E5%A2%83%E5%AE%89%E8%A3%85).

# Get Started

You can refer to [quick_start](https://modelscope.cn/docs/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B) for quick start.

We also provide other documentations including:
* [Introduction to tasks](https://modelscope.cn/docs/%E4%BB%BB%E5%8A%A1%E7%9A%84%E4%BB%8B%E7%BB%8D)
* [Use pipeline for model inference](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline)
* [Finetune example](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train)
* [Preprocessing of data](https://modelscope.cn/docs/%E6%95%B0%E6%8D%AE%E7%9A%84%E9%A2%84%E5%A4%84%E7%90%86)
* [Evaluation metrics](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AF%84%E4%BC%B0)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
