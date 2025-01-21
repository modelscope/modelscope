## v 0.2.2 (05/07/2022)
Second internal release.

### Highlights

### Algorithms
#### CV
* add cv-person-image-cartoon pipeline
* add action recognition pipeline
* add ocr detection pipeline
* add animal recognition model
* add cmdssl video embedding extraction pipeline

#### NLP
* add speech AEC pipeline
* add palm2.0
* add space model
* add MPLUG model
* add dialog_intent, dialog_modeling, dialog state tracking pipeline
* add maskedlm model and fill_mask pipeline
* add nli pipeline
* add sentence similarity pipeline
* add sentiment_classification pipeline
* add text generation pipeline
* add translation pipeline
* add chinese word segmentation pipeline
* add zero-shot classification

#### Audio
* add tts pipeline
* add kws kwsbp pipeline
* add linear aec pipeline
* add ans pipeline

#### Multi-Modal
* add image captioning pipeline
* add multi-modal feature extraction pipeline
* add text to image synthesis pipeline
* add VQA pipeline

### Framework
* add msdataset interface
* add hub interface and cache support
* support multiple models in single pipeline
* add default model configuration for each pipeline
* remove task field image and video, using cv instead
* dockerfile support
* multi-level tests support
* sphinx-docs use book theme
* formalize the output of pipeline and make pipeline reusable
* pipeline refactor and standardize module_name
* self-host repo support

### Bug Fix
* support kwargs in pipeline
* fix errors in task name definition

## v 0.1.0 (20/05/2022)

First internal release for pipeline inference

* provide basic modules including fileio, logging
* config file parser
* module registry and build, which support group management
* add modules including preprocessor, model and pipeline
* image loading and nlp tokenize support in preprocessor
* add two pipeline: image-matting pipeline and text-classification pipeline
* add task constants according to PRD
* citest support
* makefile and scripts which support packaging whl, build docs, unittest
