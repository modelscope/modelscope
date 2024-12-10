# ModelScope command line usage
## Supported commands
```bash
modelscope --help
usage: modelscope <command> [<args>]

positional arguments:
  {download,plugin,pipeline,modelcard,model,server,login}
                        modelscope commands helpers

options:
  -h, --help            show this help message and exit

```
## login
```bash
modelscope login --help
usage: modelscope <command> [<args>] login [-h] --token TOKEN

options:
  -h, --help     show this help message and exit
  --token TOKEN  The Access Token for modelscope.
```
Get access token: [我的页面](https://modelscope.cn/my/myaccesstoken)获取**SDK 令牌**


## download model
```bash
modelscope download --help

    usage: modelscope <command> [<args>] download [-h] --model MODEL [--revision REVISION] [--cache_dir CACHE_DIR] [--local_dir LOCAL_DIR] [--include [INCLUDE ...]] [--exclude [EXCLUDE ...]] [files ...]

    positional arguments:
      files                 Specify relative path to the repository file(s) to download.(e.g 'tokenizer.json', 'onnx/decoder_model.onnx').

    options:
      -h, --help            show this help message and exit
      --model MODEL         The model id to be downloaded.
      --revision REVISION   Revision of the model.
      --cache_dir CACHE_DIR
                            Cache directory to save model.
      --local_dir LOCAL_DIR
                            File will be downloaded to local location specified bylocal_dir, in this case, cache_dir parameter will be ignored.
      --include [INCLUDE ...]
                            Glob patterns to match files to download.Ignored if file is specified
      --exclude [EXCLUDE ...]
                            Glob patterns to exclude from files to download.Ignored if file is specified
```
## Usage Examples

Command Examples（[gpt2](https://www.modelscope.cn/models/AI-ModelScope/gpt2/files)）

### Specify downloading of a single file
```bash
    modelscope download --model 'AI-ModelScope/gpt2' 64.tflite
```

### Specify multiple files to download
```bash
    modelscope download --model 'AI-ModelScope/gpt2' 64.tflite config.json
```
### Specify certain files to download 
```bash
    modelscope download --model 'AI-ModelScope/gpt2' --include 'onnx/*' '*.tflite'
```
### Filter specified files
```bash
    modelscope download --model 'AI-ModelScope/gpt2' --exclude 'onnx/*' '*.tflite' 
```
### Specify the download cache directory
```bash
    modelscope download --model 'AI-ModelScope/gpt2' --include '*.json' --cache_dir './cache_dir'
```
   The model files will be downloaded to cache\_dir/AI-ModelScope/gpt2/

### Specify the local directory for downloading    
```bash
    modelscope download --model 'AI-ModelScope/gpt2' --include '*.json' --cache_dir './local_dir'
```
  The model files will be downloaded to ./local\_dir

If both the local directory and the cache directory are specified, the local directory will take precedence.

## model operation
Supports creating models and uploading model files.
```bash
modelscope model --help
usage: modelscope <command> [<args>] modelcard [-h] [-tk ACCESS_TOKEN] -act {create,upload,download} [-gid GROUP_ID] -mid MODEL_ID [-vis VISIBILITY] [-lic LICENSE] [-ch CHINESE_NAME] [-md MODEL_DIR] [-vt VERSION_TAG] [-vi VERSION_INFO]

options:
  -h, --help            show this help message and exit
  -tk ACCESS_TOKEN, --access_token ACCESS_TOKEN
                        the certification of visit ModelScope
  -act {create,upload,download}, --action {create,upload,download}
                        the action of api ModelScope[create, upload]
  -gid GROUP_ID, --group_id GROUP_ID
                        the group name of ModelScope, eg, damo
  -mid MODEL_ID, --model_id MODEL_ID
                        the model name of ModelScope
  -vis VISIBILITY, --visibility VISIBILITY
                        the visibility of ModelScope[PRIVATE: 1, INTERNAL:3, PUBLIC:5]
  -lic LICENSE, --license LICENSE
                        the license of visit ModelScope[Apache License 2.0|GPL-2.0|GPL-3.0|LGPL-2.1|LGPL-3.0|AFL-3.0|ECL-2.0|MIT]
  -ch CHINESE_NAME, --chinese_name CHINESE_NAME
                        the chinese name of ModelScope
  -md MODEL_DIR, --model_dir MODEL_DIR
                        the model_dir of configuration.json
  -vt VERSION_TAG, --version_tag VERSION_TAG
                        the tag of uploaded model
  -vi VERSION_INFO, --version_info VERSION_INFO
                        the info of uploaded model
```

### Create model
```bash
    modelscope model -act create -gid 'YOUR_GROUP_ID' -mid 'THE_MODEL_ID' -vis 1 -lic 'MIT' -ch '中文名字'
```
Will create model THE_MODEL_ID in www.modelscope.cn

### Upload model files
```bash
    modelscope model -act upload -gid 'YOUR_GROUP_ID' -mid 'THE_MODEL_ID' -md modelfiles/ -vt 'v0.0.1' -vi 'upload model files'
```

## Pipeline
Create the template files needed for pipeline.

```bash
modelscope pipeline --help
usage: modelscope <command> [<args>] pipeline [-h] -act {create} [-tpl TPL_FILE_PATH] [-s SAVE_FILE_PATH] [-f FILENAME] -t TASK_NAME [-m MODEL_NAME] [-p PREPROCESSOR_NAME] [-pp PIPELINE_NAME] [-config CONFIGURATION_PATH]

options:
  -h, --help            show this help message and exit
  -act {create}, --action {create}
                        the action of command pipeline[create]
  -tpl TPL_FILE_PATH, --tpl_file_path TPL_FILE_PATH
                        the template be selected for ModelScope[template.tpl]
  -s SAVE_FILE_PATH, --save_file_path SAVE_FILE_PATH
                        the name of custom template be saved for ModelScope
  -f FILENAME, --filename FILENAME
                        the init name of custom template be saved for ModelScope
  -t TASK_NAME, --task_name TASK_NAME
                        the unique task_name for ModelScope
  -m MODEL_NAME, --model_name MODEL_NAME
                        the class of model name for ModelScope
  -p PREPROCESSOR_NAME, --preprocessor_name PREPROCESSOR_NAME
                        the class of preprocessor name for ModelScope
  -pp PIPELINE_NAME, --pipeline_name PIPELINE_NAME
                        the class of pipeline name for ModelScope
  -config CONFIGURATION_PATH, --configuration_path CONFIGURATION_PATH
                        the path of configuration.json for ModelScope
```

### Create pipeline files
```bash
    modelscope pipeline -act 'create' -t 'THE_PIPELINE_TASK' -m 'THE_MODEL_NAME' -pp 'THE_PIPELINE_NAME'
```
