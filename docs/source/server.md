# modelscope server使用
## 1. 通用服务
modelscope库基于fastapi开发一个简单模型服务，可以通过一条命令拉起绝大多数模型
使用方法：

```bash
modelscope server --model_id=modelscope/Llama-2-7b-chat-ms --revision=v1.0.5
```
我们提供的官方镜像中也可以一个命令启动(镜像还未完成)
```bash
docker run --rm --name maas_dev --shm-size=50gb --gpus='"device=0"' -e MODELSCOPE_CACHE=/modelscope_cache -v /host_path_to_modelscope_cache:/modelscope_cache -p 8000:8000 reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0-1.9.5-server modelscope server --model_id=modelscope/Llama-2-7b-chat-ms --revision=v1.0.5
```
服务默认监听8000端口，您也可以通过--port改变端口，默认服务提供两个接口，接口文档您可以通过
http://ip:port/docs查看
通过describe接口，可以获取服务输入输出信息以及输入sample数据，如下图：
![describe](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/describe.jpg)
服务调用接口，可以直接拷贝describe接口example示例数据，如下图：
![call](https://modelscope.oss-cn-beijing.aliyuncs.com/resource/call.jpg)

## 2. vllm大模型推理
对于LLM我们提供了vllm推理支持，目前只有部分模型支持vllm。

### 2.1 vllm直接支持modelscope模型
可以通过设置环境变量使得vllm从www.modelscope.cn下载模型。

启动普通server
```bash
VLLM_USE_MODELSCOPE=True python -m vllm.entrypoints.api_server  --model="damo/nlp_gpt2_text-generation_english-base" --revision="v1.0.0"
```
启动openai兼容接口
```bash
VLLM_USE_MODELSCOPE=True python -m vllm.entrypoints.openai.api_server  --model="damo/nlp_gpt2_text-generation_english-base" --revision="v1.0.0"
```

如果模型在modelscope cache目录已经存在，则会直接使用cache中的模型，否则会从www.modelscope.cn下载模型。

通过modelscope官方镜像启动vllm，指定端口为9090

```bash
docker run --rm --name maas_dev --shm-size=50gb --gpus='"device=0"' -e MODELSCOPE_CACHE=/modelscope_cache -v /host_path_to_modelscope_cache:/modelscope_cache -p 9090:9090 reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0-1.9.5-server python -m vllm.entrypoints.api_server --model "modelscope/Llama-2-7b-chat-ms" --revision "v1.0.5" --port 9090
```
