# ms-swift Ascend

> [English](./OVERVIEW.ascend.md) | 中文

ms-swift Ascend 镜像面向华为昇腾 Atlas NPU，提供可直接使用的 ms-swift 运行环境。镜像基于 Ascend CANN 容器镜像构建，包含 Ascend 推理和训练工作流所需的 Python、CANN、PyTorch NPU、vLLM Ascend、Megatron、MindSpeed、mcore-bridge、ms-swift 以及 ModelScope 运行组件。

## 快速参考

- 基础镜像：`quay.io/ascend/cann:<cann-version>-<hardware>-<os>-py<python-version>`
- 构建模板：`docker/Dockerfile.ascend`
- 构建入口：`docker/build_image.py --image_type ascend`
- 默认基础镜像：`quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`
- 支持的基础 OS：Ubuntu 和 openEuler，由 CANN 基础镜像 tag 选择
- 默认输出 tag：`${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-<arch>`
- Ascend runtime 环境来自 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 如果镜像内存在 NNAL/ATB，则会加载 `/usr/local/Ascend/nnal/atb/set_env.sh`

## 镜像内容

Ascend Dockerfile 会安装和配置：

| 组件 | 版本 / 来源 |
| --- | --- |
| CANN | 继承自选定的 `quay.io/ascend/cann` 基础镜像 |
| Python | 继承自基础镜像 tag，例如 `py3.11` |
| PyTorch | 默认 `torch==2.9.0`；可通过 `--torch_version` 配置 |
| torch-npu | 默认 `torch_npu==2.9.0.post2`；可通过 `--torch_npu_version` 配置 |
| torchvision / torchaudio | 默认 `torchvision==0.24.0`、`torchaudio==2.9.0`；覆盖 `--torch_version` 时必须同时显式传入两者 |
| vLLM | 从 `vllm-project/vllm` 源码安装，默认 `0.18.0`；可通过 `--vllm_version` 配置 |
| vLLM Ascend | 从 `vllm-project/vllm-ascend` 源码安装，默认 `0.18.0`；可通过 `--vllm_ascend_version` 配置 |
| Megatron-LM | 源码 checkout，默认分支 `v0.15.3` |
| MindSpeed | 源码 checkout，默认分支 `core_r0.15.3` |
| mcore-bridge | PyPI 上的最新发布版 |
| ms-swift | 来自 `modelscope/ms-swift` 的源码 checkout，默认分支 `main` |
| ModelScope | 来自 `modelscope/modelscope` 的源码 checkout，默认分支 `master` |
| triton-ascend | CANN `8.5.*` 默认 `3.2.0`；CANN `9.0.*` 默认 `3.2.1`；可通过 `--triton_ascend_version` 配置，并从 Triton Ascend PyPI 源安装 |

## 支持的 Tag 格式

通过 `docker/build_image.py --image_type ascend` 构建的镜像使用以下 tag 格式：

```text
${DOCKER_REGISTRY}:<swift-branch>-<cann-version-tag>-torch_npu<torch-npu-version>-<atlas-hardware>-<os-tag>-<python-tag>-<arch>
```

| 字段 | 示例 | 说明 |
| --- | --- | --- |
| `swift-branch` | `main` | 构建镜像时使用的 ms-swift 分支 |
| `cann-version-tag` | `cann8.5.1`、`cann9.0.0` | 从 CANN 基础镜像 tag 解析 |
| `torch-npu-version` | `2.9.0.post2` | 来自 `--torch_npu_version`，默认 `2.9.0.post2` |
| `atlas-hardware` | `a2`、`a3`、`300i`、`a5` | 从 `--soc_version` 推导 |
| `os-tag` | `ubuntu22.04`、`openeuler24.03` | 从 CANN 基础镜像 tag 解析；避免不同 OS 的镜像 tag 冲突 |
| `python-tag` | `py3.11` | 从 CANN 基础镜像 tag 解析 |
| `arch` | `aarch64`、`x86_64` | 从宿主机架构或 `--arch` 推导 |

ARM64 宿主机上的默认示例：

```text
${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-aarch64
```

A2 / CANN 9.0.0 示例：

```text
${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64
```

## 本地构建

先设置目标镜像仓库。构建脚本会把 `docker/Dockerfile.ascend` 渲染成根目录 `Dockerfile`，然后执行构建；Ascend 镜像分支当前不执行 push。

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend
```

构建 A2 / CANN 9.0.0 镜像：

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend \
  --base_image quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11 \
  --soc_version ascend910b1
```

构建 openEuler 镜像。系统依赖层会自动使用 `yum`，Ubuntu 镜像继续使用 `apt-get`：

```bash
python docker/build_image.py \
  --image_type ascend \
  --base_image quay.io/ascend/cann:8.5.1-a3-openeuler24.03-py3.11 \
  --soc_version ascend910_9391
```

覆盖 PyTorch 版本组。`--torch_version` 必须与 `--torch_npu_version` 的基础版本一致；覆盖 PyTorch 时，必须显式传入匹配的 torchvision 和 torchaudio 版本：

```bash
python docker/build_image.py \
  --image_type ascend \
  --torch_version 2.9.0 \
  --torch_npu_version 2.9.0.post2 \
  --torchvision_version 0.24.0 \
  --torchaudio_version 2.9.0
```

覆盖 vLLM 版本组或 triton-ascend。vLLM 版本参数会选择对应 Git tag，例如 `0.18.0` 会选择 `v0.18.0`。

```bash
python docker/build_image.py \
  --image_type ascend \
  --vllm_version 0.18.0 \
  --vllm_ascend_version 0.18.0 \
  --triton_ascend_version 3.2.1
```

需要时可以覆盖 Megatron 或 MindSpeed 源码分支：

```bash
python docker/build_image.py \
  --image_type ascend \
  --megatron_branch v0.15.3 \
  --mindspeed_branch core_r0.15.3
```

如需手工构建生成后的根目录 `Dockerfile`，可使用：

```bash
docker build \
  -t ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64 \
  -f Dockerfile .
```

## 运行 Ascend 容器

宿主机需要提前安装兼容的 Ascend driver 和 firmware。容器通过挂载宿主机 NPU 设备和 driver 库使用昇腾硬件。

```bash
docker run --rm -it \
  --name ms_swift_ascend \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /mnt/workspace:/mnt/workspace \
  ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64 \
  bash
```

进入容器后可以验证 NPU 和 Python 包：

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import vllm, vllm_ascend; print('vllm ascend ok')"
pip show ms-swift modelscope torch-npu triton-ascend
```

## 环境变量

| 变量 | 值 |
| --- | --- |
| `SOC_VERSION` | 选定的 Ascend SoC，例如 `ascend910b1` 或 `ascend910_9391` |
| `CANN_VERSION` | 从基础镜像 tag 解析得到 |
| `MEGATRON_LM_PATH` | `/Megatron-LM` |
| `PYTHONPATH` | 包含 `/Megatron-LM` |
| `VLLM_USE_MODELSCOPE` | `True` |
| `LMDEPLOY_USE_MODELSCOPE` | `True` |
| `MODELSCOPE_CACHE` | `/mnt/workspace/.cache/modelscope/hub` |

## 注意事项

- CANN、firmware 和 driver 版本必须互相兼容。
- Ubuntu 基础镜像通过 `apt-get` 安装系统依赖；openEuler 基础镜像通过 `yum` 安装对应 RPM 包。
- `triton-ascend` 从 `https://triton-ascend.osinfra.cn/pypi/simple` 安装；请选择与 CANN、Python 和架构兼容的版本。
- 该镜像面向 Ascend NPU 上的 ms-swift 工作流。依赖安装过程中引入且与 NPU runtime 冲突的 CUDA-only 包会被移除。
- 生产任务建议使用固定镜像 tag，不要依赖浮动分支名。

## License

ms-swift 和 ModelScope 组件遵循各自上游仓库的 license。CANN、MindSpeed、torch-npu、vLLM Ascend 以及其他预装第三方组件遵循各自上游 license。
