# ms-swift Ascend

> [English](./OVERVIEW.ascend.md) | 中文

ms-swift Ascend 镜像面向华为昇腾 Atlas NPU，提供可直接使用的 ms-swift 运行环境。镜像基于 Ascend CANN 容器镜像构建，包含 Ascend 推理和训练工作流所需的 Python、CANN、TorchNPU、vLLM Ascend、FLA、Megatron、MindSpeed、mcore-bridge、ms-swift 以及 ModelScope 运行组件。

## 快速参考

- 基础镜像：`quay.io/ascend/cann:<cann-version>-<hardware>-<os>-py<python-version>`
- 构建模板：`docker/ascend/Dockerfile.ascend`
- 构建入口：`docker/build_image.py --image_type ascend`
- 默认基础镜像：`quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`
- 支持的基础 OS：Ubuntu 和 openEuler，由 CANN 基础镜像 tag 选择
- 默认输出 tag：`${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-<arch>`
- Ascend runtime 环境来自 `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- 如果镜像内存在 NNAL/ATB，则会加载 `/usr/local/Ascend/nnal/atb/set_env.sh`

## 镜像内容

Ascend Dockerfile 会安装和配置：

```text
| 组件                       | 版本 / 来源 |
|----------------------------|------------|
| CANN                       | 继承自选定的 `quay.io/ascend/cann` 基础镜像；可通过 `--base_image` 配置 |
| Python                     | 继承自基础镜像 tag，例如 `py3.11`；通过 `--base_image` 选择匹配的 Python tag |
| PyTorch                    | 默认 `torch==2.9.0`；可通过 `--torch_version` 配置（同时需要传入匹配的 `--torchvision_version` 和 `--torchaudio_version`） |
| TorchNPU                   | 默认 `torch_npu==2.9.0.post2`；可通过 `--torch_npu_version` 配置，但必须与 PyTorch 基础版本匹配 |
| torchvision / torchaudio  | 默认 `torchvision==0.24.0`、`torchaudio==2.9.0`；覆盖 `--torch_version` 时可通过 `--torchvision_version` 和 `--torchaudio_version` 配置 |
| vLLM                      | 默认从官方 `vllm-project/vllm` 源码 tag `v0.18.0` 构建；可通过 `--vllm_version` 配置；使用 empty target device |
| vLLM Ascend               | 默认从官方 `vllm-project/vllm-ascend` 源码 tag `v0.18.0` 构建；可通过 `--vllm_ascend_version` 配置；会初始化 submodule；构建与运行依赖以华为云 Ascend PyPI 为主索引、官方 PyPI 为备用索引 |
| FLA                       | 从 `fla-org/flash-linear-attention` 源码 checkout，默认 `main` 分支；可通过 `--fla_version` 指定分支或 release tag |
| Megatron-LM               | 源码 checkout，默认分支 `v0.15.3`；可通过 `--megatron_branch` 配置 |
| MindSpeed                 | 源码 checkout，默认分支 `core_r0.15.3`；可通过 `--mindspeed_branch` 配置 |
| mcore-bridge              | PyPI 上的最新发布版；Ascend 构建入口没有独立的版本配置参数 |
| ms-swift                  | 来自 `modelscope/ms-swift` 的源码 checkout，默认分支 `main`；可通过 `--swift_branch` 配置 |
| DeepSpeed                 | 安装满足 `deepspeed>=0.19` 的最新发布包；`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 仅作用于构建时的安装命令 |
| ModelScope                | 通过 `pip install -U modelscope` 安装 PyPI 最新发布包；Ascend 镜像不再 clone ModelScope 或 modelscope-hub 源码仓库 |
| triton-ascend             | 在 vLLM 之前安装，并在所有 editable 安装完成后再次固定版本；安装时以华为云 Ascend PyPI 为主索引、官方 PyPI 为备用索引；CANN `8.5.*` 默认 `3.2.0`、CANN `9.0.*` 默认 `3.2.1`；可通过 `--triton_ascend_version` 配置 |
```

## 镜像 Tag 说明

通过 `docker/build_image.py --image_type ascend` 构建的镜像使用以下 tag 格式：

已发布 tag 索引见 [`docker/ascend/supported_tags.md`](./supported_tags.md)。

```text
${DOCKER_REGISTRY}:<swift-branch>-<cann-version-tag>-torch_npu<TorchNPU-version>-<hardware-tag>-<os-tag>-<python-tag>-<arch>
```

```text
| 字段               | 示例                             | 说明 |
|--------------------|----------------------------------|------|
| `swift-branch`     | `main`                           | 构建镜像时使用的 ms-swift 分支 |
| `cann-version-tag` | `cann8.5.1`、`cann9.0.0`         | 从 CANN 基础镜像 tag 解析 |
| `TorchNPU-version` | `2.9.0.post2`                    | 来自 `--torch_npu_version`，默认 `2.9.0.post2` |
| `hardware-tag`     | `910b`、`a3`、`950`              | 直接从 CANN 基础镜像 tag 解析；当前对应的 Atlas 平台名称为 `a2`、`a3`、`a5`，仅用于文档说明，不参与 tag 映射。 |
| `os-tag`           | `ubuntu22.04`、`openeuler24.03` | 从 CANN 基础镜像 tag 解析；避免不同 OS 的镜像 tag 冲突 |
| `python-tag`       | `py3.11`                         | 从 CANN 基础镜像 tag 解析 |
| `arch`             | `aarch64`、`x86_64`              | 从宿主机架构或 `--arch` 推导 |
```

A2 / CANN 9.0.0 示例（CANN 硬件 tag 为 `910b`）：

```text
${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-aarch64
```

## 最新镜像

最新的 A2 和 A3 镜像发布在 [quay.io/ascend/ms-swift](https://quay.io/repository/ascend/ms-swift?tab=tags)。

**设备 / CANN 基础镜像 / OS / 镜像 Tag / Dockerfile**

- A3 — 9.1.0 — openEuler 24.03 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-a3-openeuler24.03-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A3 — 9.1.0 — Ubuntu 22.04 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A2 — 9.1.0 — openEuler 24.03 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-openeuler24.03-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A2 — 9.1.0 — Ubuntu 22.04 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)

## 本地构建

先设置目标镜像仓库。构建脚本会把 `docker/ascend/Dockerfile.ascend` 渲染成根目录 `Dockerfile`，然后执行构建；Ascend 镜像分支当前不执行 push。

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend
```

完整固定版本参考（CANN 9.1.0、Atlas A2、Ubuntu 22.04、Python 3.12、ARM64）：

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend \
  --base_image quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12 \
  --soc_version ascend910b1 \
  --arch arm \
  --torch_version 2.10.0 \
  --torch_npu_version 2.10.0.post2 \
  --torchvision_version 0.25.0 \
  --torchaudio_version 2.10.0 \
  --vllm_version v0.23.0 \
  --vllm_ascend_version v0.23.0 \
  --fla_version main \
  --triton_ascend_version 3.2.2 \
  --modelscope_branch master \
  --swift_branch v4.5.2 \
  --megatron_branch core_v0.16.0 \
  --mindspeed_branch core_r0.16.0
```

## 自定义构建参数

使用 `--image_type ascend` 选择 Ascend 构建器。当前 Ascend 构建路径实际使用的参数如下：

- `--base_image`（默认：`quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`）：选择 CANN、硬件、OS 和 Python；tag 必须符合 `<cann-version>-<hardware>-<os>-py<python-version>` 格式。
- `--soc_version`（默认：`ascend910_9391`）：设置 vLLM Ascend 构建的目标 SoC 和运行时 `SOC_VERSION`；必须与目标硬件匹配。[vLLM Ascend v0.23.0 安装文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html)给出了以下参考值。这些是 vLLM Ascend 上游的构建目标，不代表本 Dockerfile 已验证全部硬件路径：
  - Atlas A2：`ascend910b1`
  - Atlas A3：`ascend910_9391`
  - Atlas 300I DUO：`ascend310p1`
  - Atlas 950DT：`ascend950dt_9582`
- `--arch`（默认：根据宿主机自动检测）：可选 `arm` 或 `x86`；输出 tag 中会规范化为 `aarch64` 或 `x86_64`。
- `--torch_version`（默认：`2.9.0`）：选择 PyTorch；覆盖时必须同时传入匹配的 `--torchvision_version` 和 `--torchaudio_version`。
- `--torch_npu_version`（默认：`2.9.0.post2`）：基础版本必须与 `--torch_version` 完全一致。
- `--torchvision_version`（默认：`0.24.0`）：选择 torchvision；覆盖 `--torch_version` 时必须显式传入。
- `--torchaudio_version`（默认：`2.9.0`）：选择 torchaudio；覆盖 `--torch_version` 时必须显式传入。
- `--vllm_version`（默认：`0.18.0`）：选择官方 vLLM 源码 tag；可以带或不带前置 `v`。
- `--vllm_ascend_version`（默认：`0.18.0`）：选择官方 vLLM Ascend 源码 tag；可以带或不带前置 `v`。
- `--fla_version`（默认：`main`）：选择 FLA 分支或 release tag。
- `--triton_ascend_version`（默认值取决于 CANN 版本）：CANN 8.5 使用 `3.2.0`，CANN 9.0 使用 `3.2.1`；其他 CANN 系列（包括 9.1）必须显式传入。
- `--pip_extra_index_url`（默认：华为云 Ascend PyPI）：设置 Ascend 特定依赖的主索引。
- `--pypi_official_index_url`（默认：`https://pypi.org/simple`）：设置备用包索引。
- `--swift_branch`（默认：`main`）：选择 ms-swift 源码分支或 tag，并写入输出镜像 tag。
- `--megatron_branch`（默认：`v0.15.3`）：选择 Megatron-LM 源码分支或 tag。
- `--mindspeed_branch`（默认：`core_r0.15.3`）：选择 MindSpeed 源码分支或 tag。

Ascend 镜像的 Python 版本必须通过 `--base_image` 选择，`--python_version`
不会覆盖它。`--modelscope_branch` 虽然会被共用参数解析器接受，但 Ascend
Dockerfile 不使用该参数；当前通过 `pip install -U modelscope` 安装 ModelScope
最新发布包。

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
  ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-aarch64 \
  bash
```

进入容器后可以验证 NPU 和 Python 包：

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import vllm, vllm_ascend; print('vllm ascend ok')"
pip show ms-swift modelscope mcore-bridge torch-npu triton-ascend
```

## 环境变量

```text
| 变量                       | 值 |
|----------------------------|----|
| `SOC_VERSION`              | 选定的 Ascend SoC，例如 `ascend910b1` 或 `ascend910_9391`；可通过 `--soc_version` 配置；用于指定 vLLM Ascend 的构建目标，并保留在运行时环境中（tag 中的硬件字段来自 `--base_image`） |
| `CANN_VERSION`             | 从基础镜像 tag 解析得到 |
| `MEGATRON_LM_PATH`         | `/Megatron-LM` |
| `PYTHONPATH`               | 包含 `/Megatron-LM` |
| `VLLM_USE_MODELSCOPE`      | `True` |
| `LMDEPLOY_USE_MODELSCOPE`  | `True` |
| `MODELSCOPE_CACHE`         | `/mnt/workspace/.cache/modelscope/hub` |
```

## 注意事项

- CANN、firmware 和 driver 版本必须互相兼容。
- Ubuntu 基础镜像通过 `apt-get` 安装系统依赖；openEuler 基础镜像通过 `yum` 安装对应 RPM 包。
- `triton-ascend` 和 vLLM Ascend editable 安装的依赖以 `https://mirrors.huaweicloud.com/ascend/repos/pypi` 为主索引、`https://pypi.org/simple` 为备用索引。pip 会汇总所有已配置索引的候选版本，并不保证严格的仓库优先级；请选择与 CANN、Python 和架构兼容的版本。
- 该镜像面向 Ascend NPU 上的 ms-swift 工作流。依赖安装过程中引入且与 NPU runtime 冲突的 CUDA-only 包会被移除。
- 生产任务建议使用固定镜像 tag，不要依赖浮动分支名。

## License

ms-swift 和 ModelScope 组件遵循各自上游仓库的 license。CANN、MindSpeed、TorchNPU、vLLM Ascend 以及其他预装第三方组件遵循各自上游 license。
