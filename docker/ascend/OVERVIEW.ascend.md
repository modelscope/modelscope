# ms-swift Ascend

> English | [中文](./OVERVIEW.ascend.zh.md)

ms-swift Ascend images provide a ready-to-use ms-swift environment for Huawei Ascend Atlas NPUs. The images are built on top of the Ascend CANN container images and include the Python, CANN, TorchNPU, vLLM Ascend, FLA, Megatron, MindSpeed, mcore-bridge, ms-swift, and ModelScope runtime components needed for Ascend inference and training workflows.

## Quick Reference

- Base image: `quay.io/ascend/cann:<cann-version>-<hardware>-<os>-py<python-version>`
- Build template: `docker/ascend/Dockerfile.ascend`
- Build entrypoint: `docker/build_image.py --image_type ascend`
- Default base image: `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`
- Supported base OSes: Ubuntu and openEuler, selected from the CANN base-image tag
- Default output tag: `${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-<arch>`
- Ascend runtime environment is sourced from `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- If available, NNAL/ATB runtime is sourced from `/usr/local/Ascend/nnal/atb/set_env.sh`

## Image Contents

The Ascend Dockerfile installs and configures:

```text
| Component                | Version / Source |
|--------------------------|------------------|
| CANN                     | inherited from the selected `quay.io/ascend/cann` base image; configurable with `--base_image` |
| Python                   | inherited from the base image tag, for example `py3.11`; configure it by selecting a matching tag with `--base_image` |
| PyTorch                  | `torch==2.9.0` by default; configurable with `--torch_version` (also pass matching `--torchvision_version` and `--torchaudio_version`) |
| TorchNPU                 | `torch_npu==2.9.0.post2` by default; configurable with `--torch_npu_version`, which must match the PyTorch base version |
| torchvision / torchaudio | `torchvision==0.24.0`, `torchaudio==2.9.0` by default; configurable with `--torchvision_version` and `--torchaudio_version` when overriding `--torch_version` |
| vLLM                     | Official `vllm-project/vllm` source tag `v0.18.0` by default; configurable with `--vllm_version`; built with the empty target device |
| vLLM Ascend              | Official `vllm-project/vllm-ascend` source tag `v0.18.0` by default; configurable with `--vllm_ascend_version`; initializes submodules; resolves build and runtime dependencies from Huawei Cloud Ascend PyPI with official PyPI as fallback |
| FLA                      | source checkout from `fla-org/flash-linear-attention`, default branch `main`; configurable with `--fla_version` using a branch or release tag |
| Megatron-LM              | source checkout, default branch `v0.15.3`; configurable with `--megatron_branch` |
| MindSpeed                | source checkout, default branch `core_r0.15.3`; configurable with `--mindspeed_branch` |
| mcore-bridge             | editable source checkout from `modelscope/mcore-bridge`, default branch `main`; configurable with `--mcore_bridge_branch` using a branch or release tag |
| ms-swift                 | source checkout from `modelscope/ms-swift`, default branch `main`; configurable with `--swift_branch` |
| DeepSpeed                | newest package satisfying `deepspeed>=0.19`; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is scoped to its build-time install command |
| ModelScope               | latest published package resolved by `pip install -U modelscope`; the Ascend image does not clone ModelScope or modelscope-hub source repositories |
| triton-ascend            | Installed before vLLM and re-pinned after all editable installs, using Huawei Cloud Ascend PyPI as the primary index and official PyPI as fallback; CANN `8.5.*` defaults to `3.2.0`, CANN `9.0.*` to `3.2.1`; configurable with `--triton_ascend_version` |
```

## Image Tag Description

Images built by `docker/build_image.py --image_type ascend` use this tag format:

The published-tag index is maintained in [`docker/ascend/supported_tags.md`](./supported_tags.md).

```text
${DOCKER_REGISTRY}:<swift-branch>-<cann-version-tag>-torch_npu<TorchNPU-version>-<hardware-tag>-<os-tag>-<python-tag>-<arch>
```

```text
| Field              | Example                         | Description |
|--------------------|---------------------------------|-------------|
| `swift-branch`     | `main`                          | ms-swift branch used during image build |
| `cann-version-tag` | `cann8.5.1`, `cann9.0.0`        | Parsed from the CANN base image tag |
| `TorchNPU-version` | `2.9.0.post2`                   | From `--torch_npu_version`; defaults to `2.9.0.post2` |
| `hardware-tag`     | `910b`, `a3`, `950`             | Parsed directly from the CANN base-image tag. The current Atlas platform names are `a2`, `a3`, and `a5`; these are documentation labels only. |
| `os-tag`           | `ubuntu22.04`, `openeuler24.03` | Parsed from the CANN base-image tag; prevents tags for different OSes from colliding |
| `python-tag`       | `py3.11`                        | Parsed from the CANN base image tag |
| `arch`             | `aarch64`, `x86_64`             | Derived from host architecture or `--arch` |
```

A2 / CANN 9.0.0 example (`910b` CANN hardware tag):

```text
${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-aarch64
```

## Latest Images

The latest A2 and A3 images are hosted at [quay.io/ascend/ms-swift](https://quay.io/repository/ascend/ms-swift?tab=tags).

**Device / CANN Base Image / OS / Image Tag / Dockerfile**

- A3 — 9.1.0 — openEuler 24.03 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-a3-openeuler24.03-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A3 — 9.1.0 — Ubuntu 22.04 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-a3-ubuntu22.04-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A2 — 9.1.0 — openEuler 24.03 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-openeuler24.03-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)
- A2 — 9.1.0 — Ubuntu 22.04 — `v4.5.2-cann9.1.0-torch_npu2.10.0.post2-910b-ubuntu22.04-py3.12` — [Dockerfile.ascend](./Dockerfile.ascend)

## Build Locally

Set the target registry first. The build script renders `docker/ascend/Dockerfile.ascend` into the root `Dockerfile`, builds it, and skips push for Ascend images.

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend
```

Complete version-pinned reference (CANN 9.1.0, Atlas A2, Ubuntu 22.04,
Python 3.12, ARM64):

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
  --mindspeed_branch core_r0.16.0 \
  --mcore_bridge_branch v1.6.2
```

## Custom Build Parameters

Use `--image_type ascend` to select the Ascend builder. The following options
are consumed by the current Ascend build path:

- `--base_image` (default: `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`): selects CANN, hardware, OS, and Python. The tag must follow `<cann-version>-<hardware>-<os>-py<python-version>`.
- `--soc_version` (default: `ascend910_9391`): sets the target SoC for the vLLM Ascend build and the runtime `SOC_VERSION`; it must match the target hardware. The [vLLM Ascend v0.23.0 installation guide](https://docs.vllm.ai/projects/ascend/en/v0.23.0/installation.html) provides the reference values below. These are upstream vLLM Ascend build targets, not a claim that every hardware path has been verified by this Dockerfile:
  - Atlas A2: `ascend910b1`
  - Atlas A3: `ascend910_9391`
  - Atlas 300I DUO: `ascend310p1`
  - Atlas 950DT: `ascend950dt_9582`
- `--arch` (default: detected from the host): accepts `arm` or `x86`; normalized to `aarch64` or `x86_64` in the output tag.
- `--torch_version` (default: `2.9.0`): selects PyTorch. If overridden, matching `--torchvision_version` and `--torchaudio_version` are required.
- `--torch_npu_version` (default: `2.9.0.post2`): its base version must exactly match `--torch_version`.
- `--torchvision_version` (default: `0.24.0`): selects torchvision; required when `--torch_version` is overridden.
- `--torchaudio_version` (default: `2.9.0`): selects torchaudio; required when `--torch_version` is overridden.
- `--vllm_version` (default: `0.18.0`): selects the official vLLM source tag; values with or without the leading `v` are accepted.
- `--vllm_ascend_version` (default: `0.18.0`): selects the official vLLM Ascend source tag; values with or without the leading `v` are accepted.
- `--fla_version` (default: `main`): selects the FLA branch or release tag.
- `--triton_ascend_version` (CANN-specific default): uses `3.2.0` for CANN 8.5 and `3.2.1` for CANN 9.0. Other CANN series, including 9.1, require an explicit value.
- `--pip_extra_index_url` (default: Huawei Cloud Ascend PyPI): sets the primary package index for Ascend-specific dependencies.
- `--pypi_official_index_url` (default: `https://pypi.org/simple`): sets the fallback package index.
- `--swift_branch` (default: `main`): selects the ms-swift source branch or tag and is included in the output image tag.
- `--megatron_branch` (default: `v0.15.3`): selects the Megatron-LM source branch or tag.
- `--mindspeed_branch` (default: `core_r0.15.3`): selects the MindSpeed source branch or tag.
- `--mcore_bridge_branch` (default: `main`): selects the mcore-bridge source branch or release tag for the editable install.

`--python_version` does not override the Python version for Ascend images;
select it through `--base_image`. `--modelscope_branch` is accepted by the
shared argument parser but is not consumed by the Ascend Dockerfile, which
installs the latest published ModelScope package with `pip install -U
modelscope`.

## Run An Ascend Container

The host must have a compatible Ascend driver and firmware installed. The container uses the host NPU devices and driver libraries.

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

Inside the container, verify the NPU and Python packages:

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import vllm, vllm_ascend; print('vllm ascend ok')"
pip show ms-swift modelscope mcore-bridge torch-npu triton-ascend
```

## Environment Variables

```text
| Variable                  | Value |
|---------------------------|-------|
| `SOC_VERSION`             | Selected Ascend SoC version, for example `ascend910b1` or `ascend910_9391`; configurable with `--soc_version`; used as the vLLM Ascend build target and retained in the runtime environment (the tag hardware comes from `--base_image`) |
| `CANN_VERSION`            | Parsed from the base image tag |
| `MEGATRON_LM_PATH`        | `/Megatron-LM` |
| `PYTHONPATH`               | includes `/Megatron-LM` |
| `VLLM_USE_MODELSCOPE`     | `True` |
| `LMDEPLOY_USE_MODELSCOPE` | `True` |
| `MODELSCOPE_CACHE`        | `/mnt/workspace/.cache/modelscope/hub` |
```

## Notes

- CANN, firmware, and driver versions must be compatible with each other.
- Ubuntu base images install system dependencies through `apt-get`; openEuler base images install the corresponding RPM packages through `yum`.
- `triton-ascend` and the dependencies of the editable vLLM Ascend install use `https://mirrors.huaweicloud.com/ascend/repos/pypi` as the primary index and `https://pypi.org/simple` as fallback. pip evaluates candidates from all configured indexes rather than enforcing strict repository priority. Select versions compatible with the chosen CANN, Python, and architecture.
- The image is intended for Ascend NPU ms-swift workflows. CUDA-only packages pulled in by dependencies are removed when they conflict with NPU runtime libraries.
- Use a fixed image tag for production jobs instead of relying on a moving branch name.

## License

ms-swift and ModelScope components follow their upstream repository licenses. CANN, MindSpeed, TorchNPU, vLLM Ascend, and other pre-installed third-party components are subject to their own upstream licenses.
