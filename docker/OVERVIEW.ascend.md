# ms-swift Ascend

> English | [中文](./OVERVIEW.ascend.zh.md)

ms-swift Ascend images provide a ready-to-use ms-swift environment for Huawei Ascend Atlas NPUs. The images are built on top of the Ascend CANN container images and include the Python, CANN, PyTorch NPU, vLLM Ascend, Megatron, MindSpeed, mcore-bridge, ms-swift, and ModelScope runtime components needed for Ascend inference and training workflows.

## Quick Reference

- Base image: `quay.io/ascend/cann:<cann-version>-<hardware>-<os>-py<python-version>`
- Build template: `docker/Dockerfile.ascend`
- Build entrypoint: `docker/build_image.py --image_type ascend`
- Default base image: `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`
- Supported base OSes: Ubuntu and openEuler, selected from the CANN base-image tag
- Default output tag: `${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-<arch>`
- Ascend runtime environment is sourced from `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- If available, NNAL/ATB runtime is sourced from `/usr/local/Ascend/nnal/atb/set_env.sh`

## Image Contents

The Ascend Dockerfile installs and configures:

| Component | Version / Source |
| --- | --- |
| CANN | inherited from the selected `quay.io/ascend/cann` base image |
| Python | inherited from the base image tag, for example `py3.11` |
| PyTorch | `torch==2.9.0` by default; configurable with `--torch_version` |
| torch-npu | `torch_npu==2.9.0.post2` by default; configurable with `--torch_npu_version` |
| torchvision / torchaudio | `torchvision==0.24.0`, `torchaudio==2.9.0` by default; pass both explicitly when overriding `--torch_version` |
| vLLM | source install from `vllm-project/vllm`, default `0.18.0`; configurable with `--vllm_version` |
| vLLM Ascend | source install from `vllm-project/vllm-ascend`, default `0.18.0`; configurable with `--vllm_ascend_version` |
| Megatron-LM | source checkout, default branch `v0.15.3` |
| MindSpeed | source checkout, default branch `core_r0.15.3` |
| mcore-bridge | latest release from PyPI |
| ms-swift | source checkout from `modelscope/ms-swift`, default branch `main` |
| ModelScope | source checkout from `modelscope/modelscope`, default branch `master` |
| triton-ascend | CANN `8.5.*` defaults to `3.2.0`; CANN `9.0.*` defaults to `3.2.1`; configurable with `--triton_ascend_version` and installed from the Triton Ascend PyPI index |

## Supported Tag Format

Images built by `docker/build_image.py --image_type ascend` use this tag format:

```text
${DOCKER_REGISTRY}:<swift-branch>-<cann-version-tag>-torch_npu<torch-npu-version>-<atlas-hardware>-<os-tag>-<python-tag>-<arch>
```

| Field | Example | Description |
| --- | --- | --- |
| `swift-branch` | `main` | ms-swift branch used during image build |
| `cann-version-tag` | `cann8.5.1`, `cann9.0.0` | Parsed from the CANN base image tag |
| `torch-npu-version` | `2.9.0.post2` | From `--torch_npu_version`; defaults to `2.9.0.post2` |
| `atlas-hardware` | `a2`, `a3`, `300i`, `a5` | Derived from `--soc_version` |
| `os-tag` | `ubuntu22.04`, `openeuler24.03` | Parsed from the CANN base-image tag; prevents tags for different OSes from colliding |
| `python-tag` | `py3.11` | Parsed from the CANN base image tag |
| `arch` | `aarch64`, `x86_64` | Derived from host architecture or `--arch` |

Default example on an ARM64 host:

```text
${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-aarch64
```

A2 / CANN 9.0.0 example:

```text
${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64
```

## Build Locally

Set the target registry first. The build script renders `docker/Dockerfile.ascend` into the root `Dockerfile`, builds it, and skips push for Ascend images.

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend
```

Build an A2 / CANN 9.0.0 image:

```bash
export DOCKER_REGISTRY=registry.example.com/ms-swift/ms-swift

python docker/build_image.py \
  --image_type ascend \
  --base_image quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.11 \
  --soc_version ascend910b1
```

Build an openEuler image. The system-dependency layer automatically uses `yum`; Ubuntu images continue to use `apt-get`.

```bash
python docker/build_image.py \
  --image_type ascend \
  --base_image quay.io/ascend/cann:8.5.1-a3-openeuler24.03-py3.11 \
  --soc_version ascend910_9391
```

Override the PyTorch stack. `--torch_version` must match the base version of
`--torch_npu_version`; when overriding PyTorch, pass its matching torchvision
and torchaudio versions explicitly.

```bash
python docker/build_image.py \
  --image_type ascend \
  --torch_version 2.9.0 \
  --torch_npu_version 2.9.0.post2 \
  --torchvision_version 0.24.0 \
  --torchaudio_version 2.9.0
```

Override the vLLM stack or triton-ascend. The vLLM version arguments select
the matching Git tag, for example `0.18.0` selects `v0.18.0`.

```bash
python docker/build_image.py \
  --image_type ascend \
  --vllm_version 0.18.0 \
  --vllm_ascend_version 0.18.0 \
  --triton_ascend_version 3.2.1
```

Override Megatron or MindSpeed source branches when needed:

```bash
python docker/build_image.py \
  --image_type ascend \
  --megatron_branch v0.15.3 \
  --mindspeed_branch core_r0.15.3
```

To run the rendered Dockerfile manually, use:

```bash
docker build \
  -t ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64 \
  -f Dockerfile .
```

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
  ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-a2-ubuntu22.04-py3.11-aarch64 \
  bash
```

Inside the container, verify the NPU and Python packages:

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
python -c "import vllm, vllm_ascend; print('vllm ascend ok')"
pip show ms-swift modelscope torch-npu triton-ascend
```

## Environment Variables

| Variable | Value |
| --- | --- |
| `SOC_VERSION` | Selected Ascend SoC version, for example `ascend910b1` or `ascend910_9391` |
| `CANN_VERSION` | Parsed from the base image tag |
| `MEGATRON_LM_PATH` | `/Megatron-LM` |
| `PYTHONPATH` | includes `/Megatron-LM` |
| `VLLM_USE_MODELSCOPE` | `True` |
| `LMDEPLOY_USE_MODELSCOPE` | `True` |
| `MODELSCOPE_CACHE` | `/mnt/workspace/.cache/modelscope/hub` |

## Notes

- CANN, firmware, and driver versions must be compatible with each other.
- Ubuntu base images install system dependencies through `apt-get`; openEuler base images install the corresponding RPM packages through `yum`.
- `triton-ascend` is installed from `https://triton-ascend.osinfra.cn/pypi/simple`; select a version compatible with the chosen CANN, Python, and architecture.
- The image is intended for Ascend NPU ms-swift workflows. CUDA-only packages pulled in by dependencies are removed when they conflict with NPU runtime libraries.
- Use a fixed image tag for production jobs instead of relying on a moving branch name.

## License

ms-swift and ModelScope components follow their upstream repository licenses. CANN, MindSpeed, torch-npu, vLLM Ascend, and other pre-installed third-party components are subject to their own upstream licenses.
