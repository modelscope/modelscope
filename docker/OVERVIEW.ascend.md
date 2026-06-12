# ms-swift Ascend

> English | [中文](./OVERVIEW.ascend.zh.md)

ms-swift Ascend images provide a ready-to-use ms-swift environment for Huawei Ascend Atlas NPUs. The images are built on top of the Ascend CANN container images and include the Python, CANN, PyTorch NPU, vLLM Ascend, Megatron, MindSpeed, mcore-bridge, ms-swift, and ModelScope runtime components needed for Ascend inference and training workflows.

## Quick Reference

- Base image: `quay.io/ascend/cann:<cann-version>-<hardware>-<os>-py<python-version>`
- Build template: `docker/Dockerfile.ascend`
- Build entrypoint: `docker/build_image.py --image_type ascend`
- Default base image: `quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11`
- Default output tag: `${DOCKER_REGISTRY}:main-A3-py311-CANN8.5.1-ubuntu22.04-<arch>`
- Ascend runtime environment is sourced from `/usr/local/Ascend/ascend-toolkit/set_env.sh`
- If available, NNAL/ATB runtime is sourced from `/usr/local/Ascend/nnal/atb/set_env.sh`

## Image Contents

The Ascend Dockerfile installs and configures:

| Component | Version / Source |
| --- | --- |
| CANN | inherited from the selected `quay.io/ascend/cann` base image |
| Python | inherited from the base image tag, for example `py3.11` |
| PyTorch | `torch==2.9.0` |
| torch-npu | `torch_npu==2.9.0.post2` |
| torchvision / torchaudio | `torchvision==0.24.0`, `torchaudio==2.9.0` |
| vLLM | source install from `vllm-project/vllm`, default branch `v0.18.0` |
| vLLM Ascend | source install from `vllm-project/vllm-ascend`, default branch `v0.18.0` |
| Megatron-LM | source checkout, default branch `v0.15.3` |
| MindSpeed | source checkout, default branch `core_r0.15.3` |
| mcore-bridge | source checkout from `modelscope/mcore-bridge` |
| ms-swift | source checkout from `modelscope/ms-swift`, default branch `main` |
| ModelScope | source checkout from `modelscope/modelscope`, default branch `master` |
| triton-ascend | `3.2.0` for CANN `8.5.*`; local wheel install of `3.2.1` for CANN `9.0.0` |

## Supported Tag Format

Images built by `docker/build_image.py --image_type ascend` use this tag format:

```text
${DOCKER_REGISTRY}:<swift-branch>-<atlas-hardware>-<python-tag>-<cann-version-tag>-<os-tag>-<arch>
```

| Field | Example | Description |
| --- | --- | --- |
| `swift-branch` | `main` | ms-swift branch used during image build |
| `atlas-hardware` | `A2`, `A3`, `300I`, `A5` | Derived from `--soc_version` |
| `python-tag` | `py311` | Derived from `--python_version` |
| `cann-version-tag` | `CANN8.5.1`, `CANN9.0.0` | Parsed from the CANN base image tag |
| `os-tag` | `ubuntu22.04` | Parsed from the CANN base image tag |
| `arch` | `arm`, `x86` | Derived from host architecture or `--arch` |

Default example on an ARM64 host:

```text
${DOCKER_REGISTRY}:main-A3-py311-CANN8.5.1-ubuntu22.04-arm
```

A2 / CANN 9.0.0 example:

```text
${DOCKER_REGISTRY}:main-A2-py311-CANN9.0.0-ubuntu22.04-arm
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

Override Megatron or MindSpeed source branches when needed:

```bash
python docker/build_image.py \
  --image_type ascend \
  --megatron_branch v0.15.3 \
  --mindspeed_branch core_r0.15.3
```

For slow networks, Linux hosts can use Docker host networking after the root `Dockerfile` is generated:

```bash
docker build --network host \
  -t ${DOCKER_REGISTRY}:main-A2-py311-CANN9.0.0-ubuntu22.04-arm \
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
  ${DOCKER_REGISTRY}:main-A2-py311-CANN9.0.0-ubuntu22.04-arm \
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
- CANN `8.5.*` and CANN `9.0.0` use different `triton-ascend` install paths in this Dockerfile.
- The image is intended for Ascend NPU ms-swift workflows. CUDA-only packages pulled in by dependencies are removed when they conflict with NPU runtime libraries.
- Use a fixed image tag for production jobs instead of relying on a moving branch name.

## License

ms-swift and ModelScope components follow their upstream repository licenses. CANN, MindSpeed, torch-npu, vLLM Ascend, and other pre-installed third-party components are subject to their own upstream licenses.
