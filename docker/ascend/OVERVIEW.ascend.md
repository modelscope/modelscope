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
| mcore-bridge             | latest release from PyPI; no dedicated version option in the Ascend build entrypoint |
| ms-swift                 | source checkout from `modelscope/ms-swift`, default branch `main`; configurable with `--swift_branch` |
| DeepSpeed                | newest package satisfying `deepspeed>=0.19`; `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is scoped to its build-time install command |
| ModelScope               | latest published package resolved by `pip install -U modelscope`; the Ascend image does not clone ModelScope or modelscope-hub source repositories |
| triton-ascend            | Installed before vLLM and re-pinned after all editable installs, using Huawei Cloud Ascend PyPI as the primary index and official PyPI as fallback; CANN `8.5.*` defaults to `3.2.0`, CANN `9.0.*` to `3.2.1`; configurable with `--triton_ascend_version` |
```

## Supported Tag Format

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

Default example on an ARM64 host:

```text
${DOCKER_REGISTRY}:main-cann8.5.1-torch_npu2.9.0.post2-a3-ubuntu22.04-py3.11-aarch64
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
  --mindspeed_branch core_r0.16.0
```

For Ascend builds, CANN, hardware, OS, and Python versions come from
`--base_image`; `--python_version` does not override the Python tag. The
`--soc_version` value selects the target SoC used when building vLLM Ascend and
remains available in the runtime environment, while the hardware field in the
image tag comes directly from `--base_image`.
The FLA clone uses the `main` branch by default; pass `--fla_version` with a
branch name or release tag such as `v0.5.2` to select another revision.

Build an A2 / CANN 9.0.0 image (`910b` CANN hardware tag):

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

Override the vLLM source tags or triton-ascend version. Both vLLM components
are built from their official `v0.18.0` source tags. `triton-ascend` is first
installed before vLLM and then re-pinned after all editable installs so that
dependencies cannot replace the selected CANN-specific version. Both steps use
Huawei Cloud Ascend PyPI as the primary index and official PyPI as fallback.
The same index pair is used while installing vLLM Ascend and its dependencies.

```bash
python docker/build_image.py \
  --image_type ascend \
  --vllm_version 0.18.0 \
  --vllm_ascend_version 0.18.0 \
  --triton_ascend_version 3.2.1 \
  --pip_extra_index_url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --pypi_official_index_url https://pypi.org/simple
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
  -t ${DOCKER_REGISTRY}:main-cann9.0.0-torch_npu2.9.0.post2-910b-ubuntu22.04-py3.11-aarch64 \
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
