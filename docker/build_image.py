import argparse
import os
import platform
import re
import subprocess
import urllib.error
import urllib.request
from copy import copy
from datetime import datetime
from typing import Any, List, Optional

import json

docker_registry = os.environ['DOCKER_REGISTRY']
assert docker_registry, 'You must pass a valid DOCKER_REGISTRY'
timestamp = datetime.now()
formatted_time = timestamp.strftime('%Y%m%d%H%M%S')
VLLM_ROCM_REPO = 'vllm/vllm-openai-rocm'
_FLOATING_ROCM_TAGS = frozenset({
    'latest',
    'latest-base',
    'nightly',
    'base-nightly',
})
_VERSION_TAG_PATTERN = re.compile(r'^v\d+(?:\.\d+)*$')
_NIGHTLY_HASH_PATTERN = re.compile(r'^(?:base-)?nightly-[0-9a-f]{7,40}$')


class Builder:

    def __init__(self, args: Any, dry_run: bool):
        self.args = self.init_args(args)
        self.dry_run = dry_run
        self.args.cudatoolkit_version = self._generate_cudatoolkit_version(
            args.cuda_version)
        self.args.python_tag = self._generate_python_tag(args.python_version)

    def init_args(self, args: Any) -> Any:
        if not args.base_image:
            # A mirrored image of nvidia/cuda:12.4.0-devel-ubuntu22.04
            args.base_image = 'nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04'
        if not args.torch_version:
            args.torch_version = '2.10.0'
            args.torchaudio_version = '2.10.0'
            args.torchvision_version = '0.25.0'
        if not args.optimum_version:
            args.optimum_version = '2.0.0'
        if not args.tf_version:
            args.tf_version = '2.16.1'
        if not args.cuda_version:
            args.cuda_version = '12.8.1'
        if not args.vllm_version:
            args.vllm_version = '0.19.1'
        if not args.lmdeploy_version:
            args.lmdeploy_version = '0.11.0'
        if not args.autogptq_version:
            args.autogptq_version = '0.7.1'
        if not args.flashattn_version:
            args.flashattn_version = '2.8.3'
        return args

    def _generate_cudatoolkit_version(self, cuda_version: str) -> str:
        cuda_version = cuda_version[:cuda_version.rfind('.')]
        return 'cu' + cuda_version.replace('.', '')

    def _generate_python_tag(self, python_version: str) -> str:
        python_version = python_version[:python_version.rfind('.')]
        return 'py' + python_version.replace('.', '')

    def generate_dockerfile(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _remove_pynini_related_dependency(content: str) -> str:
        return content.replace(
            'pip install --no-cache-dir funtextprocessing typeguard==2.13.3 scikit-learn -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html &&',  # noqa: E501
            'pip install --no-cache-dir typeguard==2.13.3 scikit-learn -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html &&'  # noqa: E501
        )

    def _save_dockerfile(self, content: str) -> None:
        if os.path.exists('./Dockerfile'):
            os.remove('./Dockerfile')
        with open('./Dockerfile', 'w') as f:
            f.write(content)

    def run_cmd(self, *args: str) -> int:
        """Run a shell command safely via subprocess (no shell=True).

        Args:
            *args: Command and its arguments as separate strings, e.g.
                   ``self.run_cmd('docker', 'build', '-t', tag, '.')``.

        Returns:
            The process return code (0 on success).
        """
        result = subprocess.run(list(args), check=False)
        return result.returncode

    def build(self) -> int:
        pass

    def push(self) -> int:
        pass

    def image(self) -> str:
        pass

    def __call__(self):
        content = self.generate_dockerfile()
        self._save_dockerfile(content)
        if not self.dry_run:
            ret = self.build()
            if ret != 0:
                raise RuntimeError(f'Docker build error with errno: {ret}')

            ret = self.push()
            if ret != 0:
                raise RuntimeError(f'Docker push error with errno: {ret}')

            if self.args.ci_image != 0:
                ret = self.run_cmd('docker', 'tag', self.image(),
                                   f'{docker_registry}:ci_image')
                if ret != 0:
                    raise RuntimeError(
                        f'Docker tag ci_image error with errno: {ret}')


class OldCPUImageBuilder(Builder):

    def init_args(self, args: Any) -> Any:
        if not args.torch_version:
            args.torch_version = '2.3.1'
            args.torchaudio_version = '2.3.1'
            args.torchvision_version = '0.18.1'
        if not args.tf_version:
            args.tf_version = '2.16.1'
        if not args.cuda_version:
            args.cuda_version = '12.1.0'
        if not args.vllm_version:
            args.vllm_version = '0.5.3'
        if not args.lmdeploy_version:
            args.lmdeploy_version = '0.6.2'
        if not args.autogptq_version:
            args.autogptq_version = '0.7.1'
        if not args.flashattn_version:
            args.flashattn_version = '2.7.1.post4'
        return args

    def generate_dockerfile(self) -> str:
        with open('docker/Dockerfile.ubuntu.old', 'r') as f:
            content = f.read()
        old_cpu_image = (
            'modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:'
            'ubuntu22.04-py311-torch2.3.1-1.34.0-test')
        content = content.replace('{base_image}', old_cpu_image)
        content = content.replace('{modelscope_branch}',
                                  self.args.modelscope_branch)
        content = content.replace('{cur_time}', formatted_time)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-{self.args.modelscope_version}-test'
        )

    def build(self):
        return self.run_cmd('docker', 'build',
                            '--build-arg', 'DOCKER_BUILDKIT=0', '-t',
                            self.image(), '-f', 'Dockerfile', '.')

    def push(self):
        ret = self.run_cmd('docker', 'push', self.image())
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-{self.args.modelscope_version}-{formatted_time}-test'
        )
        ret = self.run_cmd('docker', 'tag', self.image(), image_tag2)
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'push', image_tag2)


class OldGPUImageBuilder(Builder):

    def init_args(self, args: Any) -> Any:
        if not args.torch_version:
            args.torch_version = '2.3.1'
            args.torchaudio_version = '2.3.1'
            args.torchvision_version = '0.18.1'
        if not args.tf_version:
            args.tf_version = '2.16.1'
        if not args.cuda_version:
            args.cuda_version = '12.1.0'
        if not args.vllm_version:
            args.vllm_version = '0.5.3'
        if not args.lmdeploy_version:
            args.lmdeploy_version = '0.6.2'
        if not args.autogptq_version:
            args.autogptq_version = '0.7.1'
        if not args.flashattn_version:
            args.flashattn_version = '2.7.1.post4'
        return args

    def generate_dockerfile(self) -> str:
        old_gpu_image = (
            'modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:'
            'ubuntu22.04-cuda12.1.0-py311-torch2.3.1-tf2.16.1-1.34.0-test')
        with open('docker/Dockerfile.ubuntu.old', 'r') as f:
            content = f.read()
        content = content.replace('{base_image}', old_gpu_image)
        content = content.replace('{modelscope_branch}',
                                  self.args.modelscope_branch)
        content = content.replace('{cur_time}', formatted_time)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-base')

    def build(self):
        return self.run_cmd('docker', 'build',
                            '--build-arg', 'DOCKER_BUILDKIT=0', '-t',
                            self.image(), '-f', 'Dockerfile', '.')

    def push(self):
        ret = self.run_cmd('docker', 'push', self.image())
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-tf{self.args.tf_version}-'
            f'{self.args.modelscope_version}-{formatted_time}-test')
        ret = self.run_cmd('docker', 'tag', self.image(), image_tag2)
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'push', image_tag2)


class BaseCPUImageBuilder(Builder):

    def generate_dockerfile(self) -> str:
        with open('docker/Dockerfile.ubuntu_base', 'r') as f:
            content = f.read()
        content = content.replace('{base_image}', self.args.base_image)
        content = content.replace('{use_gpu}', 'False')
        content = content.replace('{python_version}', self.args.python_version)
        content = content.replace('{torch_version}', self.args.torch_version)
        content = content.replace('{cudatoolkit_version}',
                                  self.args.cudatoolkit_version)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-base')

    def build(self):
        return self.run_cmd('docker', 'build',
                            '--build-arg', 'DOCKER_BUILDKIT=0', '-t',
                            self.image(), '-f', 'Dockerfile', '.')

    def push(self):
        return self.run_cmd('docker', 'push', self.image())


class BaseGPUImageBuilder(Builder):

    def generate_dockerfile(self) -> str:
        with open('docker/Dockerfile.ubuntu_base', 'r') as f:
            content = f.read()
        content = content.replace('{base_image}', self.args.base_image)
        content = content.replace('{use_gpu}', 'True')
        content = content.replace('{python_version}', self.args.python_version)
        content = content.replace('{torch_version}', self.args.torch_version)
        content = content.replace('{cudatoolkit_version}',
                                  self.args.cudatoolkit_version)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-test')

    def build(self) -> int:
        return self.run_cmd('docker', 'build',
                            '--build-arg', 'DOCKER_BUILDKIT=0', '-t',
                            self.image(), '-f', 'Dockerfile', '.')

    def push(self):
        return self.run_cmd('docker', 'push', self.image())


class StableCPUImageBuilder(Builder):

    def generate_dockerfile(self) -> str:
        meta_file = './docker/install_cpu.sh'
        version_args = (
            f'{self.args.torch_version} {self.args.torchvision_version} '
            f'{self.args.torchaudio_version}')
        base_image = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}'
            f'-torch{self.args.torch_version}-base')
        extra_content = ''

        with open('docker/Dockerfile.ubuntu', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', base_image)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{meta_file}', meta_file)
            content = content.replace('{version_args}', version_args)
            content = content.replace('{cur_time}', formatted_time)
            content = content.replace('{install_ms_deps}', 'True')
            content = content.replace('{image_type}', 'cpu')
            content = content.replace('{torch_version}',
                                      self.args.torch_version)
            content = content.replace('{torchvision_version}',
                                      self.args.torchvision_version)
            content = content.replace('{torchaudio_version}',
                                      self.args.torchaudio_version)
            content = content.replace(
                '{index_url}',
                '--index-url https://download.pytorch.org/whl/cpu')
            content = content.replace('{modelscope_branch}',
                                      self.args.modelscope_branch)
            content = content.replace('{swift_branch}', self.args.swift_branch)
        return self._remove_pynini_related_dependency(content)

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-{self.args.modelscope_version}-test'
        )

    def build(self) -> int:
        return self.run_cmd('docker', 'build', '-t', self.image(), '-f',
                            'Dockerfile', '.')

    def push(self):
        ret = self.run_cmd('docker', 'push', self.image())
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-{self.args.modelscope_version}-{formatted_time}-test'
        )
        ret = self.run_cmd('docker', 'tag', self.image(), image_tag2)
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'push', image_tag2)


class StableGPUImageBuilder(Builder):
    """Dependencies will be stable versions"""

    def init_args(self, args: Any) -> Any:
        if not args.torch_version:
            args.torch_version = '2.10.0'
            args.torchaudio_version = '2.10.0'
            args.torchvision_version = '0.25.0'
        if not args.vllm_version:
            args.vllm_version = '0.19.1'
        return super().init_args(args)

    def generate_dockerfile(self) -> str:
        meta_file = './docker/install.sh'
        with open('docker/Dockerfile.extra_install', 'r') as f:
            extra_content = f.read()
            extra_content = extra_content.replace('{python_version}',
                                                  self.args.python_version)
            extra_content += """
RUN export PIP_EXTRA_INDEX_URL=https://pypi.org/simple && \
    pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
"""
        version_args = (
            f'{self.args.torch_version} {self.args.torchvision_version} {self.args.torchaudio_version} '
            f'{self.args.vllm_version} {self.args.lmdeploy_version} {self.args.autogptq_version} '
            f'{self.args.flashattn_version} {self.args.optimum_version}')
        with open('docker/Dockerfile.ubuntu', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{meta_file}', meta_file)
            content = content.replace('{version_args}', version_args)
            content = content.replace('{cur_time}', formatted_time)
            content = content.replace('{install_ms_deps}', 'True')
            content = content.replace('{image_type}', 'gpu')
            content = content.replace('{torch_version}',
                                      self.args.torch_version)
            content = content.replace('{torchvision_version}',
                                      self.args.torchvision_version)
            content = content.replace('{torchaudio_version}',
                                      self.args.torchaudio_version)
            content = content.replace('{index_url}', '')
            content = content.replace('{modelscope_branch}',
                                      self.args.modelscope_branch)
            content = content.replace('{swift_branch}', self.args.swift_branch)
        return self._remove_pynini_related_dependency(content)

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-{self.args.modelscope_version}-test'
        )

    def build(self) -> int:
        return self.run_cmd('docker', 'build', '-t', self.image(), '-f',
                            'Dockerfile', '.')

    def push(self):
        ret = self.run_cmd('docker', 'push', self.image())
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-'
            f'{self.args.modelscope_version}-{formatted_time}-test')
        ret = self.run_cmd('docker', 'tag', self.image(), image_tag2)
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'push', image_tag2)


class LatestGPUImageBuilder(StableGPUImageBuilder):
    """Dependencies will be latest versions"""

    def generate_dockerfile(self) -> str:
        meta_file = './docker/install.sh'
        with open('docker/Dockerfile.extra_install', 'r') as f:
            extra_content = f.read()
            extra_content = extra_content.replace('{python_version}',
                                                  self.args.python_version)
        extra_content += """
RUN export PIP_EXTRA_INDEX_URL=https://pypi.org/simple && \
    pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
"""
        version_args = (
            f'{self.args.torch_version} {self.args.torchvision_version} {self.args.torchaudio_version} '
            f'{self.args.vllm_version} {self.args.lmdeploy_version} {self.args.autogptq_version}  '
            f'{self.args.flashattn_version} {self.args.optimum_version}')
        with open('docker/Dockerfile.ubuntu', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{meta_file}', meta_file)
            content = content.replace('{version_args}', version_args)
            content = content.replace('{cur_time}', formatted_time)
            content = content.replace('{install_ms_deps}', 'False')
            content = content.replace('{image_type}', 'gpu')
            content = content.replace('{torch_version}',
                                      self.args.torch_version)
            content = content.replace('{torchvision_version}',
                                      self.args.torchvision_version)
            content = content.replace('{torchaudio_version}',
                                      self.args.torchaudio_version)
            content = content.replace('{index_url}', '')
            content = content.replace('{modelscope_branch}',
                                      self.args.modelscope_branch)
            content = content.replace('{swift_branch}', self.args.swift_branch)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-{self.args.modelscope_version}-latest-test'
        )

    def push(self):
        ret = self.run_cmd('docker', 'push', self.image())
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-'
            f'{self.args.modelscope_version}-latest-{formatted_time}-test')
        ret = self.run_cmd('docker', 'tag', self.image(), image_tag2)
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'push', image_tag2)


class AmdImageBuilder(Builder):
    """Build ModelScope image on top of vllm/vllm-openai-rocm."""

    @staticmethod
    def _is_specific_release_tag(tag: str) -> bool:
        tag = tag.strip()
        if not tag or tag.lower() in _FLOATING_ROCM_TAGS:
            return False
        if tag.endswith('-base'):
            return False
        if _NIGHTLY_HASH_PATTERN.fullmatch(tag):
            return False
        return bool(_VERSION_TAG_PATTERN.fullmatch(tag))

    @staticmethod
    def _image_digest(tag_info: dict) -> Optional[str]:
        digest = tag_info.get('digest')
        if digest:
            return digest
        for image in tag_info.get('images') or []:
            digest = image.get('digest')
            if digest:
                return digest
        return None

    @classmethod
    def _fetch_rocm_tags(cls, page_size: int = 100) -> List[dict]:
        tags: List[dict] = []
        url = (f'https://hub.docker.com/v2/repositories/{VLLM_ROCM_REPO}/tags'
               f'?page_size={page_size}&ordering=-last_updated')
        while url:
            req = urllib.request.Request(
                url, headers={'User-Agent': 'modelscope-docker-builder'})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    payload = json.load(resp)
            except (urllib.error.URLError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f'Failed to query Docker Hub tags for {VLLM_ROCM_REPO}: '
                    f'{exc}') from exc
            tags.extend(payload.get('results') or [])
            url = payload.get('next')
            # Only scan the first few pages; release tags are near the top.
            if len(tags) >= 300:
                break
        if not tags:
            raise RuntimeError(
                f'No tags returned from Docker Hub for {VLLM_ROCM_REPO}')
        return tags

    @classmethod
    def resolve_latest_rocm_tag(cls) -> str:
        """Resolve the newest concrete release tag for vllm-openai-rocm.

        Preference order:
        1. Semver tag (vX.Y.Z) that shares digest with floating ``latest``
        2. Newest semver tag by Docker Hub ``last_updated``
        """
        tags = cls._fetch_rocm_tags()
        by_name = {item['name']: item for item in tags if item.get('name')}
        release_tags = [
            item for item in tags
            if cls._is_specific_release_tag(item.get('name', ''))
        ]
        latest_info = by_name.get('latest')
        latest_digest = cls._image_digest(latest_info) if latest_info else None
        if latest_digest:
            matched = [
                item for item in release_tags
                if cls._image_digest(item) == latest_digest
            ]
            if matched:
                # Prefer the first match in last_updated order from API.
                chosen = matched[0]['name']
                print(
                    f'Resolved {VLLM_ROCM_REPO} latest digest to release tag: '
                    f'{chosen}')
                return chosen

        if not release_tags:
            raise RuntimeError(
                f'No concrete release tags found for {VLLM_ROCM_REPO}')
        chosen = release_tags[0]['name']
        print(f'Resolved newest {VLLM_ROCM_REPO} release tag: {chosen}')
        return chosen

    def init_args(self, args: Any) -> Any:
        # Auto-discover from Docker Hub unless an explicit override is given.
        override = getattr(args, 'base_image_tag', None)
        if override and str(override).strip() and str(
                override).strip().lower() not in {'auto', 'latest'}:
            args.base_image_tag = str(override).strip()
            if not self._is_specific_release_tag(args.base_image_tag):
                raise ValueError(
                    'base_image_tag override must be a concrete release tag '
                    f'(e.g. v0.25.1), got: {args.base_image_tag}')
            print(f'Using override AMD ROCm base image tag: '
                  f'{args.base_image_tag}')
        else:
            args.base_image_tag = self.resolve_latest_rocm_tag()
        if not args.base_image:
            args.base_image = f'{VLLM_ROCM_REPO}:{args.base_image_tag}'
        if not args.cuda_version:
            args.cuda_version = '0.0.0'
        return args

    @staticmethod
    def _sanitize_tag(tag: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', '-', tag)

    @staticmethod
    def _normalize_version(version: str) -> str:
        version = version.strip().lstrip('vV')
        version = version.split('+')[0].split(' ')[0]
        return re.sub(r'[^0-9A-Za-z._-]+', '', version)

    @staticmethod
    def _python_tag_from_version(version: str) -> str:
        parts = version.strip().split('.')
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f'py{parts[0]}{parts[1]}'
        return f'py{re.sub(r"[^0-9]", "", version)}'

    @classmethod
    def _run_capture(cls, *cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(cmd), capture_output=True, text=True, check=False)

    @classmethod
    def _probe_via_entrypoint(cls, base_image: str) -> dict:
        """Read versions with docker run --entrypoint (no GPU required)."""
        script = (
            'import json,os,pathlib,sys\n'
            'info={"python":"%d.%d.%d"%sys.version_info[:3]}\n'
            'try:\n'
            ' import torch\n'
            ' info["torch"]=torch.__version__\n'
            ' hip=getattr(torch.version,"hip",None)\n'
            ' if hip: info["torch_hip"]=hip\n'
            'except Exception as e:\n'
            ' info["torch_error"]=str(e)\n'
            'for p in ("/opt/rocm/.info/version","/opt/rocm/.info/version-dev"):\n'
            ' f=pathlib.Path(p)\n'
            ' if f.is_file():\n'
            '  info["rocm_file"]=f.read_text().strip().splitlines()[0]\n'
            '  break\n'
            'for k in ("ROCM_VERSION","HIP_VERSION","TORCH_VERSION"):\n'
            ' if os.environ.get(k): info[k.lower()]=os.environ[k]\n'
            'print(json.dumps(info))\n')
        for py in ('python3', 'python'):
            result = cls._run_capture('docker', 'run', '--rm', '--network',
                                      'none', '--entrypoint', py, base_image,
                                      '-c', script)
            if result.returncode == 0 and result.stdout.strip():
                try:
                    return json.loads(result.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    continue
        return {}

    @classmethod
    def _probe_via_history(cls, base_image: str) -> dict:
        """Parse build ARGs from docker history (no container start)."""
        result = cls._run_capture('docker', 'history', '--no-trunc',
                                  '--format', '{{.CreatedBy}}', base_image)
        if result.returncode != 0:
            return {}
        text = result.stdout
        info = {}
        for key, pattern in (
            ('rocm', r'ROCM_VERSION=([0-9]+(?:\.[0-9]+)*)'),
            ('python', r'PYTHON_VERSION=([0-9]+(?:\.[0-9]+)*)'),
            ('ubuntu',
             r'org\.opencontainers\.image\.version=([0-9]+(?:\.[0-9]+)*)'),
        ):
            matches = re.findall(pattern, text)
            if matches:
                # docker history lists newest layers first.
                info[key] = matches[0]
        return info

    @classmethod
    def _probe_via_create_cp(cls, base_image: str) -> dict:
        """Copy version files out of a created (not started) container."""
        import tempfile
        create = cls._run_capture('docker', 'create', base_image)
        if create.returncode != 0:
            return {}
        cid = create.stdout.strip()
        info = {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest = os.path.join(tmp, 'version')
                for src in ('/opt/rocm/.info/version',
                            '/opt/rocm/.info/version-dev'):
                    result = cls._run_capture('docker', 'cp', f'{cid}:{src}',
                                              dest)
                    if result.returncode == 0 and os.path.isfile(dest):
                        with open(dest, 'r', encoding='utf-8') as f:
                            line = f.read().strip().splitlines()
                        if line:
                            info['rocm_file'] = line[0].strip()
                            break
        finally:
            cls._run_capture('docker', 'rm', '-f', cid)
        return info

    @classmethod
    def probe_base_image_versions(cls, base_image: str) -> dict:
        """Discover rocm/python/torch without needing AMD GPU.

        Methods (in order):
        1. docker run --entrypoint python -c ...  (CPU-only, no --device)
        2. docker history --no-trunc parse ROCM_VERSION/PYTHON_VERSION
        3. docker create + docker cp /opt/rocm/.info/version
        """
        probed = {}
        entry = cls._probe_via_entrypoint(base_image)
        history = cls._probe_via_history(base_image)
        copied = cls._probe_via_create_cp(base_image)
        probed.update(history)
        probed.update(copied)
        probed.update(entry)

        rocm = (
            probed.get('rocm_file') or probed.get('rocm_version')
            or probed.get('rocm') or probed.get('torch_hip')
            or probed.get('hip_version'))
        python_ver = probed.get('python')
        torch_ver = probed.get('torch') or probed.get('torch_version')
        ubuntu_ver = probed.get('ubuntu')

        versions = {
            'rocm': cls._normalize_version(rocm) if rocm else None,
            'python':
            cls._normalize_version(python_ver) if python_ver else None,
            'torch': cls._normalize_version(torch_ver) if torch_ver else None,
            'ubuntu':
            cls._normalize_version(ubuntu_ver) if ubuntu_ver else None,
        }
        print('Probed AMD base image versions:')
        for key, value in versions.items():
            print(f'  {key}: {value or "unknown"}')
        return versions

    def generate_dockerfile(self) -> str:
        with open('docker/Dockerfile.amd', 'r') as f:
            content = f.read()
        content = content.replace('{base_image}', self.args.base_image)
        content = content.replace('{base_image_tag}', self.args.base_image_tag)
        content = content.replace('{modelscope_branch}',
                                  self.args.modelscope_branch)
        content = content.replace('{cur_time}', formatted_time)
        return content

    def image(self) -> str:
        ubuntu = getattr(self.args, 'amd_ubuntu_version',
                         None) or self.args.ubuntu_version
        rocm = getattr(self.args, 'amd_rocm_version', None)
        py_tag = getattr(self.args, 'amd_python_tag', None) or getattr(
            self.args, 'python_tag', None)
        torch = getattr(self.args, 'amd_torch_version', None)
        if not (rocm and py_tag and torch):
            raise RuntimeError(
                'AMD image tag requires probed rocm/python/torch versions. '
                f'Got rocm={rocm}, python={py_tag}, torch={torch}')
        return (f'{docker_registry}:ubuntu{ubuntu}-rocm{rocm}-{py_tag}-'
                f'torch{torch}-{self.args.modelscope_version}-test')

    def _log_base_image_info(self) -> int:
        base_image = self.args.base_image
        print('=' * 60)
        print(f'AMD ROCm base image: {base_image}')
        print(f'AMD ROCm base image tag: {self.args.base_image_tag}')
        print('=' * 60)
        ret = self.run_cmd('docker', 'pull', base_image)
        if ret != 0:
            return ret
        result = self._run_capture(
            'docker', 'image', 'inspect', base_image,
            '--format={{.Id}} {{if index .RepoDigests 0}}'
            '{{index .RepoDigests 0}}{{else}}local-only{{end}}')
        if result.returncode == 0:
            print(f'AMD base image resolved: {result.stdout.strip()}')
        else:
            print(f'AMD base image inspect warning: {result.stderr.strip()}')

        versions = self.probe_base_image_versions(base_image)
        if not versions.get('rocm') or not versions.get(
                'python') or not versions.get('torch'):
            print('ERROR: failed to probe rocm/python/torch from base image')
            return 1
        self.args.amd_rocm_version = versions['rocm']
        self.args.amd_torch_version = versions['torch']
        self.args.amd_python_tag = self._python_tag_from_version(
            versions['python'])
        if versions.get('ubuntu'):
            self.args.amd_ubuntu_version = versions['ubuntu']
        else:
            self.args.amd_ubuntu_version = self.args.ubuntu_version
        print(f'AMD output image tag will be: {self.image()}')
        print('=' * 60)
        return 0

    def build(self) -> int:
        ret = self._log_base_image_info()
        if ret != 0:
            return ret
        return self.run_cmd('docker', 'build', '-t', self.image(), '-f',
                            'Dockerfile', '.')

    def push(self):
        image_name = self.image()
        ret = self.run_cmd('docker', 'push', image_name)
        if ret != 0:
            return ret
        ubuntu = self.args.amd_ubuntu_version
        rocm = self.args.amd_rocm_version
        py_tag = self.args.amd_python_tag
        torch = self.args.amd_torch_version
        image_tag2 = (f'{docker_registry}:ubuntu{ubuntu}-rocm{rocm}-{py_tag}-'
                      f'torch{torch}-{self.args.modelscope_version}-'
                      f'{formatted_time}-test')
        ret = self.run_cmd('docker', 'tag', image_name, image_tag2)
        if ret != 0:
            return ret
        print(f'AMD image timestamp tag: {image_tag2}')
        return self.run_cmd('docker', 'push', image_tag2)


class AscendImageBuilder(StableGPUImageBuilder):

    _CANN_VERSION_PATTERN = re.compile(r'^\d+(?:\.[0-9A-Za-z]+)+$')
    _OS_TAG_PATTERN = re.compile(r'^[A-Za-z]+[0-9][0-9A-Za-z.]*$')

    @staticmethod
    def _normalize_arch(arch: str = None) -> str:
        arch = arch or platform.machine()
        arch = arch.lower()
        arch_mapping = {
            'x86': 'x86',
            'x86_64': 'x86',
            'amd64': 'x86',
            'arm': 'arm',
            'aarch64': 'arm',
            'arm64': 'arm',
        }
        if arch not in arch_mapping:
            raise ValueError(f'Unsupported architecture: {arch}. '
                             'Please pass --arch x86 or --arch arm.')
        return arch_mapping[arch]

    @staticmethod
    def _get_atlas_hardware(soc_version: str) -> str:
        soc_version = soc_version.lower()
        atlas_mapping = {
            'ascend910b1': 'A2',
            'ascend910_9391': 'A3',
            'ascend310p1': '300I',
        }
        if soc_version.startswith('ascend950'):
            return 'A5'
        if soc_version not in atlas_mapping:
            raise ValueError(
                f'Unsupported soc_version: {soc_version}. '
                'Supported values are ascend910b1, ascend910_9391, '
                'ascend310p1, and values starting with ascend950.')
        return atlas_mapping[soc_version]

    @classmethod
    def _get_cann_os_tags(cls, base_image: str) -> tuple:
        if ':' not in base_image.rsplit('/', 1)[-1]:
            raise ValueError(
                f'Ascend base image must include a tag: {base_image}')

        base_tag = base_image.rsplit(':', 1)[1]
        parts = base_tag.split('-')
        if len(parts) < 4:
            raise ValueError(
                'Ascend base image tag must look like '
                f'<cann_version>-<hardware>-<os_tag>-py<version>, got: '
                f'{base_tag}')

        cann_version = parts[0]
        os_tag = parts[2]
        if not cls._CANN_VERSION_PATTERN.fullmatch(cann_version):
            raise ValueError(f'Invalid CANN version in Ascend base image tag: '
                             f'{cann_version}')
        if not cls._OS_TAG_PATTERN.fullmatch(os_tag):
            raise ValueError(
                f'Invalid OS tag in Ascend base image tag: {os_tag}')

        return cann_version, f'CANN{cann_version}', os_tag

    def init_args(self, args) -> Any:
        if not args.base_image:
            # Reuse the prebuilt vllm-ascend image to avoid rebuilding its stack.
            args.base_image = 'quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11'
        args.arch = self._normalize_arch(args.arch)
        args.atlas_hardware = self._get_atlas_hardware(args.soc_version)
        args.cann_version, args.cann_version_tag, args.os_tag = (
            self._get_cann_os_tags(args.base_image))
        return super().init_args(args)

    def generate_dockerfile(self) -> str:
        extra_content = """
RUN export PIP_EXTRA_INDEX_URL=https://pypi.org/simple && \
    pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
"""
        with open('docker/Dockerfile.ascend', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{soc_version}', self.args.soc_version)
            content = content.replace('{cann_version}', self.args.cann_version)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{cur_time}', formatted_time)
            content = content.replace('{install_ms_deps}', 'False')
            content = content.replace('{modelscope_branch}',
                                      self.args.modelscope_branch)
            content = content.replace('{swift_branch}', self.args.swift_branch)
            content = content.replace('{megatron_branch}',
                                      self.args.megatron_branch)
            content = content.replace('{mindspeed_branch}',
                                      self.args.mindspeed_branch)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:{self.args.swift_branch}-'
            f'{self.args.atlas_hardware}-{self.args.python_tag}-'
            f'{self.args.cann_version_tag}-{self.args.os_tag}-{self.args.arch}'
        )

    def push(self):
        return 0


parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str, default=None)
parser.add_argument('--image_type', type=str)
parser.add_argument('--python_version', type=str, default='3.12.13')
parser.add_argument('--ubuntu_version', type=str, default='22.04')
parser.add_argument('--torch_version', type=str, default=None)
parser.add_argument('--torchvision_version', type=str, default=None)
parser.add_argument('--cuda_version', type=str, default=None)
parser.add_argument('--ci_image', type=int, default=0)
parser.add_argument('--torchaudio_version', type=str, default=None)
parser.add_argument('--tf_version', type=str, default=None)
parser.add_argument('--vllm_version', type=str, default=None)
parser.add_argument('--lmdeploy_version', type=str, default=None)
parser.add_argument('--flashattn_version', type=str, default=None)
parser.add_argument('--autogptq_version', type=str, default=None)
parser.add_argument('--optimum_version', type=str, default=None)
parser.add_argument('--modelscope_branch', type=str, default='master')
parser.add_argument('--modelscope_version', type=str, default='9.99.0')
parser.add_argument('--swift_branch', type=str, default='main')
parser.add_argument('--megatron_branch', type=str, default='v0.15.3')
parser.add_argument('--mindspeed_branch', type=str, default='core_r0.15.3')
parser.add_argument('--soc_version', type=str, default='ascend910_9391')
parser.add_argument('--arch', type=str, choices=['x86', 'arm'], default=None)
parser.add_argument(
    '--base_image_tag',
    type=str,
    default=None,
    help='Optional AMD ROCm override tag. Default: auto-resolve newest '
    'concrete vllm/vllm-openai-rocm release tag from Docker Hub.')
parser.add_argument('--dry_run', type=int, default=0)
args = parser.parse_args()

if args.image_type.lower() == 'base':
    builder_cls = [BaseCPUImageBuilder, BaseGPUImageBuilder]
elif args.image_type.lower() == 'old':
    builder_cls = [OldCPUImageBuilder, OldGPUImageBuilder]
elif args.image_type.lower() == 'stable':
    builder_cls = [StableCPUImageBuilder, StableGPUImageBuilder]
elif args.image_type.lower() == 'ascend':
    builder_cls = [AscendImageBuilder]
elif args.image_type.lower() == 'amd':
    builder_cls = [AmdImageBuilder]
elif args.image_type.lower() == 'latest':
    builder_cls = [LatestGPUImageBuilder]
else:
    raise ValueError(f'Unsupported image_type: {args.image_type}')

for builder in builder_cls:
    args = copy(args)
    builder(args, args.dry_run)()
