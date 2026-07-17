import argparse
import os
import platform
import re
import subprocess
from copy import copy
from datetime import datetime
from typing import Any

docker_registry = os.environ['DOCKER_REGISTRY']
assert docker_registry, 'You must pass a valid DOCKER_REGISTRY'
timestamp = datetime.now()
formatted_time = timestamp.strftime('%Y%m%d%H%M%S')


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
RUN pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
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
RUN pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
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


class AscendImageBuilder(StableGPUImageBuilder):

    _DEFAULT_TORCH_VERSION = '2.9.0'
    _DEFAULT_TORCHVISION_VERSION = '0.24.0'
    _DEFAULT_TORCHAUDIO_VERSION = '2.9.0'
    _DEFAULT_TORCH_NPU_VERSION = '2.9.0.post2'
    _DEFAULT_VLLM_VERSION = '0.18.0'
    _DEFAULT_VLLM_ASCEND_VERSION = '0.18.0'
    _DEFAULT_TRITON_ASCEND_VERSIONS = {
        '8.5': '3.2.0',
        '9.0': '3.2.1',
    }
    _CANN_VERSION_PATTERN = re.compile(r'^\d+(?:\.[0-9A-Za-z]+)+$')
    _OS_TAG_PATTERN = re.compile(r'^[A-Za-z]+[0-9][0-9A-Za-z.]*$')
    _PYTHON_TAG_PATTERN = re.compile(r'^py\d+\.\d+$', re.IGNORECASE)
    _TORCH_NPU_VERSION_PATTERN = re.compile(
        r'^(?P<torch_version>\d+\.\d+\.\d+)(?:\.post\d+)?$')

    @staticmethod
    def _normalize_arch(arch: str = None) -> str:
        arch = arch or platform.machine()
        arch = arch.lower()
        arch_mapping = {
            'x86': 'x86_64',
            'x86_64': 'x86_64',
            'amd64': 'x86_64',
            'arm': 'aarch64',
            'aarch64': 'aarch64',
            'arm64': 'aarch64',
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
        python_tag = parts[3]
        if not cls._CANN_VERSION_PATTERN.fullmatch(cann_version):
            raise ValueError(f'Invalid CANN version in Ascend base image tag: '
                             f'{cann_version}')
        if not cls._OS_TAG_PATTERN.fullmatch(os_tag):
            raise ValueError(
                f'Invalid OS tag in Ascend base image tag: {os_tag}')
        if not cls._PYTHON_TAG_PATTERN.fullmatch(python_tag):
            raise ValueError(
                f'Invalid Python tag in Ascend base image tag: {python_tag}')

        return cann_version, f'CANN{cann_version}', os_tag, python_tag

    @staticmethod
    def _get_os_family(os_tag: str) -> str:
        os_tag = os_tag.lower()
        if os_tag.startswith('ubuntu'):
            return 'ubuntu'
        if os_tag.startswith('openeuler'):
            return 'openeuler'
        raise ValueError(f'Unsupported Ascend base image OS tag: {os_tag}. '
                         'Supported OS families are Ubuntu and openEuler.')

    @classmethod
    def _init_torch_versions(cls, args) -> None:
        torch_version_specified = args.torch_version is not None
        torchvision_version_specified = args.torchvision_version is not None
        torchaudio_version_specified = args.torchaudio_version is not None

        if torch_version_specified and (not torchvision_version_specified
                                        or not torchaudio_version_specified):
            raise ValueError(
                'When overriding --torch_version for an Ascend image, also '
                'pass matching --torchvision_version and '
                '--torchaudio_version.')
        if (torchvision_version_specified or
                torchaudio_version_specified) and not torch_version_specified:
            raise ValueError(
                '--torchvision_version and --torchaudio_version require an '
                'explicit --torch_version for an Ascend image.')

        args.torch_version = args.torch_version or cls._DEFAULT_TORCH_VERSION
        args.torchvision_version = (
            args.torchvision_version or cls._DEFAULT_TORCHVISION_VERSION)
        args.torchaudio_version = (
            args.torchaudio_version or cls._DEFAULT_TORCHAUDIO_VERSION)
        args.torch_npu_version = (
            args.torch_npu_version or cls._DEFAULT_TORCH_NPU_VERSION)

        match = cls._TORCH_NPU_VERSION_PATTERN.fullmatch(
            args.torch_npu_version)
        if not match:
            raise ValueError('Invalid --torch_npu_version. Expected '
                             '<major>.<minor>.<patch> or '
                             '<major>.<minor>.<patch>.post<revision>.')
        if args.torch_version != match.group('torch_version'):
            raise ValueError(
                '--torch_version must exactly match the base version of '
                f'--torch_npu_version, got torch={args.torch_version} and '
                f'torch_npu={args.torch_npu_version}.')

    @classmethod
    def _init_component_versions(cls, args) -> None:
        args.vllm_version = args.vllm_version or cls._DEFAULT_VLLM_VERSION
        args.vllm_ascend_version = (
            args.vllm_ascend_version or cls._DEFAULT_VLLM_ASCEND_VERSION)
        args.vllm_git_ref = cls._get_vllm_git_ref(args.vllm_version)
        args.vllm_ascend_git_ref = cls._get_vllm_git_ref(
            args.vllm_ascend_version)

        if not args.triton_ascend_version:
            cann_series = '.'.join(args.cann_version.split('.')[:2])
            try:
                args.triton_ascend_version = (
                    cls._DEFAULT_TRITON_ASCEND_VERSIONS[cann_series])
            except KeyError as e:
                raise ValueError('No default triton-ascend version for CANN '
                                 f'{args.cann_version}. Please pass '
                                 '--triton_ascend_version explicitly.') from e

    @staticmethod
    def _get_vllm_git_ref(version: str) -> str:
        return version if version.startswith('v') else f'v{version}'

    def init_args(self, args) -> Any:
        if not args.base_image:
            # Reuse the prebuilt vllm-ascend image to avoid rebuilding its stack.
            args.base_image = 'quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11'
        self._init_torch_versions(args)
        args.arch = self._normalize_arch(args.arch)
        args.atlas_hardware = self._get_atlas_hardware(args.soc_version)
        (args.cann_version, args.cann_version_tag, args.os_tag,
         args.ascend_python_tag) = (
             self._get_cann_os_tags(args.base_image))
        self._get_os_family(args.os_tag)
        self._init_component_versions(args)
        return super().init_args(args)

    def _generate_python_tag(self, _python_version: str) -> str:
        return self.args.ascend_python_tag

    def generate_dockerfile(self) -> str:
        extra_content = """
RUN pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
"""
        with open('docker/Dockerfile.ascend', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{soc_version}', self.args.soc_version)
            content = content.replace('{cann_version}', self.args.cann_version)
            content = content.replace('{torch_version}',
                                      self.args.torch_version)
            content = content.replace('{torchvision_version}',
                                      self.args.torchvision_version)
            content = content.replace('{torchaudio_version}',
                                      self.args.torchaudio_version)
            content = content.replace('{torch_npu_version}',
                                      self.args.torch_npu_version)
            content = content.replace('{vllm_git_ref}', self.args.vllm_git_ref)
            content = content.replace('{vllm_ascend_git_ref}',
                                      self.args.vllm_ascend_git_ref)
            content = content.replace('{triton_ascend_version}',
                                      self.args.triton_ascend_version)
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
        tag = (f'{self.args.swift_branch}-{self.args.cann_version_tag}-'
               f'torch_npu{self.args.torch_npu_version}-'
               f'{self.args.atlas_hardware}-{self.args.os_tag}-'
               f'{self.args.python_tag}-'
               f'{self.args.arch}')
        return f'{docker_registry}:{tag.lower()}'

    def push(self):
        return 0


parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str, default=None)
parser.add_argument('--image_type', type=str)
parser.add_argument('--python_version', type=str, default='3.12.13')
parser.add_argument('--ubuntu_version', type=str, default='22.04')
parser.add_argument('--torch_version', type=str, default=None)
parser.add_argument('--torch_npu_version', type=str, default=None)
parser.add_argument('--torchvision_version', type=str, default=None)
parser.add_argument('--cuda_version', type=str, default=None)
parser.add_argument('--ci_image', type=int, default=0)
parser.add_argument('--torchaudio_version', type=str, default=None)
parser.add_argument('--tf_version', type=str, default=None)
parser.add_argument('--vllm_version', type=str, default=None)
parser.add_argument('--vllm_ascend_version', type=str, default=None)
parser.add_argument('--triton_ascend_version', type=str, default=None)
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
elif args.image_type.lower() == 'latest':
    builder_cls = [LatestGPUImageBuilder]
else:
    raise ValueError(f'Unsupported image_type: {args.image_type}')

for builder in builder_cls:
    args = copy(args)
    builder(args, args.dry_run)()
