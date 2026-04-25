import argparse
import os
import platform
import subprocess
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
            args.torch_version = '2.9.1'
            args.torchaudio_version = '2.9.1'
            args.torchvision_version = '0.24.1'
        if not args.optimum_version:
            args.optimum_version = '2.0.0'
        if not args.tf_version:
            args.tf_version = '2.16.1'
        if not args.cuda_version:
            args.cuda_version = '12.8.1'
        if not args.vllm_version:
            args.vllm_version = '0.15.1'
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
        return content

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
        return content

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

    def init_args(self, args: Any) -> Any:
        if not args.vllm_version:
            args.vllm_version = '0.16.0'
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
            f'{self.args.vllm_version} {self.args.lmdeploy_version} {self.args.autogptq_version}  '
            f'{self.args.optimum_version}'
            f'{self.args.flashattn_version}')
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
            raise ValueError(
                f'Unsupported architecture: {arch}. '
                'Please pass --arch x86 or --arch arm.'
            )
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

    def init_args(self, args) -> Any:
        if not args.base_image:
            # Reuse the prebuilt vllm-ascend image to avoid rebuilding its stack.
            args.base_image = 'quay.io/ascend/cann:8.5.1-a3-ubuntu22.04-py3.11'
        args.arch = self._normalize_arch(args.arch)
        args.atlas_hardware = self._get_atlas_hardware(args.soc_version)
        return super().init_args(args)

    def generate_dockerfile(self) -> str:
        extra_content = """
RUN pip install --no-cache-dir -U icecream soundfile pybind11 py-spy
"""
        with open('docker/Dockerfile.ascend', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{soc_version}', self.args.soc_version)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{cur_time}', formatted_time)
            content = content.replace('{install_ms_deps}', 'False')
            content = content.replace('{modelscope_branch}',
                                      self.args.modelscope_branch)
            content = content.replace('{swift_branch}', self.args.swift_branch)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:{self.args.swift_branch}-'
            f'{self.args.atlas_hardware}-{self.args.python_tag}-{self.args.arch}'
        )

    def push(self):
        return 0


parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str, default=None)
parser.add_argument('--image_type', type=str)
parser.add_argument('--python_version', type=str, default='3.11.11')
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
    builder(args, args.dry_run)()
