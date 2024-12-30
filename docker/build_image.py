import argparse
import os
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
            args.base_image = 'nvidia/cuda:12.1.0-devel-ubuntu22.04'
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
                ret = os.system(
                    f'docker tag {self.image()} {docker_registry}:ci_image')
                if ret != 0:
                    raise RuntimeError(
                        f'Docker tag ci_image error with errno: {ret}')


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
        content = content.replace('{tf_version}', self.args.tf_version)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-base')

    def build(self):
        return os.system(
            f'DOCKER_BUILDKIT=0 docker build -t {self.image()} -f Dockerfile .'
        )

    def push(self):
        return os.system(f'docker push {self.image()}')


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
        content = content.replace('{tf_version}', self.args.tf_version)
        return content

    def image(self) -> str:
        return (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-tf{self.args.tf_version}-base')

    def build(self) -> int:
        return os.system(
            f'DOCKER_BUILDKIT=0 docker build -t {self.image()} -f Dockerfile .'
        )

    def push(self):
        return os.system(f'docker push {self.image()}')


class CPUImageBuilder(Builder):

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
            content = content.replace('{install_ms_deps}', 'True')
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
        return os.system(f'docker build -t {self.image()} -f Dockerfile .')

    def push(self):
        ret = os.system(f'docker push {self.image()}')
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-{self.args.modelscope_version}-{formatted_time}-test'
        )
        ret = os.system(f'docker tag {self.image()} {image_tag2}')
        if ret != 0:
            return ret
        return os.system(f'docker push {image_tag2}')


class GPUImageBuilder(Builder):

    def generate_dockerfile(self) -> str:
        meta_file = './docker/install.sh'
        extra_content = """
RUN pip install tf-keras==2.16.0 --no-dependencies && \
    pip install --no-cache-dir torchsde jupyterlab torchmetrics==0.11.4 basicsr pynvml shortuuid && \
    CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0" \
        pip install --no-cache-dir  'git+https://github.com/facebookresearch/detectron2.git'
"""

        version_args = (
            f'{self.args.torch_version} {self.args.torchvision_version} {self.args.torchaudio_version} '
            f'{self.args.vllm_version} {self.args.lmdeploy_version} {self.args.autogptq_version}'
        )
        base_image = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-{self.args.python_tag}-'
            f'torch{self.args.torch_version}-tf{self.args.tf_version}-base')
        with open('docker/Dockerfile.ubuntu', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', base_image)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{meta_file}', meta_file)
            content = content.replace('{version_args}', version_args)
            content = content.replace('{install_ms_deps}', 'True')
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
            f'{self.args.python_tag}-torch{self.args.torch_version}-tf{self.args.tf_version}-'
            f'{self.args.modelscope_version}-test')

    def build(self) -> int:
        return os.system(f'docker build -t {self.image()} -f Dockerfile .')

    def push(self):
        ret = os.system(f'docker push {self.image()}')
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-tf{self.args.tf_version}-'
            f'{self.args.modelscope_version}-{formatted_time}-test')
        ret = os.system(f'docker tag {self.image()} {image_tag2}')
        if ret != 0:
            return ret
        return os.system(f'docker push {image_tag2}')


class LLMImageBuilder(Builder):

    def init_args(self, args) -> Any:
        if not args.base_image:
            # A mirrored image of nvidia/cuda:12.4.0-devel-ubuntu22.04
            args.base_image = 'nvidia/cuda:12.4.0-devel-ubuntu22.04'
        if not args.torch_version:
            args.torch_version = '2.4.0'
            args.torchaudio_version = '2.4.0'
            args.torchvision_version = '0.19.0'
        if not args.cuda_version:
            args.cuda_version = '12.4.0'
        if not args.vllm_version:
            args.vllm_version = '0.6.3.post1'
        if not args.lmdeploy_version:
            args.lmdeploy_version = '0.6.2'
        if not args.autogptq_version:
            args.autogptq_version = '0.7.1'
        return args

    def generate_dockerfile(self) -> str:
        meta_file = './docker/install.sh'
        with open('docker/Dockerfile.extra_install', 'r') as f:
            extra_content = f.read()
            extra_content = extra_content.replace('{python_version}',
                                                  self.args.python_version)
        version_args = (
            f'{self.args.torch_version} {self.args.torchvision_version} {self.args.torchaudio_version} '
            f'{self.args.vllm_version} {self.args.lmdeploy_version} {self.args.autogptq_version}'
        )
        with open('docker/Dockerfile.ubuntu', 'r') as f:
            content = f.read()
            content = content.replace('{base_image}', self.args.base_image)
            content = content.replace('{extra_content}', extra_content)
            content = content.replace('{meta_file}', meta_file)
            content = content.replace('{version_args}', version_args)
            content = content.replace('{install_ms_deps}', 'False')
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
            f'{self.args.python_tag}-torch{self.args.torch_version}-{self.args.modelscope_version}-LLM-test'
        )

    def build(self) -> int:
        return os.system(f'docker build -t {self.image()} -f Dockerfile .')

    def push(self):
        ret = os.system(f'docker push {self.image()}')
        if ret != 0:
            return ret
        image_tag2 = (
            f'{docker_registry}:ubuntu{self.args.ubuntu_version}-cuda{self.args.cuda_version}-'
            f'{self.args.python_tag}-torch{self.args.torch_version}-'
            f'{self.args.modelscope_version}-LLM-{formatted_time}-test')
        ret = os.system(f'docker tag {self.image()} {image_tag2}')
        if ret != 0:
            return ret
        return os.system(f'docker push {image_tag2}')


parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str, default=None)
parser.add_argument('--image_type', type=str)
parser.add_argument('--python_version', type=str, default='3.10.14')
parser.add_argument('--ubuntu_version', type=str, default='22.04')
parser.add_argument('--torch_version', type=str, default=None)
parser.add_argument('--torchvision_version', type=str, default=None)
parser.add_argument('--cuda_version', type=str, default=None)
parser.add_argument('--ci_image', type=int, default=0)
parser.add_argument('--torchaudio_version', type=str, default=None)
parser.add_argument('--tf_version', type=str, default=None)
parser.add_argument('--vllm_version', type=str, default=None)
parser.add_argument('--lmdeploy_version', type=str, default=None)
parser.add_argument('--autogptq_version', type=str, default=None)
parser.add_argument('--modelscope_branch', type=str, default='master')
parser.add_argument('--modelscope_version', type=str, default='9.99.0')
parser.add_argument('--swift_branch', type=str, default='main')
parser.add_argument('--dry_run', type=int, default=0)

args = parser.parse_args()

if args.image_type.lower() == 'base_cpu':
    builder_cls = BaseCPUImageBuilder
elif args.image_type.lower() == 'base_gpu':
    builder_cls = BaseGPUImageBuilder
elif args.image_type.lower() == 'cpu':
    builder_cls = CPUImageBuilder
elif args.image_type.lower() == 'gpu':
    builder_cls = GPUImageBuilder
elif args.image_type.lower() == 'llm':
    builder_cls = LLMImageBuilder
else:
    raise ValueError(f'Unsupported image_type: {args.image_type}')

builder_cls(args, args.dry_run)()
