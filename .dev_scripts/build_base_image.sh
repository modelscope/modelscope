#!/bin/bash
# default values.
BASE_CPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04
BASE_GPU_CUDA113_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04-cuda11.3.0-cudnn8-devel
BASE_GPU_CUDA117_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04-cuda11.7.1-cudnn8-devel
BASE_GPU_CUDA118_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04-cuda11.8.0-cudnn8-devel
MODELSCOPE_REPO_ADDRESS=reg.docker.alibaba-inc.com/modelscope/modelscope
python_version=3.7.13
torch_version=1.11.0
cuda_version=11.7.1
cudatoolkit_version=11.3
tensorflow_version=1.15.5
version=None
is_cpu=False
function usage(){
    echo "usage: build.sh "
    echo "       --python=python_version set python version, default: $python_version"
    echo "       --cuda=cuda_version set cuda version,only[11.3.0, 11.7.1], fefault: $cuda_version"
    echo "       --torch=torch_version set pytorch version, fefault: $torch_version"
    echo "       --tensorflow=tensorflow_version set tensorflow version, default: $tensorflow_version"
    echo "       --test option for run test before push image, only push on ci test pass"
    echo "       --cpu option for build cpu version"
    echo "       --push option for push image to remote repo"
}
for i in "$@"; do
  case $i in
    --python=*)
      python_version="${i#*=}"
      shift
      ;;
    --cuda=*)
      cuda_version="${i#*=}"
      shift # pytorch version
      ;;
    --torch=*)
      torch_version="${i#*=}"
      shift # pytorch version
      ;;
    --tensorflow=*)
      tensorflow_version="${i#*=}"
      shift # tensorflow version
      ;;
    --version=*)
      version="${i#*=}"
      shift # version
      ;;
    --cpu)
      is_cpu=True
      shift # is cpu image
      ;;
    --push)
      is_push=True
      shift # option for push image to remote repo
      ;;
    --help)
      usage
      exit 0
      ;;
    -*|--*)
      echo "Unknown option $i"
      usage
      exit 1
      ;;
    *)
      ;;
  esac
done

if [ "$cuda_version" == 11.3.0 ]; then
    echo "Building base image cuda11.3.0"
    BASE_GPU_IMAGE=$BASE_GPU_CUDA113_IMAGE
    cudatoolkit_version=cu113
elif [ "$cuda_version" == 11.7.1 ]; then
    echo "Building base image cuda11.7.1"
    cudatoolkit_version=cu117
    BASE_GPU_IMAGE=$BASE_GPU_CUDA117_IMAGE
elif [ "$cuda_version" == 11.8.0 ]; then
    echo "Building base image cuda11.8.0"
    cudatoolkit_version=cu118
    BASE_GPU_IMAGE=$BASE_GPU_CUDA118_IMAGE
else
    echo "Unsupport cuda version: $cuda_version"
    exit 1
fi

if [ "$is_cpu" == "True" ]; then
    export BASE_IMAGE=$BASE_CPU_IMAGE
    base_tag=ubuntu20.04
    export USE_GPU=False
else
    export BASE_IMAGE=$BASE_GPU_IMAGE
    base_tag=ubuntu20.04-cuda$cuda_version
    export USE_GPU=True
fi
if [[ $python_version == 3.7* ]]; then
    base_tag=$base_tag-py37
elif [[ $python_version == 3.8* ]]; then
    base_tag=$base_tag-py38
else
    echo "Unsupport python version: $python_version"
    exit 1
fi

target_image_tag=$base_tag-torch$torch_version-tf$tensorflow_version-base
export IMAGE_TO_BUILD=$MODELSCOPE_REPO_ADDRESS:$target_image_tag
export PYTHON_VERSION=$python_version
export TORCH_VERSION=$torch_version
export CUDATOOLKIT_VERSION=$cudatoolkit_version
export TENSORFLOW_VERSION=$tensorflow_version
echo -e "Building image with:\npython$python_version\npytorch$torch_version\ntensorflow:$tensorflow_version\ncudatoolkit:$cudatoolkit_version\ncpu:$is_cpu\n"
docker_file_content=`cat docker/Dockerfile.ubuntu_base`
printf "$docker_file_content" > Dockerfile

while true
do
  docker build -t $IMAGE_TO_BUILD  \
             --build-arg USE_GPU \
             --build-arg BASE_IMAGE \
             --build-arg PYTHON_VERSION \
             --build-arg TORCH_VERSION \
             --build-arg CUDATOOLKIT_VERSION \
             --build-arg TENSORFLOW_VERSION \
             -f Dockerfile .
  if [ $? -eq 0 ]; then
    echo "Image build done"
    break
  else
    echo "Running docker build command error, we will retry"
  fi
done

if [ "$is_push" == "True" ]; then
    echo "Pushing image: $IMAGE_TO_BUILD"
    docker push $IMAGE_TO_BUILD
fi
