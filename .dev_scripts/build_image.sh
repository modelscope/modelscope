#!/bin/bash
# default values.
#BASE_PY37_CPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-py37-torch1.11.0-tf1.15.5-base
#BASE_PY38_CPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-py38-torch1.11.0-tf1.15.5-base
#BASE_PY38_GPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda11.3.0-py38-torch1.11.0-tf1.15.5-base
#BASE_PY38_GPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-base
#BASE_PY38_GPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda11.7.1-py38-torch1.13.1-tf2.6.0-base
#BASE_PY37_GPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-base
MODELSCOPE_REPO_ADDRESS=reg.docker.alibaba-inc.com/modelscope/modelscope
python_version=3.7.13
torch_version=1.11.0
cudatoolkit_version=11.7
tensorflow_version=1.15.5
modelscope_version=None
cuda_version=11.7.1
is_dsw=False
is_cpu=False
build_branch='master'
function usage(){
    echo "usage: build.sh "
    echo "       --python=python_version set python version, default: $python_version"
    echo "       --cuda=cuda_version set cuda version,only[11.3.0, 11.7.1], fefault: $cuda_version"
    echo "       --torch=torch_version set pytorch version, fefault: $torch_version"
    echo "       --tensorflow=tensorflow_version set tensorflow version, default: $tensorflow_version"
    echo "       --modelscope=modelscope_version set modelscope version, default: $modelscope_version"
    echo "       --branch=build_branch set modelscope build branch, default: $build_branch"
    echo "       --cpu option for build cpu version"
    echo "       --dsw option for build dsw version"
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
      if [ "$cuda_version" == "11.3.0" ]; then
          cudatoolkit_version=11.3
      elif [ "$cuda_version" == "11.7.1" ]; then
          cudatoolkit_version=11.7
      elif [ "$cuda_version" == "11.8.0" ]; then
          cudatoolkit_version=11.8
      elif [ "$cuda_version" == "12.1.0" ]; then
          cudatoolkit_version=12.1
      else
          echo "Unsupport cuda version $cuda_version"
          exit 1
      fi
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
    --cudatoolkit=*)
      cudatoolkit_version="${i#*=}"
      shift # cudatoolkit for pytorch
      ;;
    --modelscope=*)
      modelscope_version="${i#*=}"
      shift # modelscope version
      ;;
    --branch=*)
      build_branch="${i#*=}"
      shift # build branch
      ;;
    --cpu)
      is_cpu=True
      shift # is cpu image
      ;;
    --dsw)
      is_dsw=True
      shift # is dsw, will set dsw cache location
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

if [ "$modelscope_version" == "None" ]; then
    echo "ModelScope version must specify!"
    exit 1
fi
if [ "$is_cpu" == "True" ]; then
    base_tag=ubuntu20.04
    export USE_GPU=False
else
    base_tag=ubuntu20.04-cuda$cuda_version
    export USE_GPU=True
fi

if [[ $python_version == 3.7* ]]; then
    if [ "$is_cpu" == "True" ]; then
        echo "Building python3.7 cpu image"
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-py37-torch$torch_version-tf$tensorflow_version-base
    else
        echo "Building python3.7 gpu image"
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda$cuda_version-py37-torch$torch_version-tf$tensorflow_version-base
    fi
    base_tag=$base_tag-py37
elif [[ $python_version == 3.8* ]]; then
    if [ "$is_cpu" == "True" ]; then
        echo "Building python3.8 cpu image"
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-py38-torch$torch_version-tf$tensorflow_version-base
    else
        echo "Building python3.8 gpu image"
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu20.04-cuda$cuda_version-py38-torch$torch_version-tf$tensorflow_version-base
    fi
    base_tag=$base_tag-py38
elif [[ $python_version == 3.10* ]]; then
    if [ "$is_cpu" == "True" ]; then
        echo "Building python3.10 cpu image"
        base_tag=ubuntu22.04-py310
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-py310-torch$torch_version-base
    else
        echo "Building python3.10 gpu image"
        base_tag=ubuntu22.04-cuda$cuda_version-py310
        # reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.0-tf2.14.0-base
        export BASE_IMAGE=reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda$cuda_version-py310-torch$torch_version-tf$tensorflow_version-base
    fi
else
    echo "Unsupport python version: $python_version"
    exit 1
fi
# cpu not intall tensorflow
# target_image_tag=$base_tag-torch$torch_version-tf$tensorflow_version-$modelscope_version-test
if [ "$is_cpu" == "True" ]; then
    target_image_tag=$base_tag-torch$torch_version-$modelscope_version-test
else
    target_image_tag=$base_tag-torch$torch_version-tf$tensorflow_version-$modelscope_version-test
fi
export IMAGE_TO_BUILD=$MODELSCOPE_REPO_ADDRESS:$target_image_tag
export PYTHON_VERSION=$python_version
export TORCH_VERSION=$torch_version
export CUDATOOLKIT_VERSION=$cudatoolkit_version
export TENSORFLOW_VERSION=$tensorflow_version
echo -e "Building image with:\npython$python_version\npytorch$torch_version\ntensorflow:$tensorflow_version\ncudatoolkit:$cudatoolkit_version\ncpu:$is_cpu\nis_ci:$is_ci_test\nis_dsw:$is_dsw\n"
echo -e "Base iamge: $BASE_IMAGE"
docker_file_content=`cat docker/Dockerfile.ubuntu`

BUILD_HASH_ID=$(git rev-parse HEAD)
# install thrid part library
docker_file_content="${docker_file_content} \nRUN export COMMIT_ID=$BUILD_HASH_ID && pip install --no-cache-dir -U adaseq pai-easycv &&  pip install --no-cache-dir -U git+https://github.com/modelscope/ms-swift.git@release/2.4 'decord' 'qwen_vl_utils' 'pyav' 'librosa' 'funasr' autoawq 'timm>0.9.5' 'accelerate' 'gradio' 'peft' 'optimum' 'trl' && pip install git+https://github.com/huggingface/transformers.git"

docker_file_content="${docker_file_content} \nRUN pip uninstall modelscope -y && export COMMIT_ID=$BUILD_HASH_ID && cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $build_branch  --single-branch $REPO_URL && cd modelscope && pip install . && cd / && rm -fr /tmp/modelscope && pip cache purge;"

echo "$is_dsw"
if [ "$is_dsw" == "False" ]; then
    echo "Not DSW image"
else
    echo "Building dsw image will need set ModelScope lib cache location."
    docker_file_content="${docker_file_content} \nENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope"
    # pre compile extension
    docker_file_content="${docker_file_content} \nRUN pip uninstall -y tb-nightly tensorboard && pip install --no-cache-dir -U tensorboard && TORCH_CUDA_ARCH_LIST='6.0 6.1 7.0 7.5 8.0 8.9 9.0 8.6+PTX' python -c 'from modelscope.utils.pre_compile import pre_compile_all;pre_compile_all()'"
fi


docker_file_content="${docker_file_content} \n RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    cp /tmp/resources/ubuntu2204.aliyun /etc/apt/sources.list "

printf "$docker_file_content" > Dockerfile

while true
do
  docker build --progress=plain -t $IMAGE_TO_BUILD  \
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
