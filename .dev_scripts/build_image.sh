#!/bin/bash
# default values.
BASE_CPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04
BASE_GPU_IMAGE=reg.docker.alibaba-inc.com/modelscope/ubuntu:20.04-cuda11.3.0-cudnn8-devel
MODELSCOPE_REPO_ADDRESS=reg.docker.alibaba-inc.com/modelscope/modelscope
python_version=3.7.13
torch_version=1.11.0
cudatoolkit_version=11.3
tensorflow_version=1.15.5
modelscope_version=None
is_ci_test=False
is_dsw=False
is_cpu=False
run_ci_test=False
function usage(){
    echo "usage: build.sh "
    echo "       --python=python_version set python version, default: $python_version"
    echo "       --torch=torch_version set pytorch version, fefault: $torch_version"
    echo "       --cudatoolkit=cudatoolkit_version set cudatoolkit version used for pytorch, default: $cudatoolkit_version"
    echo "       --tensorflow=tensorflow_version set tensorflow version, default: $tensorflow_version"
    echo "       --modelscope=modelscope_version set modelscope version, default: $modelscope_version"
    echo "       --test option for run test before push image, only push on ci test pass"
    echo "       --cpu option for build cpu version"
    echo "       --dsw option for build dsw version"
    echo "       --ci  option for build ci version"
    echo "       --push option for push image to remote repo"
}
for i in "$@"; do
  case $i in
    --python=*)
      python_version="${i#*=}"
      shift
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
      shift # cudatoolkit for pytorch
      ;;
    --test)
      run_ci_test=True
      shift # will run ci test
      ;;
    --cpu)
      is_cpu=True
      shift # is cpu image
      ;;
    --ci)
      is_ci_test=True
      shift # is ci, will not install modelscope
      ;;
    --dsw)
      is_dsw=True
      shift # is dsw, will set dsw cache location
      ;;
    --push)
      is_push=True
      shift # is dsw, will set dsw cache location
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
    export BASE_IMAGE=$BASE_CPU_IMAGE
    base_tag=ubuntu20.04
    export USE_GPU=False
else
    export BASE_IMAGE=$BASE_GPU_IMAGE
    base_tag=ubuntu20.04-cuda11.3.0
    export USE_GPU=True
fi
if [[ $python_version == 3.7* ]]; then
    base_tag=$base_tag-py37
elif [[ $python_version == z* ]]; then
    base_tag=$base_tag-py38
elif [[ $python_version == z* ]]; then
    base_tag=$base_tag-py39
else
    echo "Unsupport python version: $python_version"
    exit 1
fi

target_image_tag=$base_tag-torch$torch_version-tf$tensorflow_version
if [ "$is_ci_test" == "True" ]; then
    target_image_tag=$target_image_tag-$modelscope_version-ci
else
    target_image_tag=$target_image_tag-$modelscope_version-test
fi
export IMAGE_TO_BUILD=$MODELSCOPE_REPO_ADDRESS:$target_image_tag
export PYTHON_VERSION=$python_version
export TORCH_VERSION=$torch_version
export CUDATOOLKIT_VERSION=$cudatoolkit_version
export TENSORFLOW_VERSION=$tensorflow_version
echo -e "Building image with:\npython$python_version\npytorch$torch_version\ntensorflow:$tensorflow_version\ncudatoolkit:$cudatoolkit_version\ncpu:$is_cpu\nis_ci:$is_ci_test\nis_dsw:$is_dsw\n"
docker_file_content=`cat docker/Dockerfile.ubuntu`
if [ "$is_ci_test" != "True" ]; then
    echo "Building ModelScope lib, will install ModelScope lib to image"
    docker_file_content="${docker_file_content} \nRUN pip install --no-cache-dir  modelscope==$modelscope_version -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html"
fi
echo "$is_dsw"
if [ "$is_dsw" == "False" ]; then
    echo "Not DSW image"
else
    echo "Building dsw image well need set ModelScope lib cache location."
    docker_file_content="${docker_file_content} \nENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope"
fi
printf "$docker_file_content" > Dockerfile
docker build -t $IMAGE_TO_BUILD  \
             --build-arg USE_GPU \
             --build-arg BASE_IMAGE \
             --build-arg PYTHON_VERSION \
             --build-arg TORCH_VERSION \
             --build-arg CUDATOOLKIT_VERSION \
             --build-arg TENSORFLOW_VERSION \
             -f Dockerfile .

if [ $? -ne 0 ]; then
  echo "Running docker build command error, please check the log!"
  exit -1
fi
if [ "$run_ci_test" == "True" ]; then
    echo "Running ci case."
    export MODELSCOPE_CACHE=/home/mulin.lyh/model_scope_cache
    export MODELSCOPE_HOME_CACHE=/home/mulin.lyh/ci_case_home # for credential
    export IMAGE_NAME=$MODELSCOPE_REPO_ADDRESS
    export IMAGE_VERSION=$target_image_tag
    export MODELSCOPE_DOMAIN=www.modelscope.cn
    export HUB_DATASET_ENDPOINT=http://www.modelscope.cn
    export CI_TEST=True
    export TEST_LEVEL=1
    if [ "$is_ci_test" != "True" ]; then
        echo "Testing for dsw image or MaaS-lib image"
        export CI_COMMAND="python tests/run.py"
    fi
    bash .dev_scripts/dockerci.sh
    if [ $? -ne 0 ]; then
       echo "Running unittest failed, please check the log!"
       exit -1
    fi
fi
if [ "$is_push" == "True" ]; then
    echo "Pushing image: $IMAGE_TO_BUILD"
    docker push $IMAGE_TO_BUILD
fi
