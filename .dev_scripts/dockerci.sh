#!/bin/bash
IMAGE_NAME=reg.docker.alibaba-inc.com/dinger/modelscope
MODELSCOPE_CACHE_DIR_IN_CONTAINER=/modelscope_cache
CODE_DIR=$PWD
CODE_DIR_IN_CONTAINER=/Maas-lib
echo "$USER"
gpus='7 6 5 4 3 2 1 0'
cpu_sets='0-7 8-15 16-23 24-30 31-37 38-44 45-51 52-58'
cpu_sets_arr=($cpu_sets)
is_get_file_lock=false
for gpu in $gpus
do
  exec {lock_fd}>"/tmp/gpu$gpu" || exit 1
  flock -n "$lock_fd" || { echo "WARN: gpu $gpu is in use!" >&2; continue; }
  echo "get gpu lock $gpu"
  CONTAINER_NAME="modelscope-ci-$gpu"
  let is_get_file_lock=true
  docker run --rm --name $CONTAINER_NAME --shm-size=16gb \
             --cpuset-cpus=${cpu_sets_arr[$gpu]} \
             --gpus="device=$gpu" \
             -v $CODE_DIR:$CODE_DIR_IN_CONTAINER \
             -v $MODELSCOPE_CACHE:$MODELSCOPE_CACHE_DIR_IN_CONTAINER \
             -v $MODELSCOPE_HOME_CACHE/$gpu:/root \
             -v /home/admin/pre-commit:/home/admin/pre-commit \
             -e CI_TEST=True \
             -e MODELSCOPE_CACHE=$MODELSCOPE_CACHE_DIR_IN_CONTAINER \
             -e MODELSCOPE_DOMAIN=$MODELSCOPE_DOMAIN \
             -e HUB_DATASET_ENDPOINT=$HUB_DATASET_ENDPOINT \
             --workdir=$CODE_DIR_IN_CONTAINER \
             --net host  \
             ${IMAGE_NAME}:${IMAGE_VERSION} \
             bash .dev_scripts/ci_container_test.sh
  if [ $? -ne 0 ]; then
    echo "Running test case failed, please check the log!"
    exit -1
  fi
  break
done
if [ "$is_get_file_lock" = false ] ; then
    echo 'No free GPU!'
    exit 1
fi
