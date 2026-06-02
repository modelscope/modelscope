# 1. build swift image from a ubi9 docker image
    Full build from a minimal base image, use venv virtual enviroment
## 1.1. build
    ``` bash
    bash build.sh
    ```
## 1.2. run a container
    ``` bash
    docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    --name base_image \
    ${IMAGE_ID} bash
    ```
## 1.3. activate venv environment
    here we use venv rather than conda
    ``` bash
    source /opt/venv/bin/activate
    ```
## 1.4. run swift examples
    cd /workspace/ms-swift
    bash example/train/full/train.sh

# 2. build swift image from metax release image
    Fast build based on the pre-built Metax release image, use conda virtual enviroment
## 2.1. build
    ``` bash
    bash build_from_metax_image.sh
    ```
## 2.2. run a container
    ``` bash
    docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    --name base_image \
    ${IMAGE_ID} bash
    ```
## 2.3. run swift examples
    cd /workspace/ms-swift
    bash example/train/full/train.sh
    ```
