# 1. Build swift 4.1 image from a UBI9 base image
    Full build from a minimal base image, using a venv virtual environment.

## 1.1. Build
    ``` bash
    bash build.sh
    ```

## 1.2. Run a container
    ``` bash
    docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    --name base_image \
    ${IMAGE_ID} bash
    ```

## 1.3. Activate the venv environment
    ``` bash
    source /opt/venv/bin/activate
    ```

## 1.4. Run swift examples
    ``` bash
    cd /workspace/ms-swift
    bash examples/train/full/train.sh
    ```

# 2. Build swift 4.1 image from a Metax release image
    Faster build based on the pre-built Metax release image.

## 2.1. Build
    ``` bash
    bash build_from_metax_image.sh
    ```

## 2.2. Run a container
    ``` bash
    docker run -d -it --net=host --uts=host --ipc=host --privileged=true --group-add video  \
    --shm-size 100gb --ulimit memlock=-1 \
    --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
    --device=/dev/dri --device=/dev/mxcd \
    --name base_image \
    ${IMAGE_ID} bash
    ```

## 2.3. Run swift examples
    ``` bash
    cd /workspace/ms-swift
    bash examples/train/full/train.sh
    ```
