#sudo docker run --name zwm_maas -v /home/wenmeng.zwm/workspace:/home/wenmeng.zwm/workspace   --net host -ti reg.docker.alibaba-inc.com/pai-dlc/tensorflow-training:2.3-gpu-py36-cu101-ubuntu18.04 bash
#sudo docker run --name zwm_maas_pytorch -v /home/wenmeng.zwm/workspace:/home/wenmeng.zwm/workspace --net host  -ti reg.docker.alibaba-inc.com/pai-dlc/pytorch-training:1.10PAI-gpu-py36-cu113-ubuntu18.04 bash
CONTAINER_NAME=modelscope-dev
IMAGE_NAME=registry.cn-shanghai.aliyuncs.com/modelscope/modelscope
IMAGE_VERSION=v0.1.1-16-g62856fa-devel
MOUNT_DIR=/home/wenmeng.zwm/workspace
sudo docker run --name  $CONTAINER_NAME -v $MOUNT_DIR:$MOUNT_DIR --net host  -ti ${IMAGE_NAME}:${IMAGE_VERSION} bash
