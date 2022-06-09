# MTCNN

`pytorch` implementation of **inference stage** of face detection algorithm described in
[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

## Example
![example of a face detection](images/example.png)

## How to use it
Just download the repository and then do this
```python
from src import detect_faces
from PIL import Image

image = Image.open('image.jpg')
bounding_boxes, landmarks = detect_faces(image)
```
For examples see `test_on_images.ipynb`.

## Requirements
* pytorch 0.2
* Pillow, numpy

## Credit
This implementation is heavily inspired by:
* [pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
