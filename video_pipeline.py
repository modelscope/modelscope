import cv2

from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks

pipe = pipeline(
    task=Tasks.text_to_video_synthesis,
    model='buptwq/videocomposer',
    model_revision='v1.0.1')
ds = MsDataset.load(
    'buptwq/videocomposer-depths-style',
    split='train',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)
inputs = next(iter(ds))
inputs.update({
    'text':
    'A glittering and translucent fish swimming in a small glass bowl with multicolored piece of stone, like a glass fish'
})
print('inputs: ', inputs)
output = pipe(inputs)
print('output: ', output)
