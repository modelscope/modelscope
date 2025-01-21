# from modelscope.utils.constant import Tasks
# from modelscope.pipelines import pipeline

# pipe = pipeline(task=Tasks.speech_super_resolution, 
#                 model='ACoderPassBy/HifiSSR',
#                 )  # Use the version number you specified

# data={
#     "source_wav":"./syz.wav",
#     "ref_wav":"./p228_002.wav",
#     "out_wav":"./out.wav",
# }
# pipe(input=data)

from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline

pipe = pipeline(task=Tasks.voice_conversion, 
                model='ACoderPassBy/UnetVC',
                model_revision='v1.0.0'  # Repalce with your own model name
                )  # Use the version number you specified

data={
    "source_wav":"./syz.wav",
    "target_wav":"./p228_002.wav",
    "save_path":"./out.wav",
}
pipe(input=data)