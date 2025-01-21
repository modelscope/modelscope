import os

from modelscope.metainfo import Trainers
from modelscope.msdatasets.dataset_cls.custom_datasets import ASRDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode


def modelscope_finetune(params):
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir, exist_ok=True)
    # dataset split ["train", "validation"]
    ds_dict = ASRDataset.load(
        params.data_path,
        namespace='speech_asr',
        download_mode=params.download_mode)
    kwargs = dict(
        model=params.model,
        data_dir=ds_dict,
        dataset_type=params.dataset_type,
        work_dir=params.output_dir,
        batch_bins=params.batch_bins,
        max_epoch=params.max_epoch,
        lr=params.lr)
    trainer = build_trainer(Trainers.speech_asr_trainer, default_args=kwargs)
    trainer.train()


if __name__ == '__main__':
    from funasr.utils.modelscope_param import modelscope_args

    params = modelscope_args(
        model=
        'damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
    )
    params.output_dir = './checkpoint'  # 模型保存路径
    params.data_path = 'speech_asr_aishell1_trainsets'  # 数据路径，可以为modelscope中已上传数据，也可以是本地数据
    params.dataset_type = 'small'  # 小数据量设置small，若数据量大于1000小时，请使用large
    params.batch_bins = 2000  # batch size，如果dataset_type="small"，batch_bins单位为fbank特征帧数，
    # 如果dataset_type="large"，batch_bins单位为毫秒，
    params.max_epoch = 50  # 最大训练轮数
    params.lr = 0.00005  # 设置学习率
    params.download_mode = DownloadMode.FORCE_REDOWNLOAD  # 重新下载数据，否则设置为默认值DownloadMode.REUSE_DATASET_IF_EXISTS

    modelscope_finetune(params)
