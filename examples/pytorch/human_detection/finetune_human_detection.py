import os.path as osp
from argparse import ArgumentParser

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import DownloadMode

parser = ArgumentParser()
parser.add_argument('--dataset_name', type=str, help='The dataset name')
parser.add_argument('--namespace', type=str, help='The dataset namespace')
parser.add_argument('--model', type=str, help='The model id or model dir')
parser.add_argument(
    '--num_classes', type=int, help='The num_classes in the dataset')
parser.add_argument('--batch_size', type=int, help='The training batch size')
parser.add_argument('--max_epochs', type=int, help='The training max epochs')
parser.add_argument(
    '--base_lr_per_img',
    type=float,
    help='The base learning rate for per image')

args = parser.parse_args()
print(args)

# Step 1: 数据集准备，可以使用modelscope上已有的数据集，也可以自己在本地构建COCO数据集
train_dataset = MsDataset.load(
    args.dataset_name,
    namespace=args.namespace,
    split='train',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)
val_dataset = MsDataset.load(
    args.dataset_name,
    namespace=args.namespace,
    split='validation',
    download_mode=DownloadMode.FORCE_REDOWNLOAD)

# Step 2: 相关参数设置
train_root_dir = train_dataset.config_kwargs['split_config']['train']
val_root_dir = val_dataset.config_kwargs['split_config']['validation']
train_img_dir = osp.join(train_root_dir, 'images')
val_img_dir = osp.join(val_root_dir, 'images')
train_anno_path = osp.join(train_root_dir, 'train.json')
val_anno_path = osp.join(val_root_dir, 'val.json')
kwargs = dict(
    model=args.model,  # 使用DAMO-YOLO-S模型
    gpu_ids=[  # 指定训练使用的gpu
        0,
    ],
    batch_size=args.
    batch_size,  # batch_size, 每个gpu上的图片数等于batch_size // len(gpu_ids)
    max_epochs=args.max_epochs,  # 总的训练epochs
    num_classes=args.num_classes,  # 自定义数据中的类别数
    load_pretrain=True,  # 是否载入预训练模型，若为False，则为从头重新训练
    base_lr_per_img=args.
    base_lr_per_img,  # 每张图片的学习率，lr=base_lr_per_img*batch_size
    train_image_dir=train_img_dir,  # 训练图片路径
    val_image_dir=val_img_dir,  # 测试图片路径
    train_ann=train_anno_path,  # 训练标注文件路径
    val_ann=val_anno_path,  # 测试标注文件路径
)

# Step 3: 开启训练任务
trainer = build_trainer(name=Trainers.tinynas_damoyolo, default_args=kwargs)
trainer.train()
