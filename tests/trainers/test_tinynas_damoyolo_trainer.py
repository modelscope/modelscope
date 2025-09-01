# Copyright (c) Alibaba, Inc. and its affiliates.		# Copyright (c) Alibaba, Inc. and its affiliates.


import os		import os
import shutil		import shutil
import subprocess
import unittest		import unittest


from modelscope.hub.snapshot_download import snapshot_download		from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Trainers		from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer		from modelscope.trainers import build_trainer
from modelscope.utils.test_utils import test_level		from modelscope.utils.test_utils import test_level
def _setup():		def _setup():
    model_id = 'damo/cv_tinynas_object-detection_damoyolo'		    model_id = 'damo/cv_tinynas_object-detection_damoyolo'
    cache_path = snapshot_download(model_id)		    cache_path = snapshot_download(model_id)
    return cache_path		    return cache_path
class TestTinynasDamoyoloTrainerSingleGPU(unittest.TestCase):		class TestTinynasDamoyoloTrainerSingleGPU(unittest.TestCase):


    def setUp(self):		    def setUp(self):
        # pycocotools==2.0.8
        subprocess.getstatusoutput('pip install pycocotools==2.0.8')
        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo'		        self.model_id = 'damo/cv_tinynas_object-detection_damoyolo'
        self.cache_path = _setup()		        self.cache_path = _setup()


    def tearDown(self) -> None:		    def tearDown(self) -> None:
        super().tearDown()		        super().tearDown()
        shutil.rmtree('./workdirs')		        shutil.rmtree('./workdirs', ignore_errors=True)


    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')		    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_from_scratch_singleGPU(self):		    def test_trainer_from_scratch_singleGPU(self):
        kwargs = dict(		        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),		            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[		            gpu_ids=[
                0,		                0,
            ],		            ],
            batch_size=2,		            batch_size=2,
            max_epochs=3,		            max_epochs=3,
            num_classes=80,		            num_classes=80,
            base_lr_per_img=0.001,		            base_lr_per_img=0.001,
            cache_path=self.cache_path,		            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',		            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',		            val_image_dir='./data/test/images/image_detection/images',
            train_ann=		            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=		            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            work_dir='./workdirs',		            work_dir='./workdirs',
            exp_name='damoyolo_s',		            exp_name='damoyolo_s',
        )		        )
        trainer = build_trainer(		        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)		            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()		        trainer.train()
        trainer.evaluate(		        trainer.evaluate(
            checkpoint_path=os.path.join('./workdirs/damoyolo_s',		            checkpoint_path=os.path.join('./workdirs/damoyolo_s',
                                         'epoch_3_ckpt.pth'))		                                         'epoch_3_ckpt.pth'))
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')		    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_from_scratch_singleGPU_model_id(self):		    def test_trainer_from_scratch_singleGPU_model_id(self):
        kwargs = dict(		        kwargs = dict(
            model=self.model_id,		            model=self.model_id,
            gpu_ids=[		            gpu_ids=[
                0,		                0,
            ],		            ],
            batch_size=2,		            batch_size=2,
            max_epochs=3,		            max_epochs=3,
            num_classes=80,		            num_classes=80,
            load_pretrain=True,		            load_pretrain=True,
            base_lr_per_img=0.001,		            base_lr_per_img=0.001,
            train_image_dir='./data/test/images/image_detection/images',		            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',		            val_image_dir='./data/test/images/image_detection/images',
            train_ann=		            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=		            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            work_dir='./workdirs',		            work_dir='./workdirs',
            exp_name='damoyolo_s',		            exp_name='damoyolo_s',
        )		        )
        trainer = build_trainer(		        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)		            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()		        trainer.train()
        trainer.evaluate(		        trainer.evaluate(
            checkpoint_path=os.path.join(self.cache_path,		            checkpoint_path=os.path.join(self.cache_path,
                                         'damoyolo_tinynasL25_S.pt'))		                                         'damoyolo_tinynasL25_S.pt'))
    @unittest.skip('multiGPU test is verified offline')		    @unittest.skip('multiGPU test is verified offline')
    def test_trainer_from_scratch_multiGPU(self):		    def test_trainer_from_scratch_multiGPU(self):
        kwargs = dict(		        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),		            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[		            gpu_ids=[
                0,		                0,
                1,		                1,
            ],		            ],
            batch_size=32,		            batch_size=32,
            max_epochs=3,		            max_epochs=3,
            num_classes=1,		            num_classes=1,
            cache_path=self.cache_path,		            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',		            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',		            val_image_dir='./data/test/images/image_detection/images',
            train_ann=		            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=		            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            work_dir='./workdirs',		            work_dir='./workdirs',
            exp_name='damoyolo_s',		            exp_name='damoyolo_s',
        )		        )
        trainer = build_trainer(		        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)		            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()		        trainer.train()
    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')		    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_finetune_singleGPU(self):		    def test_trainer_finetune_singleGPU(self):
        kwargs = dict(		        kwargs = dict(
            cfg_file=os.path.join(self.cache_path, 'configuration.json'),		            cfg_file=os.path.join(self.cache_path, 'configuration.json'),
            gpu_ids=[		            gpu_ids=[
                0,		                0,
            ],		            ],
            batch_size=16,		            batch_size=16,
            max_epochs=3,		            max_epochs=3,
            num_classes=1,		            num_classes=1,
            load_pretrain=True,		            load_pretrain=True,
            pretrain_model=os.path.join(self.cache_path,		            pretrain_model=os.path.join(self.cache_path,
                                        'damoyolo_tinynasL25_S.pt'),		                                        'damoyolo_tinynasL25_S.pt'),
            cache_path=self.cache_path,		            cache_path=self.cache_path,
            train_image_dir='./data/test/images/image_detection/images',		            train_image_dir='./data/test/images/image_detection/images',
            val_image_dir='./data/test/images/image_detection/images',		            val_image_dir='./data/test/images/image_detection/images',
            train_ann=		            train_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            val_ann=		            val_ann=
            './data/test/images/image_detection/annotations/coco_sample.json',		            './data/test/images/image_detection/annotations/coco_sample.json',
            work_dir='./workdirs',		            work_dir='./workdirs',
            exp_name='damoyolo_s',		            exp_name='damoyolo_s',
        )		        )
        trainer = build_trainer(		        trainer = build_trainer(
            name=Trainers.tinynas_damoyolo, default_args=kwargs)		            name=Trainers.tinynas_damoyolo, default_args=kwargs)
        trainer.train()		        trainer.train()
if __name__ == '__main__':		if __name__ == '__main__':
    unittest.main()
