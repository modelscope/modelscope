# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import shutil
import tempfile
import time
import unittest

import torch

from modelscope.utils.constant import Frameworks
from modelscope.utils.device import (create_device, device_placement,
                                     verify_device)

# import tensorflow must be imported after torch is imported when using tf1.15
import tensorflow as tf  # isort:skip


class DeviceTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def test_verify(self):
        device_name, device_id = verify_device('cpu')
        self.assertEqual(device_name, 'cpu')
        self.assertTrue(device_id is None)
        device_name, device_id = verify_device('CPU')
        self.assertEqual(device_name, 'cpu')

        device_name, device_id = verify_device('gpu')
        self.assertEqual(device_name, 'gpu')
        self.assertTrue(device_id == 0)

        device_name, device_id = verify_device('cuda')
        self.assertEqual(device_name, 'gpu')
        self.assertTrue(device_id == 0)

        device_name, device_id = verify_device('cuda:0')
        self.assertEqual(device_name, 'gpu')
        self.assertTrue(device_id == 0)

        device_name, device_id = verify_device('gpu:1')
        self.assertEqual(device_name, 'gpu')
        self.assertTrue(device_id == 1)

        with self.assertRaises(AssertionError):
            verify_device('xgu')

        with self.assertRaises(AssertionError):
            verify_device('')

        with self.assertRaises(AssertionError):
            verify_device(None)

    def test_create_device_torch(self):
        if torch.cuda.is_available():
            target_device_type = 'cuda'
            target_device_index = 0
        else:
            target_device_type = 'cpu'
            target_device_index = None
        device = create_device('gpu')
        self.assertTrue(isinstance(device, torch.device))
        self.assertTrue(device.type == target_device_type)
        self.assertTrue(device.index == target_device_index)

        device = create_device('gpu:0')
        self.assertTrue(isinstance(device, torch.device))
        self.assertTrue(device.type == target_device_type)
        self.assertTrue(device.index == target_device_index)

        device = create_device('cuda')
        self.assertTrue(device.type == target_device_type)
        self.assertTrue(isinstance(device, torch.device))
        self.assertTrue(device.index == target_device_index)

        device = create_device('cuda:0')
        self.assertTrue(isinstance(device, torch.device))
        self.assertTrue(device.type == target_device_type)
        self.assertTrue(device.index == target_device_index)

    def test_device_placement_cpu(self):
        with device_placement(Frameworks.torch, 'cpu'):
            pass

    @unittest.skip('skip this test to avoid debug logging.')
    def test_device_placement_tf_gpu(self):
        tf.debugging.set_log_device_placement(True)
        with device_placement(Frameworks.tf, 'gpu:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            s = tf.Session()
            s.run(c)
        tf.debugging.set_log_device_placement(False)

    def test_device_placement_torch_gpu(self):
        with device_placement(Frameworks.torch, 'gpu:0'):
            if torch.cuda.is_available():
                self.assertEqual(torch.cuda.current_device(), 0)


if __name__ == '__main__':
    unittest.main()
