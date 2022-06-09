# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import os.path as osp
import tempfile
import unittest
from pathlib import Path

from modelscope.fileio import dump, load
from modelscope.utils.config import Config

obj = {'a': 1, 'b': {'c': [1, 2, 3], 'd': 'dd'}}


class ConfigTest(unittest.TestCase):

    def test_json(self):
        config_file = 'configs/examples/config.json'
        cfg = Config.from_file(config_file)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, obj['b'])

    def test_yaml(self):
        config_file = 'configs/examples/config.yaml'
        cfg = Config.from_file(config_file)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, obj['b'])

    def test_py(self):
        config_file = 'configs/examples/config.py'
        cfg = Config.from_file(config_file)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, obj['b'])

    def test_dump(self):
        config_file = 'configs/examples/config.py'
        cfg = Config.from_file(config_file)
        self.assertEqual(cfg.a, 1)
        self.assertEqual(cfg.b, obj['b'])
        pretty_text = 'a = 1\n'
        pretty_text += "b = dict(c=[1, 2, 3], d='dd')\n"

        json_str = '{"a": 1, "b": {"c": [1, 2, 3], "d": "dd"}}'
        yaml_str = 'a: 1\nb:\n  c:\n  - 1\n  - 2\n  - 3\n  d: dd\n'
        with tempfile.NamedTemporaryFile(suffix='.json') as ofile:
            self.assertEqual(pretty_text, cfg.dump())
            cfg.dump(ofile.name)
            with open(ofile.name, 'r') as infile:
                self.assertEqual(json_str, infile.read())

        with tempfile.NamedTemporaryFile(suffix='.yaml') as ofile:
            cfg.dump(ofile.name)
            with open(ofile.name, 'r') as infile:
                self.assertEqual(yaml_str, infile.read())

    def test_to_dict(self):
        config_file = 'configs/examples/config.json'
        cfg = Config.from_file(config_file)
        d = cfg.to_dict()
        print(d)
        self.assertTrue(isinstance(d, dict))

    def test_to_args(self):

        def parse_fn(args):
            parser = argparse.ArgumentParser(prog='PROG')
            parser.add_argument('--model-dir', default='')
            parser.add_argument('--lr', type=float, default=0.001)
            parser.add_argument('--optimizer', default='')
            parser.add_argument('--weight-decay', type=float, default=1e-7)
            parser.add_argument(
                '--save-checkpoint-epochs', type=int, default=30)
            return parser.parse_args(args)

        cfg = Config.from_file('configs/examples/plain_args.yaml')
        args = cfg.to_args(parse_fn)

        self.assertEqual(args.model_dir, 'path/to/model')
        self.assertAlmostEqual(args.lr, 0.01)
        self.assertAlmostEqual(args.weight_decay, 1e-6)
        self.assertEqual(args.optimizer, 'Adam')
        self.assertEqual(args.save_checkpoint_epochs, 20)


if __name__ == '__main__':
    unittest.main()
