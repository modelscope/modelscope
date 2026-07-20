# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from modelscope.utils.automodel_utils import check_model_from_owner_group


class OwnerGroupPathSafetyTest(unittest.TestCase):
    """Safety checks for trusted-owner cache path recognition."""

    def test_empty_name_cache_path_rejected(self):
        # modelscope_hub layout: {cache}/{owner}--{name}/snapshots/{rev}
        # Empty name ("iic--") must not be treated as a trusted owner path.
        self.assertFalse(
            check_model_from_owner_group('/cache/iic--/snapshots/v1'))
        self.assertFalse(
            check_model_from_owner_group('/cache/damo--/snapshots/v1'))

    def test_valid_and_spoof_cache_paths(self):
        self.assertTrue(
            check_model_from_owner_group('/cache/iic--x/snapshots/v1'))
        self.assertFalse(
            check_model_from_owner_group('/cache/--iic/snapshots/v1'))
        self.assertFalse(
            check_model_from_owner_group(
                '/cache/iic--hacked--evil/snapshots/v1'))
        self.assertTrue(check_model_from_owner_group('/cache/iic/some_model'))


if __name__ == '__main__':
    unittest.main()
