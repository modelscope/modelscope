import unittest

import datasets as hfdata

from maas_lib.pydatasets import PyDataset


class PyDatasetTest(unittest.TestCase):

    def setUp(self):
        # ds1 initialized from in memory json
        self.json_data = {
            'dummy': [{
                'a': i,
                'x': i * 10,
                'c': i * 100
            } for i in range(1, 11)]
        }
        hfds1 = hfdata.Dataset.from_dict(self.json_data)
        self.ds1 = PyDataset.from_hf_dataset(hfds1)

        # ds2 initialized from hg hub
        hfds2 = hfdata.load_dataset(
            'glue', 'mrpc', revision='2.0.0', split='train')
        self.ds2 = PyDataset.from_hf_dataset(hfds2)

    def tearDown(self):
        pass

    def test_to_hf_dataset(self):
        hfds = self.ds1.to_hf_dataset()
        hfds1 = hfdata.Dataset.from_dict(self.json_data)
        self.assertEqual(hfds.data, hfds1.data)

        # simple map function
        hfds = hfds.map(lambda e: {'new_feature': e['dummy']['a']})
        self.assertEqual(len(hfds['new_feature']), 10)

        hfds2 = self.ds2.to_hf_dataset()
        self.assertTrue(hfds2[0]['sentence1'].startswith('Amrozi'))


if __name__ == '__main__':
    unittest.main()
