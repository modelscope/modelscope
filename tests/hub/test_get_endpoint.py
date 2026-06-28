import os
import unittest

from modelscope.hub.constants import (DEFAULT_MODELSCOPE_DATA_ENDPOINT,
                                              DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT,
                                              MODELSCOPE_DOMAIN)
from modelscope.hub.utils.utils import get_endpoint


class GetEndpointTest(unittest.TestCase):

    def setUp(self):
        self._original = os.environ.get(MODELSCOPE_DOMAIN)

    def tearDown(self):
        if self._original is not None:
            os.environ[MODELSCOPE_DOMAIN] = self._original
        elif MODELSCOPE_DOMAIN in os.environ:
            del os.environ[MODELSCOPE_DOMAIN]

    def test_default_cn_site(self):
        endpoint = get_endpoint(cn_site=True)
        self.assertEqual(endpoint, DEFAULT_MODELSCOPE_DATA_ENDPOINT)

    def test_default_intl_site(self):
        endpoint = get_endpoint(cn_site=False)
        self.assertEqual(endpoint, DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT)

    def test_env_domain_without_scheme(self):
        os.environ[MODELSCOPE_DOMAIN] = 'custom.modelscope.cn'
        endpoint = get_endpoint()
        self.assertEqual(endpoint, 'https://custom.modelscope.cn')

    def test_env_domain_with_https_scheme(self):
        os.environ[MODELSCOPE_DOMAIN] = 'https://custom.modelscope.cn'
        endpoint = get_endpoint()
        self.assertEqual(endpoint, 'https://custom.modelscope.cn')
        self.assertNotIn('https://https://', endpoint)

    def test_env_domain_with_http_scheme(self):
        os.environ[MODELSCOPE_DOMAIN] = 'http://custom.modelscope.cn'
        endpoint = get_endpoint()
        self.assertEqual(endpoint, 'http://custom.modelscope.cn')
        self.assertNotIn('https://http://', endpoint)


if __name__ == '__main__':
    unittest.main()
