import unittest

from maas_hub.maas_api import MaasApi

from modelscope.utils.hub import create_model_if_not_exist

USER_NAME = 'maasadmin'
PASSWORD = '12345678'


class HubExampleTest(unittest.TestCase):

    def setUp(self):
        self.api = MaasApi()
        # note this is temporary before official account management is ready
        self.api.login(USER_NAME, PASSWORD)

    @unittest.skip('to be used for local test only')
    def test_example_model_creation(self):
        # ATTENTION:change to proper model names before use
        model_name = 'cv_unet_person-image-cartoon_compound-models'
        model_chinese_name = '达摩卡通化模型'
        model_org = 'damo'
        model_id = '%s/%s' % (model_org, model_name)

        created = create_model_if_not_exist(self.api, model_id,
                                            model_chinese_name)
        if not created:
            print('!! NOT created since model already exists !!')


if __name__ == '__main__':
    unittest.main()
