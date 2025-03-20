if __name__ == '__main__':

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login('35fcee05-01b7-4361-aaa4-e811003ac0c7')

    api.delete_dataset('wangxingjun778/test_ds_del_test')
