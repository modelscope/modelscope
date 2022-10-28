# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.hub.api import HubApi


class DatasetDeleteManager(object):

    def __init__(self, dataset_name: str, namespace: str, version: str):
        self.api = HubApi()
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.version = version

    def delete(self, object_name: str) -> str:

        # single object
        if not object_name.endswith('/'):
            resp_msg = self.api.delete_oss_dataset_object(
                object_name=object_name,
                dataset_name=self.dataset_name,
                namespace=self.namespace,
                revision=self.version)
        else:
            #  multiple objects
            object_name = object_name.strip('/')
            resp_msg = self.api.delete_oss_dataset_dir(
                object_name=object_name,
                dataset_name=self.dataset_name,
                namespace=self.namespace,
                revision=self.version)

        return resp_msg
