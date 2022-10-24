import urllib
from abc import ABC, abstractmethod
from typing import Optional, Union

import json
from attr import fields
from attrs import asdict, define, field, validators


class Accelerator(object):
    CPU = 'cpu'
    GPU = 'gpu'


class Vendor(object):
    EAS = 'eas'


class EASRegion(object):
    beijing = 'cn-beijing'
    hangzhou = 'cn-hangzhou'


class EASCpuInstanceType(object):
    """EAS Cpu Instance TYpe, ref(https://help.aliyun.com/document_detail/144261.html)
    """
    tiny = 'ecs.c6.2xlarge'
    small = 'ecs.c6.4xlarge'
    medium = 'ecs.c6.6xlarge'
    large = 'ecs.c6.8xlarge'


class EASGpuInstanceType(object):
    """EAS Cpu Instance TYpe, ref(https://help.aliyun.com/document_detail/144261.html)
    """
    tiny = 'ecs.gn5-c28g1.7xlarge'
    small = 'ecs.gn5-c8g1.4xlarge'
    medium = 'ecs.gn6i-c24g1.12xlarge'
    large = 'ecs.gn6e-c12g1.3xlarge'


def min_smaller_than_max(instance, attribute, value):
    if value > instance.max_replica:
        raise ValueError(
            "'min_replica' value: %s has to be smaller than 'max_replica' value: %s!"
            % (value, instance.max_replica))


@define
class ServiceScalingConfig(object):
    """Resource scaling config
       Currently we ignore max_replica
    Args:
        max_replica: maximum replica
        min_replica: minimum replica
    """
    max_replica: int = field(default=1, validator=validators.ge(1))
    min_replica: int = field(
        default=1, validator=[validators.ge(1), min_smaller_than_max])


@define
class ServiceResourceConfig(object):
    """Eas Resource request.

    Args:
        accelerator: the accelerator(cpu|gpu)
        instance_type: the instance type.
        scaling: The instance scaling config.
    """
    instance_type: str
    scaling: ServiceScalingConfig
    accelerator: str = field(
        default=Accelerator.CPU,
        validator=validators.in_([Accelerator.CPU, Accelerator.GPU]))


@define
class ServiceParameters(ABC):
    pass


@define
class EASDeployParameters(ServiceParameters):
    """Parameters for EAS Deployment.

    Args:
        resource_group: the resource group to deploy, current default.
        region: The eas instance region(eg: cn-hangzhou).
        access_key_id: The eas account access key id.
        access_key_secret: The eas account access key secret.
        vendor: must be 'eas'
    """
    region: str
    access_key_id: str
    access_key_secret: str
    resource_group: Optional[str] = None
    vendor: str = field(
        default=Vendor.EAS, validator=validators.in_([Vendor.EAS]))
    """
    def __init__(self,
                 instance_name: str,
                 access_key_id: str,
                 access_key_secret: str,
                 region = EASRegion.beijing,
                 instance_type: str  = EASCpuInstances.small,
                 accelerator: str =  Accelerator.CPU,
                 resource_group: Optional[str] = None,
                 scaling: Optional[str] = None):
        self.instance_name=instance_name
        self.access_key_id=self.access_key_id
        self.access_key_secret = access_key_secret
        self.region = region
        self.instance_type = instance_type
        self.accelerator = accelerator
        self.resource_group = resource_group
        self.scaling = scaling
    """


@define
class EASListParameters(ServiceParameters):
    """EAS instance list parameters.

    Args:
        resource_group: the resource group to deploy, current default.
        region: The eas instance region(eg: cn-hangzhou).
        access_key_id: The eas account access key id.
        access_key_secret: The eas account access key secret.
        vendor: must be 'eas'
    """
    access_key_id: str
    access_key_secret: str
    region: str = None
    resource_group: str = None
    vendor: str = field(
        default=Vendor.EAS, validator=validators.in_([Vendor.EAS]))


@define
class DeployServiceParameters(object):
    """Deploy service parameters

    Args:
        instance_name: the name of the service.
        model_id: the modelscope model_id
        revision: the modelscope model revision
        resource: the resource requirement.
        provider: the cloud service provider.
    """
    instance_name: str
    model_id: str
    revision: str
    resource: ServiceResourceConfig
    provider: ServiceParameters


class AttrsToQueryString(ABC):
    """Convert the attrs class to json string.

    Args:
    """

    def to_query_str(self):
        self_dict = asdict(
            self.provider, filter=lambda attr, value: value is not None)
        json_str = json.dumps(self_dict)
        print(json_str)
        safe_str = urllib.parse.quote_plus(json_str)
        print(safe_str)
        query_param = 'provider=%s' % safe_str
        return query_param


@define
class ListServiceParameters(AttrsToQueryString):
    provider: ServiceParameters
    skip: int = 0
    limit: int = 100


@define
class GetServiceParameters(AttrsToQueryString):
    provider: ServiceParameters


@define
class DeleteServiceParameters(AttrsToQueryString):
    provider: ServiceParameters
