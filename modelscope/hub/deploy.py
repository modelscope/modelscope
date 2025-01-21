import urllib
from abc import ABC
from http import HTTPStatus
from typing import Optional

import json
import requests
from attrs import asdict, define, field, validators

from modelscope.hub.api import ModelScopeConfig
from modelscope.hub.constants import (API_RESPONSE_FIELD_DATA,
                                      API_RESPONSE_FIELD_MESSAGE)
from modelscope.hub.errors import (NotLoginException, NotSupportError,
                                   RequestError, handle_http_response, is_ok,
                                   raise_for_http_status)
from modelscope.hub.utils.utils import get_endpoint
from modelscope.utils.logger import get_logger

# yapf: enable

logger = get_logger()


class Accelerator(object):
    CPU = 'cpu'
    GPU = 'gpu'


class Vendor(object):
    EAS = 'eas'


class EASRegion(object):
    beijing = 'cn-beijing'
    hangzhou = 'cn-hangzhou'


class EASCpuInstanceType(object):
    """EAS Cpu Instance Type, ref(https://help.aliyun.com/document_detail/144261.html)
    """
    tiny = 'ecs.c6.2xlarge'
    small = 'ecs.c6.4xlarge'
    medium = 'ecs.c6.6xlarge'
    large = 'ecs.c6.8xlarge'


class EASGpuInstanceType(object):
    """EAS Gpu Instance Type, ref(https://help.aliyun.com/document_detail/144261.html)
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
class ServiceProviderParameters(ABC):
    pass


@define
class EASDeployParameters(ServiceProviderParameters):
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


@define
class EASListParameters(ServiceProviderParameters):
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
    provider: ServiceProviderParameters


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
    provider: ServiceProviderParameters
    skip: int = 0
    limit: int = 100


@define
class GetServiceParameters(AttrsToQueryString):
    provider: ServiceProviderParameters


@define
class DeleteServiceParameters(AttrsToQueryString):
    provider: ServiceProviderParameters


class ServiceDeployer(object):
    """Facilitate model deployment on to supported service provider(s).
    """

    def __init__(self, endpoint=None):
        self.endpoint = endpoint if endpoint is not None else get_endpoint()
        self.headers = {'user-agent': ModelScopeConfig.get_user_agent()}
        self.cookies = ModelScopeConfig.get_cookies()
        if self.cookies is None:
            raise NotLoginException(
                'Token does not exist, please login with HubApi first.')

    # deploy_model
    def create(self, model_id: str, revision: str, instance_name: str,
               resource: ServiceResourceConfig,
               provider: ServiceProviderParameters):
        """Deploy model to cloud, current we only support PAI EAS, this is an async API ,
        and the deployment could take a while to finish remotely. Please check deploy instance
        status separately via checking the status.

        Args:
            model_id (str): The deployed model id
            revision (str): The model revision
            instance_name (str): The deployed model instance name.
            resource (ServiceResourceConfig): The service resource information.
            provider (ServiceProviderParameters): The service provider parameter

        Raises:
            NotSupportError: Not supported platform.
            RequestError: The server return error.

        Returns:
            ServiceInstanceInfo: The information of the deployed service instance.
        """
        if provider.vendor != Vendor.EAS:
            raise NotSupportError(
                'Not support vendor: %s ,only support EAS current.' %
                (provider.vendor))
        create_params = DeployServiceParameters(
            instance_name=instance_name,
            model_id=model_id,
            revision=revision,
            resource=resource,
            provider=provider)
        path = f'{self.endpoint}/api/v1/deployer/endpoint'
        body = asdict(create_params)
        r = requests.post(
            path, json=body, cookies=self.cookies, headers=self.headers)
        handle_http_response(r, logger, self.cookies, 'create_service')
        if r.status_code >= HTTPStatus.OK and r.status_code < HTTPStatus.MULTIPLE_CHOICES:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None

    def get(self, instance_name: str, provider: ServiceProviderParameters):
        """Query the specified instance information.

        Args:
            instance_name (str): The deployed instance name.
            provider (ServiceProviderParameters): The cloud provider information, for eas
                need region(eg: ch-hangzhou), access_key_id and access_key_secret.

        Raises:
            RequestError: The request is failed from server.

        Returns:
            Dict: The information of the requested service instance.
        """
        params = GetServiceParameters(provider=provider)
        path = '%s/api/v1/deployer/endpoint/%s?%s' % (
            self.endpoint, instance_name, params.to_query_str())
        r = requests.get(path, cookies=self.cookies, headers=self.headers)
        handle_http_response(r, logger, self.cookies, 'get_service')
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None

    def delete(self, instance_name: str, provider: ServiceProviderParameters):
        """Delete deployed model, this api send delete command and return, it will take
        some to delete, please check through the cloud console.

        Args:
            instance_name (str): The instance name you want to delete.
            provider (ServiceProviderParameters): The cloud provider information, for eas
                need region(eg: ch-hangzhou), access_key_id and access_key_secret.

        Raises:
            RequestError: The request is failed.

        Returns:
            Dict: The deleted instance information.
        """
        params = DeleteServiceParameters(provider=provider)
        path = '%s/api/v1/deployer/endpoint/%s?%s' % (
            self.endpoint, instance_name, params.to_query_str())
        r = requests.delete(path, cookies=self.cookies, headers=self.headers)
        handle_http_response(r, logger, self.cookies, 'delete_service')
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None

    def list(self,
             provider: ServiceProviderParameters,
             skip: Optional[int] = 0,
             limit: Optional[int] = 100):
        """List deployed model instances.

        Args:
            provider (ServiceProviderParameters): The cloud service provider parameter,
                for eas, need access_key_id and access_key_secret.
            skip (int, optional): start of the list, current not support.
            limit (int, optional): maximum number of instances return, current not support

        Raises:
            RequestError: The request is failed from server.

        Returns:
            List: List of instance information
        """

        params = ListServiceParameters(
            provider=provider, skip=skip, limit=limit)
        path = '%s/api/v1/deployer/endpoint?%s' % (self.endpoint,
                                                   params.to_query_str())
        r = requests.get(path, cookies=self.cookies, headers=self.headers)
        handle_http_response(r, logger, self.cookies, 'list_service_instances')
        if r.status_code == HTTPStatus.OK:
            if is_ok(r.json()):
                data = r.json()[API_RESPONSE_FIELD_DATA]
                return data
            else:
                raise RequestError(r.json()[API_RESPONSE_FIELD_MESSAGE])
        else:
            raise_for_http_status(r)
        return None
