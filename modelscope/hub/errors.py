# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
from http import HTTPStatus

import requests
from requests.exceptions import HTTPError

from modelscope.hub.constants import MODELSCOPE_REQUEST_ID
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


class NotSupportError(Exception):
    pass


class NoValidRevisionError(Exception):
    pass


class NotExistError(Exception):
    pass


class RequestError(Exception):
    pass


class GitError(Exception):
    pass


class InvalidParameter(Exception):
    pass


class NotLoginException(Exception):
    pass


class FileIntegrityError(Exception):
    pass


class FileDownloadError(Exception):
    pass


def get_request_id(response: requests.Response):
    if MODELSCOPE_REQUEST_ID in response.request.headers:
        return response.request.headers[MODELSCOPE_REQUEST_ID]
    else:
        return ''


def is_ok(rsp):
    """ Check the request is ok

    Args:
        rsp (Response): The request response body

    Returns:
       bool: `True` if success otherwise `False`.
    """
    return rsp['Code'] == HTTPStatus.OK and rsp['Success']


def _decode_response_error(response: requests.Response):
    if 'application/json' in response.headers.get('content-type', ''):
        message = response.json()
    else:
        message = response.content.decode('utf-8')
    return message


def handle_http_post_error(response, url, request_body):
    try:
        response.raise_for_status()
    except HTTPError as error:
        message = _decode_response_error(response)
        raise HTTPError(
            'Request %s with body: %s exception, '
            'Response details: %s, request id: %s' %
            (url, request_body, message, get_request_id(response))) from error


def handle_http_response(response: requests.Response, logger, cookies,
                         model_id):
    http_error_msg = ''
    if isinstance(response.reason, bytes):
        try:
            reason = response.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = response.reason.decode('iso-8859-1')
    else:
        reason = response.reason
    request_id = get_request_id(response)
    if 404 == response.status_code:
        http_error_msg = 'The request model: %s does not exist!' % (model_id)
    elif 403 == response.status_code:
        if cookies is None:
            http_error_msg = 'Authentication token does not exist, '
            'failed to access model {model_id} which may not exist or may be '
            'private. Please login first.'
        else:
            http_error_msg = 'The authentication token is invalid, failed to access model {model_id}.'
    elif 400 <= response.status_code < 500:
        http_error_msg = u'%s Client Error: %s, Request id: %s for url: %s' % (
            response.status_code, reason, request_id, response.url)

    elif 500 <= response.status_code < 600:
        http_error_msg = u'%s Server Error: %s, Request id: %s, for url: %s' % (
            response.status_code, reason, request_id, response.url)
    if http_error_msg:  # there is error.
        logger.error(http_error_msg)
        raise HTTPError(http_error_msg, response=response)


def raise_on_error(rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response

    Raises:
        RequestError: the response error message.

    Returns:
        bool: True if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp['Code'] == HTTPStatus.OK:
        return True
    else:
        raise RequestError(rsp['Message'])


def datahub_raise_on_error(url, rsp, http_response: requests.Response):
    """If response error, raise exception

    Args:
        url (str): The request url
        rsp (HTTPResponse): The server response.
        http_response: the origin http response.

    Raises:
        RequestError: the http request error.

    Returns:
        bool: `True` if request is OK, otherwise raise `RequestError` exception.
    """
    if rsp.get('Code') == HTTPStatus.OK:
        return True
    else:
        request_id = get_request_id(http_response)
        raise RequestError(
            f"Url = {url}, Request id={request_id} Message = {rsp.get('Message')},\
                Please specify correct dataset_name and namespace.")


def raise_for_http_status(rsp):
    """Attempt to decode utf-8 first since some servers
    localize reason strings, for invalid utf-8, fall back
    to decoding with iso-8859-1.

    Args:
        rsp: The http response.

    Raises:
        HTTPError: The http error info.
    """
    http_error_msg = ''
    if isinstance(rsp.reason, bytes):
        try:
            reason = rsp.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = rsp.reason.decode('iso-8859-1')
    else:
        reason = rsp.reason
    request_id = get_request_id(rsp)
    if 404 == rsp.status_code:
        http_error_msg = 'The request resource(model or dataset) does not exist!,'
        'url: %s, reason: %s' % (rsp.url, reason)
    elif 403 == rsp.status_code:
        http_error_msg = 'Authentication token does not exist or invalid.'
    elif 400 <= rsp.status_code < 500:
        http_error_msg = u'%s Client Error: %s, Request id: %s for url: %s' % (
            rsp.status_code, reason, request_id, rsp.url)

    elif 500 <= rsp.status_code < 600:
        http_error_msg = u'%s Server Error: %s, Request id: %s, for url: %s' % (
            rsp.status_code, reason, request_id, rsp.url)

    if http_error_msg:
        req = rsp.request
        if req.method == 'POST':
            http_error_msg = u'%s, body: %s' % (http_error_msg, req.body)
        raise HTTPError(http_error_msg, response=rsp)
