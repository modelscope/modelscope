# Copyright (c) Alibaba, Inc. and its affiliates.
"""Error classes — core exceptions delegate to modelscope_hub, legacy aliases retained.

Exception classes from modelscope_hub provide the structured hierarchy.
Legacy aliases maintain isinstance compatibility for existing code.
Error handling functions with unique logic are retained.
"""
import logging
from http import HTTPStatus
from typing import Optional

import requests
from requests.exceptions import HTTPError  # noqa: F401  (re-exported)

from modelscope_hub.errors import (  # noqa: F401
    APIError,
    AuthenticationError,
    CacheNotFound,
    CorruptedCacheException,
    FileIntegrityError,
    HubError,
    InvalidParameter,
    NetworkError,
    NotExistError,
    NotSupportedError,
    PermissionDeniedError,
    RequestTimeoutError,
    ServerError,
)

from modelscope.hub.constants import MODELSCOPE_REQUEST_ID
from modelscope.utils.logger import get_logger

logger = get_logger(log_level=logging.WARNING)


# --- Legacy exception aliases (maintain isinstance backward compatibility) ---

class RequestError(APIError):
    """Legacy alias — use APIError for new code."""

    def __init__(self, message: str = '', *args, **kwargs):
        # Preserve legacy single-positional-arg constructor signature.
        super().__init__(message, **kwargs)


class NotLoginException(AuthenticationError):
    """Legacy alias — use AuthenticationError for new code."""

    def __init__(self, message: str = '', *args, **kwargs):
        super().__init__(message, **kwargs)


class FileDownloadError(NetworkError):
    """Legacy alias — use NetworkError for new code."""
    pass


class NotSupportError(NotSupportedError):
    """Legacy alias — use NotSupportedError for new code."""
    pass


class NoValidRevisionError(NotExistError):
    """Legacy alias — raised when no valid revision is found."""

    def __init__(self, message: str = '', *args, **kwargs):
        super().__init__(message, **kwargs)


class GitError(HubError):
    """Git operation failure."""
    pass


# --- Error handling functions (retained - contain unique logic) ---

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


def handle_http_response(response: requests.Response,
                         logger,
                         cookies,
                         model_id,
                         raise_on_error: Optional[bool] = True) -> int:
    http_error_msg = ''
    if isinstance(response.reason, bytes):
        try:
            reason = response.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = response.reason.decode('iso-8859-1')
    else:
        reason = response.reason
    request_id = get_request_id(response)

    # Try to extract server-side error detail from JSON response body.
    server_message = ''
    if response.status_code >= 400:
        try:
            resp_json = response.json()
            # OpenAPI envelope: {"success": false, "code": "...", "message": "..."}
            msg = resp_json.get('message') or resp_json.get('Message') or ''
            code = resp_json.get('code') or ''
            if msg:
                server_message = f' | Server message: [{code}] {msg}' if code else f' | Server message: {msg}'
        except (ValueError, AttributeError):
            # Not JSON or unexpected structure; try raw text (truncated)
            body_text = response.text[:500] if response.text else ''
            if body_text:
                server_message = f' | Response body: {body_text}'

    if 404 == response.status_code:
        http_error_msg = (
            u'404 Not Found: %s does not exist or is not accessible, '
            u'Request id: %s for url: %s%s' %
            (model_id, request_id, response.url, server_message))
    elif 403 == response.status_code:
        if cookies is None:
            http_error_msg = (
                f'Authentication token does not exist, failed to access {model_id} '
                f'which may not exist or may be private. Please login first.{server_message}'
            )

        else:
            http_error_msg = (
                f'The authentication token is invalid, failed to access {model_id}.{server_message}'
            )
    elif 400 <= response.status_code < 500:
        http_error_msg = u'%s Client Error: %s, Request id: %s for url: %s%s' % (
            response.status_code, reason, request_id, response.url,
            server_message)

    elif 500 <= response.status_code < 600:
        http_error_msg = u'%s Server Error: %s, Request id: %s, for url: %s%s' % (
            response.status_code, reason, request_id, response.url,
            server_message)
    if http_error_msg and raise_on_error:  # there is error.
        logger.error(http_error_msg)
        raise HTTPError(http_error_msg, response=response)
    else:
        return response.status_code


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
        request_id = rsp['RequestId']
        raise RequestError(
            f"Url = {url}, Request id={request_id} Code = {rsp['Code']} Message = {rsp['Message']},\
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
