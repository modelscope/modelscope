# Copyright (c) Alibaba, Inc. and its affiliates.

from http import HTTPStatus

from requests.exceptions import HTTPError

from modelscope.utils.logger import get_logger

logger = get_logger()


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


def is_ok(rsp):
    """ Check the request is ok

    Args:
        rsp (_type_): The request response body
        Failed: {'Code': 10010101004, 'Message': 'get model info failed, err: unauthorized permission',
                 'RequestId': '', 'Success': False}
        Success: {'Code': 200, 'Data': {}, 'Message': 'success', 'RequestId': '', 'Success': True}
    """
    return rsp['Code'] == HTTPStatus.OK and rsp['Success']


def handle_http_post_error(response, url, request_body):
    try:
        response.raise_for_status()
    except HTTPError as error:
        logger.error('Request %s with body: %s exception' %
                     (url, request_body))
        raise error


def handle_http_response(response, logger, cookies, model_id):
    try:
        response.raise_for_status()
    except HTTPError as error:
        if cookies is None:  # code in [403] and
            logger.error(
                f'Authentication token does not exist, failed to access model {model_id} which may not exist or may be \
                private. Please login first.')
        logger.error('Response details: %s' % response.content)
        raise error


def raise_on_error(rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response
    """
    if rsp['Code'] == HTTPStatus.OK:
        return True
    else:
        raise RequestError(rsp['Message'])


# TODO use raise_on_error instead if modelhub and datahub response have uniform structures,
def datahub_raise_on_error(url, rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response
    """
    if rsp.get('Code') == HTTPStatus.OK:
        return True
    else:
        raise RequestError(
            f"Url = {url}, Status = {rsp.get('status')}, error = {rsp.get('error')}, message = {rsp.get('message')}"
        )


def raise_for_http_status(rsp):
    """
    Attempt to decode utf-8 first since some servers
    localize reason strings, for invalid utf-8, fall back
    to decoding with iso-8859-1.
    """
    http_error_msg = ''
    if isinstance(rsp.reason, bytes):
        try:
            reason = rsp.reason.decode('utf-8')
        except UnicodeDecodeError:
            reason = rsp.reason.decode('iso-8859-1')
    else:
        reason = rsp.reason

    if 400 <= rsp.status_code < 500:
        http_error_msg = u'%s Client Error: %s for url: %s' % (rsp.status_code,
                                                               reason, rsp.url)

    elif 500 <= rsp.status_code < 600:
        http_error_msg = u'%s Server Error: %s for url: %s' % (rsp.status_code,
                                                               reason, rsp.url)

    if http_error_msg:
        req = rsp.request
        if req.method == 'POST':
            http_error_msg = u'%s, body: %s' % (http_error_msg, req.body)
        raise HTTPError(http_error_msg, response=rsp)
