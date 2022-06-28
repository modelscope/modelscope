class NotExistError(Exception):
    pass


class RequestError(Exception):
    pass


class GitError(Exception):
    pass


def is_ok(rsp):
    """ Check the request is ok

    Args:
        rsp (_type_): The request response body
        Failed: {'Code': 10010101004, 'Message': 'get model info failed, err: unauthorized permission',
                 'RequestId': '', 'Success': False}
        Success: {'Code': 200, 'Data': {}, 'Message': 'success', 'RequestId': '', 'Success': True}
    """
    return rsp['Code'] == 200 and rsp['Success']


def raise_on_error(rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response
    """
    if rsp['Code'] == 200 and rsp['Success']:
        return True
    else:
        raise RequestError(rsp['Message'])


# TODO use raise_on_error instead if modelhub and datahub response have uniform structures,
def datahub_raise_on_error(url, rsp):
    """If response error, raise exception

    Args:
        rsp (_type_): The server response
    """
    if rsp.get('Code') == 200:
        return True
    else:
        raise RequestError(
            f"Url = {url}, Status = {rsp.get('status')}, error = {rsp.get('error')}, message = {rsp.get('message')}"
        )
