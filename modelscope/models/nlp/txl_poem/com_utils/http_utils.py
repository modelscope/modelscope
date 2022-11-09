# Copyright (c) 2022 Zhipu.AI
import csv
import traceback
from io import StringIO
from urllib import parse

from flask import Response, jsonify, request, send_file


class APIException(Exception):

    def __init__(self, message):
        super().__init__(message)


class IllegalParamException(APIException):

    def __init__(self, error):
        self.error = error
        super(IllegalParamException, self).__init__(error)


class InputTooLongException(APIException):

    def __init__(self, message, payload=None):
        self.payload = payload
        super().__init__(message)


class CanNotReturnException(APIException):

    def __init__(self, message, payload=None):
        self.payload = payload
        super().__init__(message)


class MongoDBException(APIException):

    def __init__(self, error):
        self.error = error
        super(MongoDBException, self).__init__(error)


class MissParameterException(APIException):

    def __init__(self, error):
        self.error = error
        super(MissParameterException, self).__init__(error)


class HttpUtil:

    @staticmethod
    def http_response(status=0, message='success', data=None, total=False):
        # if status and not isinstance(data, APIException):
        #     sm.send_content(request.url_rule, traceback.format_exc(), request.data)
        if isinstance(data, Exception):
            data = str(data)
        r = {'status': status, 'message': message, 'result': data or []}
        if total and type(data) == list:
            if type(total) == int:
                r['total'] = total
            else:
                r['total'] = len(data)
        return jsonify(r)

    @staticmethod
    def check_param(
            name,
            request,  # noqa
            method=0,
            param_type=None,
            default=None,
            required=True):
        if method == 0:
            param = request.args.get(name)
        else:
            try:
                param = request.json.get(name)
            except Exception as e:  # noqa
                raise IllegalParamException('data format json')

        if param is None:
            if not required:
                return default
            raise IllegalParamException('param {} is required'.format(name))
        else:
            if param_type and type(param) != param_type:
                try:
                    return param_type(param)
                except ValueError:
                    raise IllegalParamException(
                        'param {}: type wrong, not {}'.format(
                            name, param_type))
            else:
                return param

    @staticmethod
    def csv_file_response(data, filename):
        response = Response(HttpUtil.get_csv_stream(data), mimetype='text/csv')
        response.headers[
            'Content-Disposition'] = f'attachment; filename={parse.quote(filename)}.csv'
        return response

    @staticmethod
    def get_csv_stream(data):
        line = StringIO()
        csv_writer = csv.writer(line)
        csv_writer.writerow(['name', 'org', 'position', 'email', 'phone'])
        for p in data:
            csv_writer.writerow(
                [p['name'], p['aff'], p['position'], p['email'], p['phone']])
        res = line.getvalue()
        line.close()
        return res
