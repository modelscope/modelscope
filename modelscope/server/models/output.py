import datetime
from http import HTTPStatus
from typing import Generic, Optional, Type, TypeVar

import json
from pydantic import BaseModel

ResultType = TypeVar('ResultType')


class ApiResponse(BaseModel, Generic[ResultType]):
    Code: Optional[int] = HTTPStatus.OK
    Success: Optional[bool] = True
    RequestId: Optional[str] = ''
    Message: Optional[str] = 'success'
    Data: Optional[ResultType] = {}
    """
        ResultType (_type_): The response data type.
        Failed: {'Code': 10010101004, 'Message': 'get model info failed, err: unauthorized permission',
                 'RequestId': '', 'Success': False}
        Success: {'Code': 200, 'Data': {}, 'Message': 'success', 'RequestId': '', 'Success': True}



    def set_data(self, data=Type[ResultType]):
        self.Data = data

    def set_message(self, message):
        self.Message = message

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.isoformat() if (isinstance(o, datetime.datetime))
                          else o.__dict__, sort_keys=True, indent=4)
    """
