from faulthandler import disable
from http import HTTPStatus
from typing import Any, Dict

from fastapi import APIRouter

from modelscope.server.models.output import ApiResponse

router = APIRouter()


@router.get('', response_model=ApiResponse[Dict], status_code=200)
def health() -> Any:
    return ApiResponse[Dict](Data={}, Code=HTTPStatus.OK, Success=True)
