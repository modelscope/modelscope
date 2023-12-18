from fastapi import APIRouter, Body
from pydantic import BaseModel
from starlette.requests import Request

from modelscope.utils.input_output import \
    pipeline_output_to_service_base64_output  # noqa E125
from modelscope.utils.input_output import call_pipeline_with_json

router = APIRouter()


@router.post('/call')
async def inference(
    request: Request,
    body: BaseModel = Body(examples=[{
        'usage': 'copy body from describe'
    }])):  # noqa E125
    """Inference general interface.

    For image, video, audio etc binary data, need encoded with base64.

    Args:
        request (Request): The request object.
        request_info (ModelScopeRequest): The post body.

    Returns:
        ApiResponse: For binary field, encoded with base64
    """
    pipeline_service = request.app.state.pipeline
    pipeline_info = request.app.state.pipeline_info
    request_json = await request.json()
    result = call_pipeline_with_json(pipeline_info, pipeline_service,
                                     request_json)
    # convert output to json, if binary field, we need encoded.
    output = pipeline_output_to_service_base64_output(
        pipeline_info['task_name'], result)
    return output


@router.get('/describe')
async def describe(request: Request):
    info = {}
    info['schema'] = request.app.state.pipeline_info
    info['sample'] = request.app.state.pipeline_sample
    return info
