from typing import Callable

from fastapi import FastAPI

from modelscope.utils.input_output import (  # yapf: disable
    create_pipeline, get_pipeline_information_by_pipeline,
    get_task_input_examples, get_task_schemas)
from modelscope.utils.logger import get_logger

# control the model start stop

logger = get_logger()


def _startup_model(app: FastAPI) -> None:
    logger.info('download model and create pipeline')
    app.state.pipeline = create_pipeline(app.state.args.model_id,
                                         app.state.args.revision,
                                         app.state.args.llm_first)
    info = {}
    info['task_name'] = app.state.pipeline.group_key
    info['schema'] = get_task_schemas(app.state.pipeline.group_key)
    app.state.pipeline_info = info
    app.state.pipeline_sample = get_task_input_examples(
        app.state.pipeline.group_key)
    logger.info('pipeline created.')


def _shutdown_model(app: FastAPI) -> None:
    app.state.pipeline = None
    logger.info('shutdown model service')


def start_app_handler(app: FastAPI) -> Callable:

    def startup() -> None:
        _startup_model(app)

    return startup


def stop_app_handler(app: FastAPI) -> Callable:

    def shutdown() -> None:
        _shutdown_model(app)

    return shutdown
