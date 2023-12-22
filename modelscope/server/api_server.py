import argparse

import uvicorn
from fastapi import FastAPI

from modelscope.server.api.routers.router import api_router
from modelscope.server.core.event_handlers import (start_app_handler,
                                                   stop_app_handler)


def get_app(args) -> FastAPI:
    app = FastAPI(
        title='modelscope_server',
        version='0.1',
        debug=True,
        swagger_ui_parameters={'tryItOutEnabled': True})
    app.state.args = args
    app.include_router(api_router)

    app.add_event_handler('startup', start_app_handler(app))
    app.add_event_handler('shutdown', stop_app_handler(app))
    return app


def add_server_args(parser):
    parser.add_argument(
        '--model_id', required=True, type=str, help='The target model id')
    parser.add_argument(
        '--revision', required=True, type=str, help='Model revision')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--debug', default='debug', help='Set debug level.')
    parser.add_argument(
        '--llm_first',
        type=bool,
        default=True,
        help='Use LLMPipeline first for llm models.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('modelscope_server')
    add_server_args(parser)
    args = parser.parse_args()
    app = get_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
