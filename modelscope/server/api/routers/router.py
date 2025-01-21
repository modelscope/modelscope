from fastapi import APIRouter

from modelscope.server.api.routers import health, model_router

api_router = APIRouter()
api_router.include_router(model_router.router, tags=['prediction'], prefix='')
api_router.include_router(health.router, tags=['health'], prefix='/health')
