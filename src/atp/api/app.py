"""FastAPI application factory for local frontend development."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .routes import router
from .serialization import _finite
from .services import ConfigurationError
from ..scenarios.spec import ScenarioError


def create_app() -> FastAPI:
    app = FastAPI(
        title="Aircraft Trajectory Planner API",
        description="Research/educational trajectory-planning interface; not for operational use.",
        version="0.1.0",
    )
    origins = [
        origin.strip()
        for origin in os.getenv("ATP_API_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "request validation failed",
                    "details": _finite(exc.errors()),
                }
            },
        )

    @app.exception_handler(KeyError)
    async def not_found_handler(_, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "SCENARIO_NOT_FOUND",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    @app.exception_handler(ConfigurationError)
    async def configuration_error_handler(_, exc: ConfigurationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_CONFIGURATION",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    @app.exception_handler(ScenarioError)
    async def scenario_configuration_error_handler(_, exc: ScenarioError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_CONFIGURATION",
                    "message": str(exc),
                    "details": {},
                }
            },
        )

    app.include_router(router)
    return app


app = create_app()
