"""
Build the Noesis gateway server.

Run the real server (OpenRouter-backed) from the repo root with:

> uvicorn research.Noesis.gateway_app:build_default_app --factory --port 8000

For setup and the overall design see `research/Noesis/docs/onboarding.md` and
`research/Noesis/docs/gateway.README.md`.

Import as:

import research.Noesis.gateway_app as rnogaapp
"""

import contextlib
import logging
import os
from typing import AsyncIterator, Mapping, Optional

import fastapi
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import helpers.hdbg as hdbg
import research.Noesis.gateway_api as rnogaapi
import research.Noesis.gateway_faults as rnogafau
import research.Noesis.gateway_openrouter as rnogaope
import research.Noesis.gateway_providers as rnogapro
import research.Noesis.gateway_routing as rnogarou
import research.Noesis.noesis_db as rnonodb
import research.Noesis.noesis_seed as rnonosee
import research.Noesis.noesis_settings as rnonoset

_LOG = logging.getLogger(__name__)


def create_app(
    settings: rnonoset.Settings,
    provider: rnogapro.Provider,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> fastapi.FastAPI:
    """
    Build the gateway app.

    On startup: connect to Postgres, apply migrations, load seed data.
    On shutdown: close the pool and the provider.

    :param settings: gateway configuration
    :param provider: real (`OpenRouterProvider`) or fake provider
    :param environ: where seed API-key env vars are read
        - Default: `research/Noesis/.env` overlaid with `os.environ`
    :return: the app, ready for `uvicorn`
    """
    if environ is None:
        dotenv_path = os.path.join(rnonoset.NOESIS_DIR, ".env")
        environ = {**rnonoset.read_dotenv(dotenv_path), **os.environ}

    @contextlib.asynccontextmanager
    async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[None]:
        pool = await rnonodb.connect(settings.database_url)
        try:
            applied = await rnonodb.apply_migrations(pool)
            seeded = await rnonosee.load_seed(pool, settings, environ)
            _LOG.info("Startup: migrations=%s seed=%s", applied, seeded)
            app.state.pool = pool
            app.state.provider = provider
            app.state.settings = settings
            yield
        finally:
            # Release the DB and HTTP connections even if startup failed.
            await pool.close()
            aclose = getattr(provider, "aclose", None)
            if aclose is not None:
                await aclose()

    app = fastapi.FastAPI(title="Noesis gateway", lifespan=lifespan)
    # Demo fault switches (in memory; off until an operator sets one).
    app.state.faults = rnogafau.FaultRegistry()

    @app.exception_handler(rnogarou.RoutingError)
    async def _routing_error(
        request: fastapi.Request, exc: rnogarou.RoutingError
    ) -> JSONResponse:
        # One place turns auth/contract/admin errors into OpenAI-style errors,
        # like `platform_api.create_app()` does for `AssertionError`.
        _ = request
        return rnogaapi.error_response(exc.http_status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _bad_request(
        request: fastapi.Request, exc: RequestValidationError
    ) -> JSONResponse:
        # OpenAI-style 400 instead of FastAPI's default 422.
        _ = request
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return rnogaapi.error_response(
            400,
            "invalid_request",
            f"{where}: {first.get('msg', 'invalid request')}",
        )

    app.include_router(rnogaapi.build_router())
    return app


def build_default_app() -> fastapi.FastAPI:
    """
    Build the real server: settings from `.env`, OpenRouter as the provider.

    :return: the app
    """
    logging.basicConfig(level=logging.INFO)
    settings = rnonoset.load_settings()
    hdbg.dassert_ne(
        settings.openrouter_api_key,
        "",
        "OPENROUTER_API_KEY is required to run the gateway; see env.example",
    )
    provider = rnogaope.OpenRouterProvider(settings.openrouter_api_key)
    app = create_app(settings, provider)
    return app
