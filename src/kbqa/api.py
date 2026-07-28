from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from kbqa.config import Settings, get_settings
from kbqa.factory import build_service
from kbqa.models import (
    BackendStatus,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexSummary,
)
from kbqa.service import IndexNotReady, QAService
from kbqa.transactions import TransactionFixtureError


def create_app(
    settings: Settings | None = None, service: QAService | None = None
) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            # An injected service pins the server to that one backend.
            services = {service.retriever.backend: service}
        else:
            services = {}
            for name in ("bm25", "vector"):
                try:
                    services[name] = build_service(
                        active_settings.model_copy(
                            update={"retrieval_backend": name}
                        )
                    )
                except Exception:
                    # A backend that cannot be constructed (for example a
                    # missing embedding key) is simply not offered; it must not
                    # take down the backend that does work.
                    continue
        for built in services.values():
            built.load()
        app.state.services = services
        app.state.service = services.get(
            active_settings.retrieval_backend, next(iter(services.values()))
        )
        yield

    app = FastAPI(
        title="Knowledge Base Q&A RAG Agent", version="0.1.0", lifespan=lifespan
    )
    ui_path = Path(__file__).with_name("ui")
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")

    def get_service(request: Request, backend: str | None = None) -> QAService:
        if backend is None:
            return request.app.state.service
        services = request.app.state.services
        if backend not in services:
            raise HTTPException(
                status_code=400,
                detail=f"Backend {backend!r} is not configured on this server.",
            )
        return services[backend]

    def backend_statuses(request: Request) -> list[BackendStatus]:
        return [
            BackendStatus(backend=name, index_loaded=built.retriever.loaded)
            for name, built in request.app.state.services.items()
        ]

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request, backend: str | None = None) -> HealthResponse:
        response = get_service(request, backend).health()
        return response.model_copy(update={"backends": backend_statuses(request)})

    @app.post("/index", response_model=IndexSummary)
    def index(request: Request, backend: str | None = None) -> IndexSummary:
        return get_service(request, backend).build_index()

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            return get_service(request, payload.backend).chat(payload)
        except IndexNotReady as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TransactionFixtureError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
