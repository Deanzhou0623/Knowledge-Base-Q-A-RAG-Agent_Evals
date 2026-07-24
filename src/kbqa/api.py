from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from kbqa.config import Settings, get_settings
from kbqa.factory import build_service
from kbqa.models import ChatRequest, ChatResponse, HealthResponse, IndexSummary
from kbqa.service import IndexNotReady, QAService
from kbqa.transactions import TransactionFixtureError


def create_app(
    settings: Settings | None = None, service: QAService | None = None
) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or build_service(active_settings)
        app.state.service.load()
        yield

    app = FastAPI(
        title="Knowledge Base Q&A RAG Agent", version="0.1.0", lifespan=lifespan
    )

    def get_service(request: Request) -> QAService:
        return request.app.state.service

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return get_service(request).health()

    @app.post("/index", response_model=IndexSummary)
    def index(request: Request) -> IndexSummary:
        return get_service(request).build_index()

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            return get_service(request).chat(payload)
        except IndexNotReady as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TransactionFixtureError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
