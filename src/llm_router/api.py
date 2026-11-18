"""HTTP API: POST /v1/chat, GET /health, GET /metrics."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator

from .claude import ClaudeNotConfigured
from .gateway import Gateway, RoutingError, build_gateway


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    system: str | None = Field(default=None, max_length=50_000)
    messages: list[Message] = Field(min_length=1, max_length=200)
    max_tokens: int | None = Field(default=None, ge=1, le=16_000)
    strategy: Literal["heuristic", "knn", "cascade", "always-small", "always-large"] | None = None
    cache: bool = True

    @field_validator("messages")
    @classmethod
    def last_message_from_user(cls, messages: list[Message]) -> list[Message]:
        if messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        return messages


def create_app(factory: Callable[[], Gateway] = build_gateway) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway = factory()
        yield

    app = FastAPI(
        title="llm-router",
        summary="Semantic cache and cost-aware model routing in front of Claude",
        lifespan=lifespan,
    )

    def gateway(request: Request) -> Gateway:
        return request.app.state.gateway

    @app.get("/health")
    def health(request: Request) -> dict:
        return gateway(request).health()

    @app.get("/metrics")
    def metrics(request: Request) -> Response:
        m = gateway(request).metrics
        return Response(m.render(), media_type=m.content_type)

    @app.post("/v1/chat")
    def chat(request: Request, body: ChatRequest) -> dict:
        g = gateway(request)
        try:
            return g.chat(
                messages=[m.model_dump() for m in body.messages],
                system=body.system,
                max_tokens=body.max_tokens,
                strategy=body.strategy,
                use_cache=body.cache,
            )
        except RoutingError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            g.metrics.errors.labels(type(exc).__name__).inc()
            raise _http_error(exc) from exc

    return app


def _http_error(exc: Exception) -> Exception:
    if isinstance(exc, ClaudeNotConfigured | anthropic.AuthenticationError):
        return HTTPException(503, "Claude credentials are missing or invalid (ANTHROPIC_API_KEY).")
    if isinstance(exc, anthropic.RateLimitError):
        return HTTPException(429, "Claude rate limit reached, retry later.")
    if isinstance(exc, anthropic.BadRequestError):
        return HTTPException(400, f"Claude rejected the request: {exc.message}")
    if isinstance(exc, anthropic.APIStatusError):
        return HTTPException(502, f"Claude API error {exc.status_code}.")
    if isinstance(exc, anthropic.APIConnectionError):
        return HTTPException(502, "Could not reach the Claude API.")
    if isinstance(exc, anthropic.AnthropicError):
        return HTTPException(503, "Claude is not configured (set ANTHROPIC_API_KEY).")
    return exc


app = create_app()
