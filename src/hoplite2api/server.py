"""Loopback OpenAI-compatible HTTP facade for Hoplite ACP."""

import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import hmac
import json
import os
import re
import subprocess
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .acp import ACPClient, UpstreamError
from .bridge import Bridge, Conflict, completion_base
from .protocol import RequestError, assistant_message, parse_request


UPSTREAM = "https://api.hoplite.sh"


def load_key():
    key = os.environ.get("HOPLITE_API_KEY")
    if key is None:
        try:
            key = subprocess.run(["omp", "token", "hoplite"], check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError("Unable to load Hoplite credential from omp") from None
    if not key or "\n" in key or "\r" in key:
        raise RuntimeError("Hoplite credential is empty or invalid")
    return key


def model_rows(data):
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise UpstreamError("Invalid upstream model roster")
    rows = []
    for model in data["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"]:
            continue
        row = {"id": model["id"], "object": "model", "owned_by": model.get("provider", "hoplite"),
               "input_modalities": ["text"], "output_modalities": ["text"],
               "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}}
        if isinstance(model.get("name"), str):
            row["name"] = model["name"]
        capabilities = model.get("capabilities")
        if isinstance(capabilities, dict):
            for source, target in (("contextWindowTokens", "context_length"), ("maxOutputTokens", "max_output_tokens")):
                if source in capabilities:
                    row[target] = capabilities[source]
            levels = capabilities.get("reasoningLevels")
            if isinstance(levels, list):
                levels = [level for level in levels if isinstance(level, str)]
                if levels:
                    row["supported_reasoning_levels"] = levels
                    config = model.get("runConfig")
                    if not isinstance(config, dict):
                        config = data.get("runConfig")
                    default = config.get("defaultReasoning") if isinstance(config, dict) else None
                    row["default_reasoning_level"] = default if default in levels else levels[0]
        pricing = model.get("pricing")
        if isinstance(pricing, dict):
            cost = {}
            for source, target in (("inputMicrosPerMTok", "input_usd_per_million_tokens"), ("outputMicrosPerMTok", "output_usd_per_million_tokens")):
                if isinstance(pricing.get(source), (int, float)) and not isinstance(pricing[source], bool):
                    cost[target] = pricing[source] / 1_000_000
            if cost:
                row["pricing"] = cost
        rows.append(row)
    return rows


def create_app(project, key, *, base_url=UPSTREAM, client_factory=ACPClient.open, http_client=None, idle_timeout=300, roster_ttl=60):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", project):
        raise ValueError("project must be a safe plaintext project ID")
    if not key:
        raise ValueError("credential must not be empty")
    bridge = Bridge(key, project, base_url, client_factory, idle_timeout)
    http = http_client or httpx.AsyncClient(base_url=base_url, timeout=20)
    roster = {"rows": None, "until": 0.0}
    roster_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await bridge.close()
        await http.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bridge = bridge

    def authorized(request):
        header = request.headers.get("authorization", "")
        scheme, sep, token = header.partition(" ")
        return bool(sep and scheme.lower() == "bearer" and hmac.compare_digest(token.encode(), key.encode()))

    def error(message, status, code=None):
        return JSONResponse({"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error", "code": code}}, status_code=status)

    async def models():
        async with roster_lock:
            now = time.monotonic()
            if roster["rows"] is None or now >= roster["until"]:
                try:
                    response = await http.get("/api/model-providers", headers={"X-Api-Key": key})
                    response.raise_for_status()
                    rows = model_rows(response.json())
                except (httpx.HTTPError, ValueError):
                    raise UpstreamError("Unable to fetch upstream model roster") from None
                roster.update(rows=rows, until=now + roster_ttl)
            return roster["rows"]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(request: Request):
        if not authorized(request):
            return error("Invalid API key", 401)
        try:
            return {"object": "list", "data": await models()}
        except UpstreamError as exc:
            return error(str(exc), 502)

    async def disconnected(request):
        while True:
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.1)

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not authorized(request):
            return error("Invalid API key", 401)
        try:
            body = await request.json()
            parsed = parse_request(body)
            if parsed["messages"][-1]["role"] != "tool":
                if parsed["model"] not in {row["id"] for row in await models()}:
                    return error("Model is not in the authorized upstream roster", 400, "model_not_found")
        except (ValueError, RequestError) as exc:
            return error(str(exc), 400)
        except UpstreamError as exc:
            return error(str(exc), 502)

        session = None
        acquire_task = asyncio.create_task(bridge.acquire(parsed))
        disconnect_task = asyncio.create_task(disconnected(request))
        try:
            done, _ = await asyncio.wait({acquire_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done:
                acquire_task.cancel()
                with suppress(asyncio.CancelledError):
                    session = await acquire_task
                if session:
                    await bridge.close_session(session)
                return error("Client disconnected", 499)
            session = await acquire_task
        except Conflict as exc:
            return error(str(exc), 409)
        except UpstreamError as exc:
            return error(str(exc), 502)
        except RequestError as exc:
            return error(str(exc), 400)
        finally:
            if not acquire_task.done():
                acquire_task.cancel()
                with suppress(asyncio.CancelledError):
                    await acquire_task
            if session is None and acquire_task.done() and not acquire_task.cancelled() and acquire_task.exception() is None:
                await bridge.close_session(acquire_task.result())
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task
        base = completion_base(parsed["model"])

        async def generate():
            text = ""
            committed = False
            try:
                yield ("role", {"role": "assistant"})
                async for event in bridge.events(session):
                    if event["type"] == "text":
                        text += event["text"]
                        yield ("text", event["text"])
                    elif event["type"] == "tool_call":
                        call = bridge.publish_tool(session, event, text)
                        yield ("tool", call)
                        bridge.commit_tool(session)
                        yield ("finish", "tool_calls")
                        committed = True
                    else:
                        if event.get("stop_reason") != "end_turn":
                            raise UpstreamError("Upstream prompt ended without completing the response")
                        if (parsed["tool_choice"] == "required" or isinstance(parsed["tool_choice"], dict)) and not session.had_tool:
                            raise UpstreamError("Upstream finished without making the required local tool call")
                        bridge.commit_done(session, text)
                        yield ("finish", "stop")
                        committed = True
            finally:
                if not committed:
                    await bridge.close_session(session)

        if parsed["stream"]:
            async def stream():
                iterator = generate()
                try:
                    async for kind, value in iterator:
                        if kind == "role":
                            delta, finish = value, None
                        elif kind == "text":
                            delta, finish = {"content": value}, None
                        elif kind == "tool":
                            delta, finish = {"tool_calls": [{"index": 0, **value}]}, None
                        else:
                            delta, finish = {}, value
                        chunk = {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                except (UpstreamError, Conflict) as exc:
                    yield "data: " + json.dumps({"error": {"message": str(exc), "type": "server_error"}}) + "\n\n"
                except Exception:
                    yield "data: " + json.dumps({"error": {"message": "Proxy response failed", "type": "server_error"}}) + "\n\n"
                finally:
                    await iterator.aclose()
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

        async def collect():
            text, call, finish = "", None, None
            iterator = generate()
            try:
                async for kind, value in iterator:
                    if kind == "text":
                        text += value
                    elif kind == "tool":
                        call = value
                    elif kind == "finish":
                        finish = value
                return {**base, "choices": [{"index": 0, "message": assistant_message(text, call), "finish_reason": finish}]}
            finally:
                await iterator.aclose()

        task = asyncio.create_task(collect())
        watcher = asyncio.create_task(disconnected(request))
        try:
            done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                return error("Client disconnected", 499)
            return await task
        except UpstreamError as exc:
            return error(str(exc), 502)
        except Exception:
            return error("Proxy response failed", 500)
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve Hoplite ACP on loopback")
    parser.add_argument("--project", required=True)
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.project):
        parser.error("--project must be a safe plaintext project ID")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = create_app(args.project, load_key())
    except RuntimeError as exc:
        parser.error(str(exc))
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
