"""Process-local ownership of ACP sessions and native tool callbacks."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from .acp import ACPClient, UpstreamError
from .protocol import assistant_message, history_key, prompt_text, registered_tools, text_content


class Conflict(Exception):
    pass


@dataclass(eq=False)
class Session:
    client: object
    request: dict
    history: list
    busy: bool = True
    pending: dict | None = None
    active_key: str | None = None
    timer: asyncio.Task | None = None
    had_tool: bool = False
    closed: bool = False


class Bridge:
    def __init__(self, key, project, base_url="https://api.hoplite.sh", client_factory=ACPClient.open, idle_timeout=300):
        self.key, self.project, self.base_url = key, project, base_url
        self.client_factory, self.idle_timeout = client_factory, idle_timeout
        self.sessions = set()
        self.pending = {}
        self.idle = {}
        self.active_keys = set()

    def key_for(self, request, messages):
        return history_key(request["model"], messages, request["tools"], request["reasoning_effort"])

    async def close_session(self, session):
        if session.closed:
            return
        session.closed = True
        self.sessions.discard(session)
        for mapping in (self.pending, self.idle):
            for key, value in list(mapping.items()):
                if value is session:
                    del mapping[key]
        if session.active_key:
            self.active_keys.discard(session.active_key)
        if session.timer and session.timer is not asyncio.current_task():
            session.timer.cancel()
        await session.client.close()

    async def close(self):
        await asyncio.gather(*(self.close_session(s) for s in list(self.sessions)), return_exceptions=True)

    def schedule_expiry(self, session, delay):
        if session.timer:
            session.timer.cancel()
        async def expire():
            try:
                await asyncio.sleep(max(0, delay))
                await self.close_session(session)
            except asyncio.CancelledError:
                pass
        session.timer = asyncio.create_task(expire())

    async def acquire(self, request):
        messages = request["messages"]
        if messages[-1]["role"] == "tool":
            call_id = messages[-1]["tool_call_id"]
            session = self.pending.get(call_id)
            if not session or session.closed or session.busy:
                raise Conflict("Unknown, expired, or already claimed tool call")
            if self.key_for(request, messages[:-1]) != self.key_for(session.request, session.history):
                raise Conflict("Tool result does not match the pending conversation")
            if registered_tools(request) != registered_tools(session.request):
                raise Conflict("Registered tools changed during the pending call")
            for field in ("tool_choice", "parallel_tool_calls", "max_tokens", "max_completion_tokens"):
                if request.get(field) != session.request.get(field):
                    raise Conflict("Prompt controls changed during the pending call")
            event = session.pending
            if asyncio.get_running_loop().time() >= event["deadline"]:
                await self.close_session(session)
                raise Conflict("Tool call expired")
            result = text_content(messages[-1].get("content"))
            payload = {"content": [{"type": "text", "text": result}], "isError": False}
            if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) > session.client.max_result_bytes:
                raise Conflict("Tool result exceeds upstream limit")
            session.busy = True
            self.pending.pop(call_id, None)
            if session.timer:
                session.timer.cancel()
                session.timer = None
            try:
                await session.client.reply_tool(event["rpc_id"], result)
            except BaseException:
                await self.close_session(session)
                raise
            session.pending = None
            session.request = request
            session.history = list(messages)
            return session

        # Only a user turn can follow an idle assistant response.
        match = None
        if messages[-1]["role"] == "user":
            for index in range(len(messages) - 1, 0, -1):
                candidate = self.key_for(request, messages[:index])
                if candidate in self.active_keys:
                    raise Conflict("Conversation already has an active request")
                session = self.idle.get(candidate)
                if session is not None:
                    if registered_tools(request) != registered_tools(session.request):
                        continue
                    match = (candidate, index, session)
                    break
        if match:
            key, index, session = match
            del self.idle[key]
            self.active_keys.add(key)
            session.active_key = key
            session.busy = True
            if session.timer:
                session.timer.cancel()
                session.timer = None
            session.had_tool = False
            session.request, session.history = request, list(messages)
            try:
                await session.client.start_prompt(prompt_text(request, messages[index:]))
            except BaseException:
                await self.close_session(session)
                raise
            return session

        key = self.key_for(request, messages)
        if key in self.active_keys:
            raise Conflict("Conversation already has an active request")
        self.active_keys.add(key)
        session = None
        try:
            client = await self.client_factory(self.base_url, self.key, self.project)
            session = Session(client, request, list(messages), active_key=key)
            self.sessions.add(session)
            await client.set_model(request["model"], request["reasoning_effort"])
            await client.register_tools(registered_tools(request))
            await client.start_prompt(prompt_text(request))
            return session
        except BaseException:
            if session is None:
                self.active_keys.discard(key)
            else:
                await self.close_session(session)
            raise

    def publish_tool(self, session, event, text):
        event = dict(event)
        timeout = session.client.tool_timeout
        event["deadline"] -= max(0, timeout - 60)
        call_id = event["call_id"]
        if not call_id or call_id in self.pending or asyncio.get_running_loop().time() >= event["deadline"]:
            raise UpstreamError("Invalid, duplicate, or expired upstream tool call")
        call = {"id": call_id, "type": "function", "function": {"name": event["name"], "arguments": json.dumps(event["arguments"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)}}
        session.history.append(assistant_message(text, call))
        session.pending = event
        session.had_tool = True
        return call

    def commit_tool(self, session):
        session.busy = False
        self.pending[session.pending["call_id"]] = session
        self.schedule_expiry(session, session.pending["deadline"] - asyncio.get_running_loop().time())

    def commit_done(self, session, text):
        session.history.append(assistant_message(text))
        key = self.key_for(session.request, session.history)
        previous = self.idle.get(key)
        session.busy = False
        self.idle[key] = session
        self.schedule_expiry(session, self.idle_timeout)
        if session.active_key:
            self.active_keys.discard(session.active_key)
            session.active_key = None
        if previous is not None and previous is not session:
            asyncio.create_task(self.close_session(previous))

    async def events(self, session):
        while True:
            event = await session.client.events.get()
            kind = event.get("type")
            if kind == "text":
                yield event
            elif kind in ("tool_call", "done"):
                yield event
                return
            elif kind == "error":
                raise event["error"]


def completion_id():
    return "chatcmpl-" + uuid.uuid4().hex


def completion_base(model):
    return {"id": completion_id(), "object": "chat.completion", "created": int(time.time()), "model": model}
