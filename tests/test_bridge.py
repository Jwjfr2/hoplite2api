import asyncio
import copy
import json
import unittest

import httpx

from hoplite2api.bridge import Bridge, Conflict
from hoplite2api.protocol import parse_request
from hoplite2api.server import create_app


TOOL = {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}


class FakeACP:
    def __init__(self):
        self.events = asyncio.Queue()
        self.max_result_bytes = 1000
        self.tool_timeout = 60
        self.started = []
        self.replies = []
        self.closed = False

    async def set_model(self, model, reasoning):
        pass

    async def register_tools(self, tools):
        self.tools = tools

    async def start_prompt(self, text):
        self.started.append(text)

    async def reply_tool(self, rpc_id, text):
        self.replies.append((rpc_id, text))

    async def close(self):
        self.closed = True


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clients = []
        async def factory(*args):
            client = FakeACP()
            self.clients.append(client)
            return client
        self.bridge = Bridge("key", "project", client_factory=factory, idle_timeout=1)
        self.body = {"model": "m", "messages": [{"role": "user", "content": "read it"}], "tools": [TOOL]}


    async def asyncTearDown(self):
        await self.bridge.close()

    async def pending(self, timeout=1):
        session = await self.bridge.acquire(parse_request(self.body))
        event = {"type": "tool_call", "rpc_id": 7, "call_id": "native-1", "name": "read", "arguments": {"path": "x"}, "deadline": asyncio.get_running_loop().time() + timeout}
        call = self.bridge.publish_tool(session, event, "")
        self.bridge.commit_tool(session)
        result = copy.deepcopy(self.body)
        result["messages"] += [{"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "tool", "tool_call_id": "native-1", "content": "file contents"}]
        return session, result

    async def test_native_result_resumes_original_prompt(self):
        session, result = await self.pending()
        resumed = await self.bridge.acquire(parse_request(result))
        self.assertIs(resumed, session)
        self.assertEqual(session.client.replies, [(7, "file contents")])
        self.assertEqual(len(session.client.started), 1)
        await session.client.events.put({"type": "text", "text": "answer"})
        await session.client.events.put({"type": "done", "stop_reason": "end_turn"})
        self.assertEqual([e["type"] async for e in self.bridge.events(session)], ["text", "done"])
        self.bridge.commit_done(session, "answer")
        turn = copy.deepcopy(result)
        turn["messages"] += [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "again"}]
        self.assertIs(await self.bridge.acquire(parse_request(turn)), session)
        self.assertEqual(len(session.client.started), 2)
        self.assertNotIn("file contents", session.client.started[-1])

    async def test_forged_history_rejected_without_new_run(self):
        session, result = await self.pending()
        result["messages"][0]["content"] = "forged"
        with self.assertRaises(Conflict):
            await self.bridge.acquire(parse_request(result))
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(session.client.replies, [])
        self.assertFalse(session.client.closed)

    async def test_expired_callback_closes_active_run(self):
        session, result = await self.pending(timeout=0.01)
        await asyncio.sleep(0.03)
        self.assertTrue(session.client.closed)
        with self.assertRaises(Conflict):
            await self.bridge.acquire(parse_request(result))
        self.assertEqual(len(self.clients), 1)

    async def test_concurrent_initial_request_does_not_open_another_session(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        async def factory(*args):
            entered.set()
            await release.wait()
            client = FakeACP()
            self.clients.append(client)
            return client
        self.bridge.client_factory = factory
        request = parse_request(self.body)
        first = asyncio.create_task(self.bridge.acquire(request))
        try:
            await entered.wait()
            with self.assertRaises(Conflict):
                await self.bridge.acquire(request)
            release.set()
            await first
            self.assertEqual(len(self.clients), 1)
        finally:
            if not first.done():
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first

    async def test_required_tool_missing_returns_error_in_both_modes(self):
        async def factory(*args):
            client = FakeACP()
            await client.events.put({"type": "done", "stop_reason": "end_turn"})
            self.clients.append(client)
            return client
        http = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": [{"id": "m"}]})))
        app = create_app("project", "key", client_factory=factory, http_client=http)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as local:
            try:
                for choice in ("required", {"type": "function", "function": {"name": "read"}}):
                    for stream in (False, True):
                        response = await local.post("/v1/chat/completions", headers={"Authorization": "Bearer key"},
                                                    json={**self.body, "tool_choice": choice, "stream": stream})
                        if stream:
                            frames = [json.loads(line[6:]) for line in response.text.splitlines()
                                      if line.startswith("data: ") and line != "data: [DONE]"]
                            self.assertEqual(frames[-1]["error"]["type"], "server_error")
                            self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
                            self.assertFalse(any(choice.get("finish_reason") == "stop"
                                                 for frame in frames for choice in frame.get("choices", [])))
                        else:
                            self.assertEqual(response.status_code, 502)
                            self.assertEqual(response.json()["error"]["type"], "server_error")
            finally:
                await app.state.bridge.close()
                await http.aclose()

    async def test_disconnected_http_stream_closes_run(self):
        async def factory(*args):
            client = FakeACP()
            self.clients.append(client)
            return client
        http = httpx.AsyncClient(base_url="https://example.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": [{"id": "m"}]})))
        app = create_app("project", "key", client_factory=factory, http_client=http)
        body = {**self.body, "stream": True}
        disconnected = asyncio.Event()
        response_started = asyncio.Event()
        first = True
        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}
        async def send(message):
            if message["type"] == "http.response.body":
                response_started.set()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "POST",
                 "scheme": "http", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                 "query_string": b"", "headers": [(b"authorization", b"Bearer key"), (b"content-type", b"application/json")],
                 "server": ("127.0.0.1", 8091), "client": ("127.0.0.1", 1234)}
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(response_started.wait(), 1)
            disconnected.set()
            await asyncio.wait_for(task, 1)
            self.assertTrue(self.clients[0].closed)
        finally:
            if not task.done():
                task.cancel()
            await app.state.bridge.close()
            await http.aclose()
