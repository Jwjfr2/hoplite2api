import asyncio
import json
import unittest
from unittest.mock import patch

from websockets.asyncio.server import serve

from hoplite2api.acp import ACPClient, UpstreamError


class Peer:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.socket = None
        self.connected = asyncio.Event()

    async def handler(self, socket):
        self.socket = socket
        self.connected.set()
        async for frame in socket:
            message = json.loads(frame)
            method = message.get("method")
            if method == "initialize":
                await self.respond(message, {"protocolVersion": 1})
            elif method == "authenticate":
                await self.respond(message, {})
            elif method == "session/new":
                await self.respond(message, {"sessionId": "session", "configOptions": [
                    {"id": "model", "options": [{"value": "test-model"}]},
                    {"id": "reasoning", "options": [{"value": "none"}]},
                ]})
            elif method == "session/set_config_option":
                await self.respond(message, {"configOptions": [
                    {"id": "model", "options": [{"value": "test-model"}]},
                    {"id": "reasoning", "options": [{"value": "none"}]},
                ]})
            elif method == "_hoplite/client_tools/register":
                servers = message["params"]["servers"]
                accepted = [{"name": s["name"], "tools": [t["name"] for t in s["tools"]], "hiddenWriteTools": 0} for s in servers]
                await self.respond(message, {"servers": accepted, "dropped": [], "limits": {"callTimeoutMs": 5000, "maxResultBytes": 256}})
            else:
                await self.messages.put(message)

    async def send(self, message):
        await self.socket.send(json.dumps(message))

    async def respond(self, request, result):
        await self.send({"jsonrpc": "2.0", "id": request["id"], "result": result})

    async def next(self):
        return await asyncio.wait_for(self.messages.get(), 1)


class ACPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.peer = Peer()
        self.server = await serve(self.peer.handler, "127.0.0.1", 0)
        port = self.server.sockets[0].getsockname()[1]
        self.client = await ACPClient.open(f"http://127.0.0.1:{port}", "test-key", "project")

    async def asyncTearDown(self):
        await self.client.close()
        self.server.close()
        await self.server.wait_closed()

    async def event(self):
        return await asyncio.wait_for(self.client.events.get(), 1)

    async def test_control_request_timeout_cleans_up_and_ignores_late_reply(self):
        with patch("hoplite2api.acp._REQUEST_TIMEOUT", 0.05):
            request = asyncio.create_task(self.client._request("test/no_reply", {}))
            upstream_request = await self.peer.next()
            with self.assertRaises(UpstreamError) as raised:
                await request
        self.assertEqual(raised.exception.code, "timeout")
        await self.peer.respond(upstream_request, {})
        await self.client.start_prompt("still connected")
        prompt = await self.peer.next()
        await self.peer.respond(prompt, {"stopReason": "end_turn"})
        self.assertEqual((await self.event())["type"], "done")

    async def test_oversized_prompt_and_invalid_tool_are_rejected(self):
        with self.assertRaises(UpstreamError):
            await self.client.register_tools([{"name": "x" * 65, "inputSchema": {"type": "object"}}])
        with self.assertRaises(UpstreamError) as raised:
            await self.client.start_prompt("x" * (1024 * 1024))
        self.assertEqual(raised.exception.code, "body_too_large")
        await self.client.start_prompt("small")
        prompt = await self.peer.next()
        await self.peer.respond(prompt, {"stopReason": "end_turn"})
        self.assertEqual((await self.event())["type"], "done")

    async def test_local_tool_result_resumes_same_prompt(self):
        await self.client.set_model("test-model", "off")
        await self.client.register_tools([{"name": "read", "inputSchema": {"type": "object"}}])
        await self.client.start_prompt("first")
        prompt = await self.peer.next()
        await self.peer.send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "session", "update": {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "private"}}}})
        await self.peer.send({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "session", "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "before"}}}})
        await self.peer.send({"jsonrpc": "2.0", "id": "tool-rpc", "method": "_hoplite/client_tools/call", "params": {"sessionId": "session", "callId": "call-1", "server": "omp", "tool": "read", "arguments": {"path": "x"}}})
        self.assertEqual(await self.event(), {"type": "text", "text": "before"})
        call = await self.event()
        self.assertEqual((call["type"], call["call_id"], call["arguments"]), ("tool_call", "call-1", {"path": "x"}))
        self.assertGreater(call["deadline"], asyncio.get_running_loop().time())
        with self.assertRaises(UpstreamError):
            await self.client.start_prompt("not while waiting")
        with self.assertRaises(UpstreamError):
            await self.client.reply_tool("tool-rpc", "x" * 300)
        await self.client.reply_tool("tool-rpc", "file contents")
        reply = await self.peer.next()
        self.assertEqual(reply["id"], "tool-rpc")
        self.assertEqual(reply["result"], {"content": [{"type": "text", "text": "file contents"}], "isError": False})
        await self.peer.respond(prompt, {"stopReason": "end_turn"})
        self.assertEqual(await self.event(), {"type": "done", "stop_reason": "end_turn"})
        await self.client.start_prompt("second")
        second = await self.peer.next()
        await self.peer.respond(second, {"stopReason": "end_turn"})
        self.assertEqual((await self.event())["type"], "done")

    async def test_disconnect_propagates_provider_and_transport_errors(self):
        await self.client.start_prompt("first")
        prompt = await self.peer.next()
        await self.peer.send({"jsonrpc": "2.0", "id": prompt["id"], "error": {"code": -32000, "message": "repository unavailable", "data": {"code": "project_repository_unavailable"}}})
        event = await self.event()
        self.assertEqual(event["type"], "error")
        self.assertEqual(event["error"].code, "project_repository_unavailable")
        self.assertIn("repository unavailable", str(event["error"]))
        await self.client.start_prompt("second")
        await self.peer.next()
        await self.peer.socket.close()
        self.assertEqual((await self.event())["type"], "error")
        with self.assertRaises(UpstreamError):
            await self.client.start_prompt("third")

    async def test_close_cancels_prompt_and_resolves_server_requests(self):
        await self.client.register_tools([{"name": "read", "inputSchema": {"type": "object"}}])
        await self.client.start_prompt("first")
        prompt = await self.peer.next()
        await self.peer.send({"jsonrpc": "2.0", "id": "pending", "method": "_hoplite/client_tools/call", "params": {"sessionId": "session", "callId": "call", "server": "omp", "tool": "read", "arguments": {}}})
        self.assertEqual((await self.event())["type"], "tool_call")
        closing = asyncio.create_task(self.client.close())
        cancel = await self.peer.next()
        rejection = await self.peer.next()
        self.assertEqual(cancel["method"], "session/cancel")
        self.assertEqual(rejection["id"], "pending")
        self.assertIn("error", rejection)
        await self.peer.respond(prompt, {"stopReason": "cancelled"})
        await asyncio.wait_for(closing, 1)
        with self.assertRaises(UpstreamError):
            await self.client.reply_tool("pending", "late")

    async def test_permission_and_unknown_requests_are_answered(self):
        await self.client.start_prompt("first")
        prompt = await self.peer.next()
        await self.peer.send({"jsonrpc": "2.0", "id": "permission", "method": "session/request_permission", "params": {}})
        reply = await self.peer.next()
        self.assertIn("error", reply)
        self.assertEqual((await self.event())["error"].code, "permission_required")
        await self.peer.send({"jsonrpc": "2.0", "id": "unknown", "method": "unknown", "params": {}})
        self.assertEqual((await self.peer.next())["error"]["code"], -32601)
        await self.peer.respond(prompt, {"stopReason": "cancelled"})
        await self.event()
