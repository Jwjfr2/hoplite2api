import copy
import unittest

from hoplite2api.protocol import (
    RequestError, assistant_message, history_key, parse_request,
    registered_tools, text_content,
)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tool = {"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
        self.body = {"model": "model", "messages": [{"role": "user", "content": "hello"}], "tools": [self.tool]}
        self.call = {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}


    def test_tool_choice(self):
        second = {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}}
        self.body["tools"].append(second)
        for choice, expected in [("auto", ["read_file", "other"]), ("required", ["read_file", "other"]), ("none", []), ({"type": "function", "function": {"name": "other"}}, ["other"])]:
            with self.subTest(choice=choice):
                self.body["tool_choice"] = choice
                request = parse_request(self.body)
                self.assertEqual([tool["name"] for tool in registered_tools(request)], expected)
        self.body["tool_choice"] = {"type": "function", "function": {"name": "missing"}}
        with self.assertRaises(RequestError):
            parse_request(self.body)

    def test_text_content_and_rejected_media(self):
        self.assertEqual(text_content([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]), "ab")
        self.assertEqual(text_content(None), "")
        for content in ([{"type": "image_url", "image_url": {"url": "x"}}], [{"type": "input_audio", "input_audio": {}}], [{"type": "text", "text": 1}], 4):
            with self.subTest(content=content), self.assertRaises(RequestError):
                text_content(content)

    def test_unsupported_request_fields_and_controls(self):
        invalid = [{"temperature": 0}, {"top_p": 1}, {"seed": 0}, {"stop": []}, {"logit_bias": {}}, {"presence_penalty": 0}, {"frequency_penalty": 0}, {"n": True}, {"n": 2}, {"max_tokens": 0}, {"max_completion_tokens": True}, {"modalities": ["audio"]}, {"response_format": {"type": "json_object"}}, {"stream_options": {"include_usage": 1}}, {"stream": 1}, {"store": True}, {"unknown": 1}]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(RequestError):
                parse_request(self.body | fields)
        self.assertEqual(parse_request(self.body | {"temperature": None, "modalities": ["text"], "response_format": {"type": "text"}})["model"], "model")

    def test_tools_and_message_validation(self):
        invalid_tools = [[self.tool, self.tool], [self.tool] * 65, [{"type": "function", "function": {"name": "bad name", "parameters": {"type": "object"}}}], [{"type": "function", "function": {"name": "ok", "parameters": {"type": "string"}}}], [{"type": "function", "function": {"name": "ok", "parameters": {"type": "object"}, "strict": True}}], [{"type": "other"}]]
        for tools in invalid_tools:
            with self.subTest(tools=tools[:2]), self.assertRaises(RequestError):
                parse_request(self.body | {"tools": tools})
        bad_messages = [[{"role": "tool", "content": "x"}], [{"role": "alien", "content": "x"}], [{"role": "assistant", "tool_calls": [self.call | {"id": ""}]}], [{"role": "assistant", "tool_calls": [self.call | {"function": {"name": "read_file", "arguments": "[]"}}]}]]
        for messages in bad_messages:
            with self.subTest(messages=messages), self.assertRaises(RequestError):
                parse_request(self.body | {"messages": messages})

    def test_history_normalizes_replayed_tool_calls_and_text(self):
        emitted = [{"role": "user", "content": "hello"}, assistant_message("", self.call), {"role": "tool", "tool_call_id": "call-1", "content": "result"}]
        replay = copy.deepcopy(emitted)
        replay[0]["content"] = [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]
        del replay[1]["content"]
        replay[1]["reasoning_content"] = "ignored"
        replay[1]["usage"] = {"total_tokens": 3}
        replay[1]["tool_calls"][0]["function"]["arguments"] = '{ "path": "a" }'
        self.assertEqual(history_key("model", emitted, [self.tool]), history_key("model", replay, [self.tool]))
        parse_request(self.body | {"messages": replay})
        changed = copy.deepcopy(replay)
        changed[2]["content"] = "different"
        self.assertNotEqual(history_key("model", emitted, [self.tool]), history_key("model", changed, [self.tool]))
        self.assertNotEqual(history_key("model", emitted, [self.tool]), history_key("model", emitted, [self.tool], "high"))
        self.assertNotEqual(history_key("model", emitted, [self.tool]), history_key("other", emitted, [self.tool]))
        self.assertNotEqual(history_key("model", emitted, [self.tool]), history_key("model", emitted, []))



if __name__ == "__main__":
    unittest.main()
