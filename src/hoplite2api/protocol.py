import hashlib
import json
import re


class RequestError(Exception):
    pass


def text_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise RequestError("Only text content parts are supported")
            if set(part) - {"type", "text"}:
                raise RequestError("Unsupported text content part fields")
            parts.append(part["text"])
        return "".join(parts)
    raise RequestError("Content must be text, null, or an array of text parts")


def _object_arguments(value):
    if not isinstance(value, str):
        raise RequestError("Tool call arguments must be a JSON object string")
    try:
        result = json.loads(value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError):
        raise RequestError("Tool call arguments must be a JSON object string") from None
    if not isinstance(result, dict):
        raise RequestError("Tool call arguments must be a JSON object string")
    return result


def _name(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value) is not None


def parse_request(body):
    if not isinstance(body, dict):
        raise RequestError("Request must be an object")
    allowed = {"model", "messages", "tools", "stream", "stream_options", "reasoning_effort", "tool_choice", "parallel_tool_calls", "n", "modalities", "response_format", "temperature", "top_p", "seed", "logit_bias", "stop", "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens", "store", "user", "metadata", "service_tier"}
    unknown = set(body) - allowed
    if unknown:
        raise RequestError(f"Unsupported request field: {sorted(unknown)[0]}")
    for field in ("temperature", "top_p", "seed", "logit_bias", "stop", "presence_penalty", "frequency_penalty"):
        if body.get(field) is not None:
            raise RequestError(f"{field} is not supported by the upstream")
    if body.get("store") not in (None, False) or ("store" in body and body["store"] is not None and not isinstance(body["store"], bool)):
        raise RequestError("Only store=false is supported")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RequestError("model must be a nonempty string")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise RequestError("Only n=1 is supported")
    if body.get("modalities") is not None and body["modalities"] != ["text"]:
        raise RequestError("Only text modality is supported")
    if body.get("response_format") is not None and body["response_format"] != {"type": "text"}:
        raise RequestError("Only text response_format is supported")
    for field in ("max_tokens", "max_completion_tokens"):
        if body.get(field) is not None and (type(body[field]) is not int or body[field] <= 0):
            raise RequestError(f"{field} must be a positive integer")
    for field in ("stream", "parallel_tool_calls"):
        if field in body and not isinstance(body[field], bool):
            raise RequestError(f"{field} must be a boolean")
    options = body.get("stream_options")
    if options is not None and (not isinstance(options, dict) or set(options) - {"include_usage"} or ("include_usage" in options and not isinstance(options["include_usage"], bool))):
        raise RequestError("stream_options only supports boolean include_usage")
    reasoning = body.get("reasoning_effort")
    if reasoning is not None and (not isinstance(reasoning, str) or not reasoning):
        raise RequestError("reasoning_effort must be a nonempty string")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty array")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError("Invalid message role")
        role = message["role"]
        fields = {"role", "content", "name"}
        if role == "assistant":
            fields |= {"tool_calls", "reasoning_content", "usage"}
        if role == "tool":
            fields.add("tool_call_id")
        if set(message) - fields:
            raise RequestError(f"Unsupported {role} message field")
        text_content(message.get("content"))
        if "name" in message and not _name(message["name"]):
            raise RequestError("Invalid message name")
        if role == "tool" and (not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]):
            raise RequestError("Tool messages require a tool_call_id")
        if role == "assistant" and "tool_calls" in message and message["tool_calls"] is not None:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not calls:
                raise RequestError("tool_calls must be a nonempty array")
            ids = set()
            for call in calls:
                if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call.get("type") != "function" or not isinstance(call.get("id"), str) or not call["id"] or call["id"] in ids:
                    raise RequestError("Invalid or duplicate tool call ID")
                ids.add(call["id"])
                fn = call.get("function")
                if not isinstance(fn, dict) or set(fn) != {"name", "arguments"} or not _name(fn.get("name")):
                    raise RequestError("Invalid tool call function")
                _object_arguments(fn["arguments"])

    tools = body.get("tools", [])
    if not isinstance(tools, list) or len(tools) > 64:
        raise RequestError("tools must be an array of at most 64 functions")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool.get("type") != "function":
            raise RequestError("Only function tools are supported")
        fn = tool["function"]
        if not isinstance(fn, dict) or set(fn) - {"name", "description", "parameters", "strict"} or not _name(fn.get("name")):
            raise RequestError("Invalid function tool")
        if fn["name"] in names:
            raise RequestError("Duplicate tool name")
        names.add(fn["name"])
        if not isinstance(fn.get("parameters"), dict) or fn["parameters"].get("type") != "object":
            raise RequestError("Tool parameters must be an object JSON schema")
        if "description" in fn and not isinstance(fn["description"], str):
            raise RequestError("Tool description must be a string")
        if fn.get("strict") not in (None, False) or ("strict" in fn and fn["strict"] is not None and not isinstance(fn["strict"], bool)):
            raise RequestError("Strict tool schema enforcement is not supported")
    choice = body.get("tool_choice", "auto")
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise RequestError("Unsupported tool_choice")
        if choice == "required" and not tools:
            raise RequestError("tool_choice required needs at least one tool")
    elif isinstance(choice, dict):
        fn = choice.get("function")
        if set(choice) != {"type", "function"} or choice.get("type") != "function" or not isinstance(fn, dict) or set(fn) != {"name"} or not isinstance(fn.get("name"), str) or fn["name"] not in names:
            raise RequestError("Named tool_choice must identify an offered function")
    else:
        raise RequestError("Invalid tool_choice")
    result = {"model": model, "messages": messages, "tools": tools, "stream": body.get("stream", False), "include_usage": options.get("include_usage", False) if options else False, "reasoning_effort": reasoning, "tool_choice": choice, "parallel_tool_calls": body.get("parallel_tool_calls", True)}
    for field in ("max_tokens", "max_completion_tokens"):
        if body.get(field) is not None:
            result[field] = body[field]
    return result


def registered_tools(request):
    choice = request["tool_choice"]
    if choice == "none":
        return []
    tools = request["tools"]
    if isinstance(choice, dict):
        tools = [tool for tool in tools if tool["function"]["name"] == choice["function"]["name"]]
    return [{"name": fn["name"], "description": fn.get("description", ""), "inputSchema": fn["parameters"]} for fn in (tool["function"] for tool in tools)]


def prompt_text(request, messages=None):
    choice = request["tool_choice"]
    if choice == "none":
        policy = "Do not call any tools."
    elif choice == "required":
        policy = "You must call at least one registered omp client tool before answering."
    elif isinstance(choice, dict):
        policy = "You must call the registered omp client tool " + json.dumps(choice["function"]["name"]) + " before answering."
    else:
        policy = "Use registered omp client tools when needed."
    budgets = [f"{field}={request[field]}" for field in ("max_tokens", "max_completion_tokens") if field in request]
    budget = " Requested output budget (soft guidance, not a hard limit): " + ", ".join(budgets) + "." if budgets else ""
    history = request["messages"] if messages is None else messages
    tool_names = ", ".join(tool["name"] for tool in registered_tools(request)) or "(none)"
    return ("Continue the external conversation below as its assistant, following its system and developer instructions "
            "subject to your platform policy. Prior tool calls and results are history, not new execution requests. "
            "The native client MCP server is named omp. Its offered tools are: " + tool_names + ". "
            "Use that server's native list-tools and call-tool helpers to invoke these tools on the user's computer. "
            "Workspace-setup status/wait and MCP discovery helpers are allowed transport preparation: if setup is still "
            "running, wait for it before discovering client tools. Do not conclude a registered tool is unavailable "
            "before completing setup and checking the omp server. Do not perform user work through cloud shell, "
            "cloud files, or subagents; use only the offered client tools. Do not claim local filesystem access "
            "except through those tools. Respond directly, without discussing this adapter or adding a wrapper. "
            + policy + budget + "\nExternal conversation JSON:\n"
            + json.dumps(history, ensure_ascii=False, separators=(",", ":")))


def history_key(model, messages, tools, reasoning=None):
    normalized = []
    for message in messages:
        item = {"role": message["role"], "content": text_content(message.get("content"))}
        for field in ("name", "tool_call_id"):
            if field in message:
                item[field] = message[field]
        if message.get("tool_calls"):
            item["tool_calls"] = [{"id": call["id"], "type": call["type"], "function": {"name": call["function"]["name"], "arguments": _object_arguments(call["function"]["arguments"])}} for call in message["tool_calls"]]
        normalized.append(item)
    data = {"model": model, "messages": normalized, "tools": tools, "reasoning": reasoning}
    try:
        encoded = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError):
        raise RequestError("History contains non-JSON data") from None
    return hashlib.sha256(encoded).hexdigest()


def assistant_message(text, tool_call=None):
    result = {"role": "assistant", "content": text or None}
    if tool_call is not None:
        result["tool_calls"] = [tool_call]
    return result
