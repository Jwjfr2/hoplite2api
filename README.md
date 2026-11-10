![hoplite2api — Hoplite models, local OMP tools](assets/thumbnail.svg)

[![Tests](https://github.com/Jwjfr2/hoplite2api/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Jwjfr2/hoplite2api/actions/workflows/tests.yml)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
[![Hoplite ACP](https://img.shields.io/badge/transport-Hoplite%20ACP-555555)](https://hoplite.sh/docs/cli/acp)

Use Hoplite models in OMP. The proxy translates Chat Completions requests into ACP sessions and sends native tool calls back to OMP for local execution.

Hoplite's cloud harness stays in the loop. This is not a raw inference endpoint.

## Run

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), a Hoplite member API key, and a repository-backed Hoplite project.

```sh
git clone https://github.com/Jwjfr2/hoplite2api.git
cd hoplite2api
uv sync --locked
uv run hoplite2api --project YOUR_PROJECT_ID
```

Set `HOPLITE_API_KEY` in your environment, or store the key in OMP under `hoplite`; the proxy falls back to `omp token hoplite`. Keep credentials out of command arguments and repository files.

The server listens on `127.0.0.1:8091`. Use `--port` to change it. Requests to `/v1/models` and `/v1/chat/completions` require the same key as a Bearer token. `/health` requires no authentication.

## Connect OMP

Add this provider to `~/.omp/agent/models.yml`:

```yaml
providers:
  hoplite:
    baseUrl: http://127.0.0.1:8091/v1
    apiKey: HOPLITE_API_KEY # Omit this line when using OMP's stored credential.
    api: openai-completions
    compat:
      supportsStrictMode: false
      supportsDeveloperRole: true
      supportsReasoningEffort: true
      thinkingFormat: openai
    discovery:
      type: openai-models-list
```

Then start OMP:

```sh
omp models hoplite
omp --model hoplite/gpt-6.1-sol
```

The model list comes from your Hoplite workspace. Context limits and reasoning levels follow its catalog. Model availability still depends on your account.

## How tools work

A native ACP callback becomes an OpenAI `tool_calls` response. OMP executes the tool locally and returns its result. The proxy delivers that result to the original callback over the same open WebSocket.

Conversation history, model, tools, and reasoning must match before a pending call can resume. Invalid or expired continuations return `409`. Completed sessions are reused for five minutes. Restarting the proxy discards pending calls and session state.

## Limits

- **Hoplite still controls the harness.** System/developer messages are included in the task prompt, not installed as upstream system messages. Instructions against cloud tools are not isolation.
- **Tool results have a 60-second deadline and a 512 KiB limit.** Local operations are not rolled back if the upstream call expires.
- **Text only.** Images, audio, sampling controls, structured response formats, strict schemas, and multiple completions are rejected.
- **ACP requests must fit 1 MiB.** Up to 64 functions can be registered; Hoplite trims schemas and descriptions.
- **Output budgets are soft guidance.** `max_tokens` and `max_completion_tokens` are not hard caps.
- **Check Hoplite for usage and billing.** Token counters are not relayed to OMP. The cloud harness can make multiple model calls per response, and sandbox time is billable. Trial and spending limits still apply.
- **Keep it local.** Do not expose this server through a public tunnel or network listener.

## Test

```sh
uv run python -m unittest discover -s tests -v
```

Tests run without Hoplite credentials. Live verification used `gpt-6.1-sol` to request OMP's local `read` tool and return an unpredictable file's exact contents. Other advertised models have not been tested end-to-end.

[Issues](https://github.com/Jwjfr2/hoplite2api/issues) · [Actions](https://github.com/Jwjfr2/hoplite2api/actions) · [ACP documentation](https://hoplite.sh/docs/cli/acp)
