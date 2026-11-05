"""Remote API-backed variants for manage.py's Proteus-routed server.

Deliberately separate from the local VARIANTS pool in manage.py: Proteus's
GBNF grammar and few-shot classifier are built ONLY from local VARIANTS
keys (see manage.py's ROUTER_GRAMMAR), so it is structurally impossible for
the auto-router to pick a remote variant on its own -- not just "told not
to" via a prompt, but incapable of ever emitting one of these keys. A
remote variant can only be reached through an explicit `manage.py start
<key>` (or the interactive menu's remote section), gated by a typed
confirmation (or --yes) -- selecting one means prompts, INCLUDING whatever
ASMC's memory layer injects into them, leave this machine and go to a third
party. Once selected, Proteus will not auto-swap away from it either (see
the `not backend_state.is_remote` guard in manage.py's chat_completions) --
a remote choice is sticky until you explicitly switch back via the CLI.

Fill in REMOTE_VARIANTS below with your own endpoints/models. API keys are
read from environment variables at request time -- never hardcode one here.

Prototype status: non-streaming works the same way for every provider
below (translate request -> send -> translate response back to OpenAI
chat/completions shape). Streaming is real passthrough for
"openai_compatible" providers (their wire format already matches what
manage.py's client expects), but for "anthropic" it currently buffers the
full reply and emits it as a single synthetic chunk instead of true
token-by-token streaming -- Anthropic's SSE event format
(content_block_delta/message_delta/...) differs from OpenAI's and isn't
translated chunk-by-chunk yet. Also not ported from the local path: the
repetition-loop detection/re-prompting manage.py does for local backends --
not implemented here yet, on the assumption frontier models need it far
less than small local quants do.
"""
import os

import httpx

REMOTE_VARIANTS = {
    "claude": {
        "label": "Claude (Anthropic API)",
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1/messages",
        "model": os.environ.get("CLAUDE_REMOTE_MODEL", "claude-sonnet-5-5"),
        "api_key_env": "ANTHROPIC_API_KEY",
        "privacy_note": "sent to Anthropic's API",
    },
    "openai": {
        "label": "GPT (OpenAI API)",
        "provider": "openai_compatible",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": os.environ.get("OPENAI_REMOTE_MODEL", "gpt-5"),
        "api_key_env": "OPENAI_API_KEY",
        "privacy_note": "sent to OpenAI's API",
    },
    "qwen-remote": {
        "label": "Qwen (remote, OpenAI-compatible endpoint)",
        "provider": "openai_compatible",
        "base_url": os.environ.get(
            "QWEN_REMOTE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        ),
        "model": os.environ.get("QWEN_REMOTE_MODEL", "qwen-max"),
        "api_key_env": "QWEN_REMOTE_API_KEY",
        "privacy_note": "sent to a remote Qwen API endpoint",
    },
}


def api_key_for(key: str) -> str:
    cfg = REMOTE_VARIANTS[key]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"Missing API key for '{key}' -- set the {cfg['api_key_env']} "
            f"environment variable before selecting this variant."
        )
    return api_key


def _flatten_content(content) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Anthropic wants `system` as a separate top-level field, not a message
    with role=system -- this pulls every system message out and joins them,
    which is also where whatever ASMC injected ends up (inject_memory() in
    manage.py prepends/merges into the first system message)."""
    system_parts, converted = [], []
    for m in messages:
        role = m.get("role")
        content = _flatten_content(m.get("content"))
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            converted.append({"role": role, "content": content})
    return "\n\n".join(system_parts), converted


def _anthropic_to_openai_shape(data: dict, fallback_model: str) -> dict:
    text = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    return {
        "id": data.get("id", "remote-claude"),
        "object": "chat.completion",
        "model": data.get("model", fallback_model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": data.get("stop_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
    }


async def send(key: str, body: dict, client: httpx.AsyncClient) -> dict:
    """Non-streaming: translate `body` (OpenAI chat/completions shape) to
    whatever the provider for `key` wants, send it, and return a response
    already in OpenAI chat/completions shape -- so manage.py's memory
    storage and response handling don't need to know which provider
    answered. Raises httpx.HTTPStatusError / RuntimeError on failure."""
    cfg = REMOTE_VARIANTS[key]
    api_key = api_key_for(key)

    if cfg["provider"] == "openai_compatible":
        headers = {"Authorization": f"Bearer {api_key}"}
        remote_body = dict(body)
        remote_body["model"] = cfg["model"]
        remote_body["stream"] = False
        resp = await client.post(cfg["base_url"], json=remote_body, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json()

    if cfg["provider"] == "anthropic":
        system, messages = _split_system(body.get("messages", []))
        remote_body = {
            "model": cfg["model"],
            "max_tokens": body.get("max_tokens") or 1024,
            "messages": messages,
        }
        if system:
            remote_body["system"] = system
        if "temperature" in body:
            remote_body["temperature"] = body["temperature"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = await client.post(cfg["base_url"], json=remote_body, headers=headers, timeout=120)
        resp.raise_for_status()
        return _anthropic_to_openai_shape(resp.json(), cfg["model"])

    raise RuntimeError(f"Unknown provider '{cfg['provider']}' for '{key}'.")


async def stream(key: str, body: dict, client: httpx.AsyncClient):
    """Streaming: yields raw SSE lines ("data: {...}\\n\\n", ..., finally
    "data: [DONE]\\n\\n") already in OpenAI delta-chunk shape. Real
    passthrough for openai_compatible providers; for anthropic, buffers via
    send() and fakes a single-chunk stream -- see module docstring."""
    cfg = REMOTE_VARIANTS[key]

    if cfg["provider"] == "openai_compatible":
        api_key = api_key_for(key)
        headers = {"Authorization": f"Bearer {api_key}"}
        remote_body = dict(body)
        remote_body["model"] = cfg["model"]
        remote_body["stream"] = True
        async with client.stream(
            "POST", cfg["base_url"], json=remote_body, headers=headers, timeout=None
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n\n"
        return

    import json as _json
    data = await send(key, body, client)
    content = data["choices"][0]["message"]["content"]
    chunk = {
        "id": data.get("id", "remote"),
        "object": "chat.completion.chunk",
        "model": data.get("model", cfg["model"]),
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
    }
    yield f"data: {_json.dumps(chunk)}\n\n"
    done_chunk = dict(chunk)
    done_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    yield f"data: {_json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"
