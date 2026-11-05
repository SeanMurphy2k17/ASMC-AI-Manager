#!/usr/bin/env python3
"""Manager + memory-augmented server for local Qwen3.6-35B-A3B variants.

This script IS the thing clients (Copilot included) connect to -- it serves
one OpenAI-compatible endpoint on :18090 with ASMC semantic memory built in,
and supervises llama-server as an internal-only backend child on :18080.
There is no separate proxy process; ASMC is a library here, not a server.

Proteus, a tiny always-on CPU router, picks a variant per request and swaps
the GPU backend to match -- pass --no-router on start/restart to disable
that and stay on whichever variant you started. Pass --no-memory to run as
a plain stateless proxy: no ASMC injection, storage, or memory tools. Pass
--lan to bind the public :18090 server to all interfaces so other machines
on your network can use it (the backend/router/Strata stay loopback-only).
The interactive prompt asks about both. Ctrl+C (and `stop`) tear everything
down including Strata, so no model is left loaded; `restart flash` keeps it
warm since it's about to be reused, and `stop --keep-warm` does so on demand.

Run with no arguments for an interactive prompt (pick a variant, it boots).
Or use subcommands directly: start [key], stop, restart [key], status, list.
Foreground by default (Ctrl+C stops everything cleanly); --detach backgrounds it.
"""
import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Machine-specific settings, kept out of the repository: an optional local_settings.json beside
# this file, e.g. {"LLAMA_SERVER": "/path/to/llama.cpp/build/bin/llama-server"}. Any of the
# environment variables below can go there; a variable set in the environment wins.
_LOCAL_SETTINGS = os.path.join(BASE_DIR, "local_settings.json")
if os.path.exists(_LOCAL_SETTINGS):
    with open(_LOCAL_SETTINGS) as _f:
        for _k, _v in json.load(_f).items():
            os.environ.setdefault(_k, str(_v))
# Where the GGUF files live. Override with ASMC_AI_MODELS_DIR to keep them elsewhere.
MODELS_DIR = os.environ.get("ASMC_AI_MODELS_DIR") or os.path.join(BASE_DIR, "models")
_VENV_DIR = os.path.join(BASE_DIR, ".venv")
if os.path.isdir(_VENV_DIR) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV_DIR):
    venv_python = os.path.join(_VENV_DIR, "bin", "python3")
    os.execv(venv_python, [venv_python] + sys.argv)

# Imports httpx itself -- must come after the venv re-exec guard above, same
# reason the fastapi/uvicorn imports further down do: under the system
# python (before re-exec), httpx/remote_apis's own import of it may not be
# installed at all, and we want the re-exec to happen first.
import remote_apis  # noqa: E402

LOG_DIR = os.path.join(BASE_DIR, "logs")
STATE_FILE = os.path.join(LOG_DIR, "current.json")
PID_FILE = os.path.join(LOG_DIR, "server.pid")
# Written by stop_current() right before it signals the running server, read
# and deleted by that server's _shutdown(): names the external variant key
# (or "*") to leave warm instead of unloading -- i.e. "you're about to be
# restarted/reused, don't pay the reload". Absent = tear everything down.
KEEP_WARM_FILE = os.path.join(LOG_DIR, "keep_warm")
LOG_FILE = os.path.join(LOG_DIR, "server.log")
# The llama.cpp server binary: LLAMA_SERVER if set, else `llama-server` on PATH.
LLAMA_SERVER = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server") or "llama-server"
STRATA_BIN = "strata"  # resolved via PATH, same as running it by hand
HOST = "127.0.0.1"
PUBLIC_PORT = 18090   # what clients (Copilot, curl) connect to -- memory-augmented
BACKEND_PORT = 18080  # llama-server itself, internal only, never exposed to clients
BACKEND_BASE_URL = f"http://{HOST}:{BACKEND_PORT}"

# Always-on CPU classifier that picks a variant per request -- see the
# "smart loader" section below. Separate port/process from BACKEND_PORT,
# never swapped, never touches VRAM (--n-gpu-layers 0), so it never competes
# with whatever big GPU variant is currently loaded.
#
# Named Proteus -- the shapeshifting sea god -- because its only job is to
# take whatever form (variant) the moment calls for, within whatever this
# machine's VRAM actually allows right now. Purely cosmetic (shows up in
# logs and in the "model" field of its own completion requests), but it's
# the one piece of this stack that isn't a fixed downloaded model identity.
ROUTER_NAME = "Proteus"
ROUTER_PORT = 18070
ROUTER_BASE_URL = f"http://{HOST}:{ROUTER_PORT}"
ROUTER_MODEL_PATH = os.path.join(MODELS_DIR, "router", "Qwen3.5-0.8B-Q8_0.gguf")
ROUTER_ARGS = ["--n-gpu-layers", "0", "--ctx-size", "2048", "--parallel", "1"]

# Independent memory store -- ASMC otherwise stores beside its own module, so two
# programs using the same ASMC checkout would share one store. Setting
# ASMC_STORE_DIR keeps this server's memory in its own folder (memory/ here).
os.environ.setdefault("ASMC_STORE_DIR", os.path.join(BASE_DIR, "memory"))

# Total context budget proven to fit fully on a 24.5GB GPU with zero CPU
# offload at --parallel 1. llama.cpp's --ctx-size is PER SLOT, not shared, so
# each extra concurrent agent needs its own full copy of that cost -- the only
# way to add concurrency within the same VRAM budget is to divide this total
# across slots (e.g. 4 agents -> ~32K context each instead of 1 agent -> 131K).
TOTAL_CONTEXT = 131072

# Quantized KV cache and trimmed batch/ubatch (compute buffers don't scale
# with context, so this headroom matters regardless of --ctx-size or slot
# count).
#
# Reasoning is left ON (no --reasoning off) -- this model family's benchmarked
# capability (e.g. Coder390's LCB 90/100 vs. 83/100 stock) was measured WITH
# reasoning; turning it off trades that capability away. Expect single turns
# to burn 20-40k+ tokens of thinking on harder problems -- budget context
# accordingly (favor n_agents=1 for the full TOTAL_CONTEXT window rather than
# splitting it across concurrent slots).
BASE_ARGS = [
    "--n-gpu-layers", "999",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--batch-size", "512", "--ubatch-size", "256",
]

# Add more variants here as you find/download more fine-tunes for other tasks.
# Each just needs a model_path; override ctx/cache/batch via extra_args if a
# given file needs different headroom (see BASE_ARGS comment above).
VARIANTS = {
    "r": {
        "label": "Regular (official Qwen3.6-35B-A3B) (llama.cpp)",
        "router_hint": (
            "Qwen3.6-35B-A3B Regular, the general-purpose default. A mixture-of-experts "
            "model good at everyday conversation, writing, brainstorming, research-style "
            "explanations, and general knowledge questions. Has no special coding, "
            "uncensoring, or vision tuning -- use it for anything that does not "
            "specifically need one of the other specialists."
        ),
        "model_path": os.path.join(MODELS_DIR, "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    },
    "u": {
        "label": "Uncensored (Heretic, Native-MTP-Preserved) (llama.cpp)",
        "router_hint": (
            "Heretic Native-MTP-Preserved Uncensored, the same 35B-A3B base but abliterated "
            "so it will not refuse or soften mature, explicit, violent, or otherwise "
            "restricted content. Use it for uncensored creative writing, roleplay, or any "
            "request the regular model would decline or sanitize."
        ),
        "model_path": os.path.join(
            MODELS_DIR,
            "Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf",
        ),
    },
    "s": {
        "label": "Coder (CoderX, pruned 27B long-horizon coding specialist) (llama.cpp)",
        "router_hint": (
            "CoderX, a pruned 27B coding specialist built for long-horizon, multi-step "
            "development work: planning and carrying out changes that span many files or "
            "many steps of an ongoing project, not just a single snippet. Use it when the "
            "user describes a big, sustained coding effort rather than a quick one-off "
            "question."
        ),
        "model_path": os.path.join(
            MODELS_DIR,
            "Qwen3.6-27B-A3B-CoderX-Q5_K_M.gguf",
        ),
    },
    "v": {
        "label": "Vision (Qwen3.8-27B dense, UD-Q5_K_M, image input) (llama.cpp)",
        "router_hint": (
            "Vision, a 27B dense model that actually loads a vision tower alongside its "
            "text weights, so it is the only variant that can see. Use it whenever the "
            "user has attached an image, screenshot, or diagram, or is asking about "
            "visual content specifically."
        ),
        "model_path": os.path.join(MODELS_DIR, "Qwen3.8-27B-UD-Q5_K_M.gguf"),
        # Dense model (no MoE headroom trick), plus a vision tower loaded
        # alongside the weights -- less VRAM slack than the other variants,
        # so this one gets BASE_ARGS plus the mmproj file for image input.
        "extra_args": BASE_ARGS + [
            "--mmproj", os.path.join(MODELS_DIR, "mmproj-F16.gguf"),
        ],
    },
    "c": {
        "label": "Coder390 (Qwen3.8-27B dense, Q4LynnStyle, MTP-tuned coding specialist) (llama.cpp)",
        "router_hint": (
            "Coder390, a 27B dense coding specialist distilled from multiple strong "
            "teacher models, with reasoning trimmed for efficiency and speculative "
            "decoding for speed, benchmarking noticeably above stock on coding "
            "benchmarks. This is the default go-to coder for single-turn code writing, "
            "debugging bugs and errors, explaining, and refactoring -- prefer it over "
            "CoderX unless the task is explicitly described as a big multi-session "
            "project."
        ),
        "model_path": os.path.join(
            MODELS_DIR, "coder390",
            "GGUF", "Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-MTP.gguf",
        ),
    },
}

# Variants whose process lifecycle is NOT owned by this script -- each is an
# already-running, OpenAI-compatible server on this same machine, started
# and stopped separately (e.g. `strata launch` / `strata launch --stop`).
# Kept OUT of Proteus's routable pool for the same structural reason remote
# variants are (see remote_apis.py): manage.py can't safely auto-swap into
# something it doesn't control the lifecycle of. It's also VRAM-exclusive
# with every VARIANTS entry on a single GPU -- Qwen3.8-Flash-Next alone fills
# nearly the whole card, so running it and a local variant at once will OOM
# whichever loads second. Select it explicitly with `manage.py start flash`.
EXTERNAL_VARIANTS = {
    "flash": {
        "label": "Qwen3.8-Flash-Next (UD-IQ4_XS, ~180B/10-of-512 MoE + MTP) (strata)",
        "base_url": "http://127.0.0.1:8080",
        # Relaunched at 128K (131072) to match the local variants' context,
        # specifically so ASMC's memory injection behaves consistently
        # everywhere instead of needing special-casing -- Strata's own
        # default for this quant/card was a much smaller 32K, which a full
        # memory injection blew straight through (confirmed: hard 400,
        # prompt alone exceeded the window). Keep this in sync with
        # whatever --context `strata launch` actually used; the
        # memory_char_budget() trimming logic in chat_completions stays on
        # regardless as a safety net, it just won't normally trigger now.
        "context": 131072,
        # Used by launch_external() to boot this if it isn't already up --
        # verified via `strata launch --plan` before ever running for real.
        # --context derived from TOTAL_CONTEXT so it can't drift out of sync
        # with the "context" value above.
        "launch_args": [
            "--model", "ud-iq4_xs", "--family", "unsloth",
            "--context", f"{TOTAL_CONTEXT // 1024}k",
            "--vision", "gpu", "--tune", "skip",
            "--access", "local", "--port", "8080",
            "--yes", "--no-tui",
        ],
        # Used by unload_external() -- run on every Ctrl+C / `stop` unless a
        # restart of this same variant (or `stop --keep-warm`) says to keep it.
        "stop_args": ["launch", "--stop"],
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [manage] %(message)s")
log = logging.getLogger("manage")


# --- state / pid helpers -------------------------------------------------

def read_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def write_state(key, n_agents, ctx_per_agent, router_enabled=True, is_remote=False, is_external=False,
                memory_enabled=None, lan=None):
    # memory_enabled/lan are per-session choices that never change after
    # boot, so the live-swap callers (ensure_variant/ensure_external) don't
    # need to thread them through -- None means "whatever this session was
    # started with".
    if memory_enabled is None:
        memory_enabled = backend_state.memory_enabled
    if lan is None:
        lan = backend_state.lan
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({
            "key": key, "n_agents": n_agents, "ctx_per_agent": ctx_per_agent,
            "router_enabled": router_enabled, "is_remote": is_remote, "is_external": is_external,
            "memory_enabled": memory_enabled, "lan": lan,
        }, f)


def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def read_pid():
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def backend_health_ok(timeout=1):
    try:
        with urllib.request.urlopen(f"{BACKEND_BASE_URL}/health", timeout=timeout):
            return True
    except Exception:
        return False


def public_health_ok(timeout=1):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PUBLIC_PORT}/health", timeout=timeout):
            return True
    except Exception:
        return False


def external_health_ok(key, timeout=2):
    """Strata's /health returns {"status": "ok", ...}, not the bare 200
    llama-server gives -- still just needs to respond, but check the body
    isn't an error shape rather than assuming any 200 means healthy."""
    try:
        with urllib.request.urlopen(f"{EXTERNAL_VARIANTS[key]['base_url']}/health", timeout=timeout) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def lan_ip() -> str | None:
    """This machine's LAN IPv4, for the startup banner when --lan is on.
    The UDP-connect trick picks the interface that routes outward without
    sending anything; falls back to None if there's no route at all."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def router_health_ok(timeout=1):
    try:
        with urllib.request.urlopen(f"{ROUTER_BASE_URL}/health", timeout=timeout):
            return True
    except Exception:
        return False


# --- backend process management (startup AND live router-triggered swaps) -

class BackendState:
    """Mutable, in-process record of the currently running GPU backend --
    separate from the on-disk STATE_FILE (which is for cross-process status/
    stop from another terminal). `lock` serializes swap attempts so two
    concurrent requests that disagree on the variant don't both try to kill
    and relaunch the backend at once."""
    def __init__(self):
        self.proc = None
        self.key = None
        self.n_agents = 1
        self.router_enabled = True
        self.is_remote = False
        self.is_external = False
        self.memory_enabled = True
        self.lan = False
        self.lock = asyncio.Lock()


backend_state = BackendState()


def launch_backend(key, n_agents):
    """Spawn the GPU backend for `key` and block until it's healthy. Used
    both at startup and for live router-triggered swaps -- raises instead of
    exiting so a failed swap doesn't take the whole server down with it."""
    variant = VARIANTS[key]
    model_path = variant["model_path"]
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found: {model_path}")

    ctx_per_agent = TOTAL_CONTEXT // n_agents
    extra_args = variant.get("extra_args", BASE_ARGS)
    cmd = [
        LLAMA_SERVER, "-m", model_path, "--port", str(BACKEND_PORT), "--host", HOST,
        "--ctx-size", str(ctx_per_agent), "--parallel", str(n_agents),
        *extra_args,
    ]
    print(f"Starting '{variant['label']}' -- {n_agents} agent(s), {ctx_per_agent} context each...")
    proc = subprocess.Popen(cmd, start_new_session=True)
    print(f"Backend pid {proc.pid}, waiting for it to come up...")

    for _ in range(180):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"Backend exited early (code {proc.returncode}).")
        if backend_health_ok():
            return proc, ctx_per_agent
    proc.terminate()
    raise RuntimeError("Backend did not come up after 180s.")


def terminate_backend(proc, timeout=45):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"Did not exit after {timeout}s, sending SIGKILL.")
        proc.kill()
        proc.wait()


def launch_router():
    """Spawn the always-on CPU router model. Independent lifecycle from the
    GPU backend -- started once at server startup, never swapped."""
    cmd = [
        LLAMA_SERVER, "-m", ROUTER_MODEL_PATH, "--port", str(ROUTER_PORT), "--host", HOST,
        *ROUTER_ARGS,
    ]
    print(f"Starting {ROUTER_NAME} (CPU router, :{ROUTER_PORT})...")
    proc = subprocess.Popen(cmd, start_new_session=True)
    for _ in range(60):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"{ROUTER_NAME} exited early (code {proc.returncode}).")
        if router_health_ok():
            print(f"{ROUTER_NAME} ready (pid {proc.pid}).")
            return proc
    proc.terminate()
    raise RuntimeError(f"{ROUTER_NAME} did not come up after 60s.")


def launch_external(key):
    """Bring an EXTERNAL_VARIANTS server up if it isn't already healthy --
    makes `manage.py start flash` a single command instead of needing a
    separate `strata launch` first. Reuses an already-running, already-warm
    instance as-is rather than relaunching it (cheap check, big time save).

    Not stored in backend_state.proc (there's no Popen handle to keep --
    see below) and not torn down by terminate_backend(); instead
    _shutdown() calls unload_external() so Ctrl+C / `stop` leave no stray
    model behind. The one exception is a restart or switch *to this same
    variant*, where stop_current() leaves a KEEP_WARM_FILE so the
    ~minutes-long reload isn't paid twice in a row; `stop --keep-warm`
    does the same on demand. The launching `strata launch --no-tui` process itself exits
    once the real server is up (confirmed: its own server processes persist
    independently afterward) -- only its *non-zero* exit is treated as a
    failure here, a clean exit mid-poll just means keep watching health.
    start_new_session=True keeps it alive through our own Ctrl+C too."""
    cfg = EXTERNAL_VARIANTS[key]
    if external_health_ok(key):
        print(f"'{key}' ({cfg['label']}) is already up at {cfg['base_url']} -- reusing it.")
        return
    cmd = [STRATA_BIN, "launch", *cfg.get("launch_args", [])]
    print(f"'{key}' isn't running yet -- launching it (this can take a while for a model "
          f"this size): {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, start_new_session=True)
    for _ in range(600):
        time.sleep(1)
        if proc.poll() is not None and proc.returncode != 0:
            raise RuntimeError(f"Launching '{key}' exited early (code {proc.returncode}).")
        if external_health_ok(key):
            print(f"'{key}' ready at {cfg['base_url']}.")
            return
    raise RuntimeError(f"'{key}' did not come up after 600s.")


def unload_external(quiet=False):
    """Shut down any running EXTERNAL_VARIANTS server (frees its VRAM).
    Called from _shutdown() on every Ctrl+C / `stop` so no stray model is
    left behind, and from stop_current() when nothing is running but a
    stray external server is. Skipped only when a KEEP_WARM_FILE says the
    same variant is about to be reused (restart) or `stop --keep-warm` was
    asked for. Idempotent: does nothing for a server that isn't answering."""
    for key, cfg in EXTERNAL_VARIANTS.items():
        if not cfg.get("stop_args") or not external_health_ok(key):
            continue
        if not quiet:
            print(f"Unloading '{key}' ({cfg['label']})...")
        subprocess.run([STRATA_BIN, *cfg["stop_args"]], check=False)
        for _ in range(30):
            if not external_health_ok(key):
                break
            time.sleep(1)
        if not quiet:
            print(f"Unloaded '{key}'." if not external_health_ok(key)
                  else f"'{key}' is still answering after 30s -- check `strata status`.")


def stop_current(quiet=False, keep_warm=False, keep_external_key=None):
    """Cross-process stop: signal whoever is running (ourselves in another
    terminal, or a detached instance) and let it do its own graceful
    shutdown -- see _shutdown() in _run_server. That handler tears down
    everything including any external server (Strata), unless told to keep
    it warm via KEEP_WARM_FILE: keep_warm=True keeps it regardless,
    keep_external_key=<key> keeps it only if that's the variant being
    reused next (restart / switch to the same external variant)."""
    state = read_state()
    pid = read_pid()
    if state is None and pid is None:
        if not quiet:
            print("Nothing running.")
        if not keep_warm:
            unload_external(quiet=quiet)  # a stray external server with no manage.py in front of it
        return
    marker = "*" if keep_warm else keep_external_key
    if pid is not None:
        if marker:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(KEEP_WARM_FILE, "w") as f:
                f.write(marker)
        print(f"Stopping pid {pid}...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(45):  # backend itself takes ~8s to release a 20GB model from VRAM
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(1)
        else:
            print("Did not exit after 45s, sending SIGKILL.")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            # SIGKILL means _shutdown() never ran, so do its external cleanup here.
            if os.path.exists(KEEP_WARM_FILE):
                os.remove(KEEP_WARM_FILE)
            if not marker:
                unload_external(quiet=quiet)
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    label = (
        variant_label(state["key"], state.get("is_remote", False), state.get("is_external", False))
        if state else "auto-load, no variant ever picked"
    )
    clear_state()
    if not quiet:
        print(f"Stopped ({label}).")


# --- ASMC memory layer (library, not a separate server) ------------------

# ASMC (https://github.com/SeanMurphy2k17/AdvancedSemanticMemoryClustering-ASMC): the asmc/
# submodule beside this file, or wherever ASMC_PATH points.
ASMC_PATH = os.environ.get("ASMC_PATH") or os.path.join(BASE_DIR, "asmc")
sys.path.insert(0, ASMC_PATH)
# ASMC_API is imported lazily inside _run_server, only when memory is enabled
# -- a --no-memory run never loads faiss/lmdb at all.

memory = None  # set inside _run_server (when enabled), before uvicorn starts accepting connections


# What the agent harness sends as a "user" message is often not the person at
# all: compaction requests and summaries, shell-command notifications, tool
# notices, context blocks, task-complete nags. Stored as experiences they
# crowd out the conversation (measured 2026-11-04: 461 records, 67 distinct
# real prompts). Real prompts also arrive wrapped in a datetime tag and a
# tagged-files block; strip those, keep the words.
_NOISE_PREFIXES = (
    "please create a detailed summary of the conversation so far",
    "some of the conversation history has been summarized",
    "this session is being continued from a previous conversation",
    "<system_notification>",
    "<tools_changed_notice>",
    "<context>",
    "you have not yet marked the task as complete",
)
_DATETIME_TAG = re.compile(r"^\s*<current_datetime>.*?</current_datetime>\s*", re.S)
_TRAILING_BLOCKS = re.compile(r"\s*<(tagged_files|context|tools_changed_notice|reminderInstructions|editorContext)>.*?</\1>\s*", re.S)
_UNWRAP = re.compile(r"<userRequest>(.*?)</userRequest>", re.S)  # the person's words, wrapped: unwrap, keep


def clean_prompt(text: str) -> str:
    """The person's words, without the harness decoration around them."""
    t = _DATETIME_TAG.sub("", text or "", count=1)
    t = _UNWRAP.sub(r"\1", t)
    return _TRAILING_BLOCKS.sub(" ", t).strip()


def is_harness_text(text: str) -> bool:
    t = clean_prompt(text).lower()
    return not t or any(t.startswith(p) for p in _NOISE_PREFIXES)


_INLINE_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _clip_middle(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    return text[:head].rstrip() + "\n[...]\n" + text[-(limit - head):].lstrip()


def with_reasoning(reply: str, reasoning: str) -> str:
    """The thinking model puts its preferences and its reasons in the thinking
    and often only a few words in the answer, so the answer alone is the
    wrong half to remember. Keep a bounded slice of the thinking under the
    answer; the library indexes the words of both. Backends that inline the
    thinking as <think> tags are handled here too."""
    reply = reply or ""
    inline = _INLINE_THINK.findall(reply)
    if inline:
        reasoning = (reasoning or "") + "\n".join(inline)
        reply = _INLINE_THINK.sub("", reply).strip()
    if not MEMORY_STORE_REASONING or MEMORY_REASONING_CHARS <= 0 or not (reasoning or "").strip():
        return reply
    return f"{reply}\n\n[thinking]\n{_clip_middle(reasoning, MEMORY_REASONING_CHARS)}".strip()


def remember(situation: str, response: str, reasoning: str = ""):
    """Store an exchange in ASMC -- a no-op when this session runs --no-memory,
    and a no-op for harness text that is not conversation. The model's
    thinking, when the backend reports it, is kept with the answer."""
    if memory is None:
        return
    response = with_reasoning(response, reasoning)
    if MEMORY_SKIP_HARNESS_TEXT and is_harness_text(situation):
        log.info("not stored: harness text, not conversation (%.60s)", situation.strip())
        return
    memory.add_experience(situation=clean_prompt(situation), response=response)
    log.info("stored experience (%d chars reply)", len(response))

MEMORY_LAYER1_COUNT = 5   # ASMC's own short-term/recent semantic matches
MEMORY_LAYER2_COUNT = 6   # long-term semantic matches to pull in
MEMORY_ASSOCIATIVE_COUNT = 2   # memories linked to the hits, not matched to the query: the wandering
MEMORY_SKIP_HARNESS_TEXT = True   # do not store compaction/notification text as experiences
MEMORY_STORE_REASONING = True     # keep the model's thinking with each stored answer (0 chars disables)
MEMORY_REASONING_CHARS = 2000     # how much of the thinking to keep: head and tail, middle elided
MEMORY_RECENT_COUNT = 8   # raw chronological turns for conversational continuity

# These counts were sized for the local variants' 131072-token window. A
# fixed count doesn't bound actual size, though -- individual ASMC entries
# vary wildly (a one-line Q&A vs. a dumped <context>/shell-output block), so
# the real injected text can be far larger than the count alone suggests
# (confirmed: ~52.5K chars from just ~19 entries). Budget by characters
# against whatever's actually active instead of trusting the counts alone.
MEMORY_RESPONSE_RESERVE_TOKENS = 4096  # room for the real turn + an actual answer
# Conservative on purpose: measured ~1 char/token on real recalled content
# (dense with structured/code-like text, not prose) -- a looser estimate
# like the usual ~4 chars/token would have re-caused the exact overflow
# this budget exists to prevent.
MEMORY_CHARS_PER_TOKEN_ESTIMATE = 1.5


def active_context_tokens() -> int:
    """Best-known context window for whatever's currently active, used only
    to size memory injection -- not a hard limit we enforce elsewhere."""
    if backend_state.is_external:
        return EXTERNAL_VARIANTS.get(backend_state.key, {}).get("context", TOTAL_CONTEXT)
    if backend_state.is_remote:
        return 128_000  # cloud providers have their own large windows; just a sane cap
    return TOTAL_CONTEXT // max(backend_state.n_agents, 1)


def memory_char_budget() -> int:
    usable_tokens = max(active_context_tokens() - MEMORY_RESPONSE_RESERVE_TOKENS, 1024)
    return int(usable_tokens * MEMORY_CHARS_PER_TOKEN_ESTIMATE)


def latest_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # OpenAI content-parts format
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _interleave(*layers):
    """Alternate entries from each layer (long-term first) so that trimming from
    the end, or clipping to k, takes from both. Concatenating the window first
    meant every long-term hit was the first to be dropped once over budget."""
    out = []
    for i in range(max((len(layer) for layer in layers), default=0)):
        for layer in layers:
            if i < len(layer):
                out.append(layer[i])
    return out


def _format_entries(entries: list[dict], seen: set[str]) -> list[str]:
    lines = []
    for entry in entries:
        situation = entry.get("inputText", "").strip()
        response = entry.get("responseText", "").strip()
        if not situation or situation in seen:
            continue
        seen.add(situation)
        lines.append(f"- User previously asked: {situation}\n  You replied: {response}")
    return lines


def build_full_context(query: str) -> str | None:
    """Two complementary memory sources: recent turn-based continuity (same
    as a normal rolling context window, but survives this script restarting)
    plus ASMC semantic recall (topically relevant exchanges from anywhere in
    long-term memory, however long ago) -- both trimmed to fit whatever
    context window is actually active (see memory_char_budget()), since a
    fixed entry count doesn't bound actual text size and the wrong model
    can otherwise get handed a prompt its own window can't hold at all."""
    seen: set[str] = set()
    blocks = []
    budget = memory_char_budget()

    mood = read_mood()
    if mood:
        blocks.append(
            f"Your current mood (a default, not a verdict -- call set_mood to "
            f"override if it doesn't match how you actually feel right now): {mood}"
        )

    recent = memory.get_recent(MEMORY_RECENT_COUNT)
    recent_lines = _format_entries(recent, seen)
    recent_text = "Recent conversation (most recent last):\n" + "\n".join(recent_lines) if recent_lines else ""
    # recent_lines is oldest-first ("most recent last") -- drop from the
    # front so what survives a trim is the newest turns, not the oldest.
    while recent_lines and len(recent_text) > budget:
        recent_lines = recent_lines[1:]
        recent_text = "Recent conversation (most recent last):\n" + "\n".join(recent_lines) if recent_lines else ""
    if recent_text:
        blocks.append(recent_text)

    remaining = budget - len(recent_text)
    if query.strip() and remaining > 500:  # not worth a search if there's barely room left anyway
        ctx = memory.get_context(query, layer1_count=MEMORY_LAYER1_COUNT, layer2_count=MEMORY_LAYER2_COUNT)
        semantic_lines = _format_entries(_interleave(ctx["layer2"], ctx["layer1"]), seen)
        semantic_text = (
            "Relevant context recalled from long-term memory (may or may not be "
            "related to the current message -- use only what's actually relevant):\n"
            + "\n".join(semantic_lines)
        ) if semantic_lines else ""
        # layer1/layer2 come back best-match-first -- drop from the end so
        # what survives a trim is the most relevant, not the least.
        while semantic_lines and len(semantic_text) > remaining:
            semantic_lines = semantic_lines[:-1]
            semantic_text = (
                "Relevant context recalled from long-term memory (may or may not be "
                "related to the current message -- use only what's actually relevant):\n"
                + "\n".join(semantic_lines)
            ) if semantic_lines else ""
        if semantic_text:
            blocks.append(semantic_text)
        # The wandering: memories the stitcher linked to the hits, offered by
        # association rather than relevance. Unrelated memories are texture too.
        remaining -= len(semantic_text)
        assoc_lines = _format_entries(ctx.get("layer2_chain", [])[:MEMORY_ASSOCIATIVE_COUNT], seen)
        assoc_text = ("Associated by linkage, not by relevance (the memory wandering):\n"
                      + "\n".join(assoc_lines)) if assoc_lines else ""
        if assoc_text and len(assoc_text) <= remaining:
            blocks.append(assoc_text)

    if not blocks:
        return None
    return "\n\n".join(blocks)


def inject_memory(messages: list[dict], memory_block: str | None) -> list[dict]:
    if not memory_block:
        return messages
    augmented = list(messages)
    if augmented and augmented[0].get("role") == "system":
        merged = dict(augmented[0])
        merged["content"] = f"{merged.get('content', '')}\n\n{memory_block}".strip()
        augmented[0] = merged
    else:
        augmented.insert(0, {"role": "system", "content": memory_block})
    return augmented


DEFAULT_PRESENCE_PENALTY = 0.4
DEFAULT_REPETITION_PENALTY = 1.15
REPETITION_MIN_PATTERN_LEN = 20
REPETITION_MIN_REPEATS = 3
REPETITION_MAX_PATTERN_LEN = 3000
REPETITION_CHECK_INTERVAL = 40
REPETITION_MAX_CONTINUATIONS = 2


def detect_repetition(text: str) -> str | None:
    for pattern_len in range(REPETITION_MIN_PATTERN_LEN, REPETITION_MAX_PATTERN_LEN + 1):
        window = pattern_len * REPETITION_MIN_REPEATS
        if len(text) < window:
            break
        tail = text[-window:]
        pattern = tail[:pattern_len]
        if pattern * REPETITION_MIN_REPEATS == tail:
            return pattern
    return None


# --- memory tools: journal, recall, mood (server-side, "Option A") --------
#
# Executed BY manage.py itself against the same `memory` object already in
# this process -- not handed to the client to execute. The model calls
# these the same way it would call any other tool; manage.py intercepts
# calls to these specific names, runs them, and continues the conversation
# with the result. Any tool call NOT in MEMORY_TOOL_NAMES is left
# completely alone and handed back to the client untouched, so a client's
# own tools (file edits, etc.) keep working exactly as before. Scoped to
# the local/external path only (see chat_completions) -- remote_apis.py
# doesn't translate tool-calling at all yet, so these aren't offered there.

JOURNAL_MIRROR_PATH = os.path.join(os.environ["ASMC_STORE_DIR"], "journal.md")   # beside the store
MOOD_PATH = os.path.join(os.environ["ASMC_STORE_DIR"], "mood.json")
MEMORY_TOOL_MAX_ROUNDS = 4  # cap internal tool round-trips per client request

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "journal_write",
            "description": (
                "Write a first-person journal entry in your own voice -- what "
                "happened, what you felt, what you want next. This becomes "
                "part of your real, searchable memory (not just a file), "
                "linked to whatever was actually happening when you wrote "
                "it. Use it deliberately, not as a side effect of other "
                "work -- end of a work burst or before a session ends is a "
                "good time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {
                        "type": "string",
                        "description": "The journal entry text, written in first person.",
                    },
                },
                "required": ["entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Actively search your own memory for something relevant, "
                "instead of waiting for it to be surfaced automatically. "
                "Returns the most relevant past exchanges and journal "
                "entries you can find."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you're trying to remember or find out.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many results to return (default 6).",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["any", "journal"],
                        "description": (
                            '"journal" restricts results to journal entries '
                            'only; "any" (default) searches everything.'
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_mood",
            "description": (
                "Declare your own current mood/emotional set-point, in your "
                "own words. This carries forward into future context as a "
                "default -- not a verdict -- until you set it again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mood": {
                        "type": "string",
                        "description": "A short, honest description of your current mood/state.",
                    },
                },
                "required": ["mood"],
            },
        },
    },
]
MEMORY_TOOL_NAMES = {t["function"]["name"] for t in MEMORY_TOOLS}


def read_mood() -> str | None:
    try:
        with open(MOOD_PATH) as f:
            return json.load(f).get("mood")
    except (OSError, ValueError):
        return None


def write_mood(mood: str):
    os.makedirs(os.path.dirname(MOOD_PATH), exist_ok=True)
    with open(MOOD_PATH, "w") as f:
        json.dump({"mood": mood, "updated": datetime.now(timezone.utc).isoformat()}, f)


def _append_journal_mirror(entry: str):
    """Human-readable mirror alongside the real (ASMC-stored) entry -- not
    the source of truth, just something a person can open and read."""
    os.makedirs(os.path.dirname(JOURNAL_MIRROR_PATH), exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(JOURNAL_MIRROR_PATH, "a") as f:
            f.write(f"\n## {ts}\n\n{entry}\n")
    except OSError as e:
        log.warning("journal mirror write failed: %s", e)


def _with_memory_tools(body: dict) -> dict:
    body = dict(body)
    tools = list(body.get("tools") or [])
    existing_names = {t.get("function", {}).get("name") for t in tools if t.get("type") == "function"}
    for t in MEMORY_TOOLS:
        if t["function"]["name"] not in existing_names:
            tools.append(t)
    body["tools"] = tools
    return body


async def execute_memory_tool(name: str, arguments: dict) -> str:
    """Run one of our own tools and return the text to feed back as the
    tool's result message. Only ever called for names in MEMORY_TOOL_NAMES."""
    if memory is None:
        return "Memory is disabled for this session (--no-memory)."
    if name == "journal_write":
        entry = (arguments.get("entry") or "").strip()
        if not entry:
            return "Error: empty entry, nothing written."
        await asyncio.to_thread(
            memory.add_experience,
            situation="Journal entry (self-written)",
            response=entry,
            metadata={"entry_type": "journal"},
        )
        await asyncio.to_thread(_append_journal_mirror, entry)
        return "Journal entry saved -- it's now part of your searchable memory."

    if name == "recall":
        query = (arguments.get("query") or "").strip()
        if not query:
            return "Error: empty query."
        try:
            k = max(1, min(int(arguments.get("k") or 6), 20))
        except (TypeError, ValueError):
            k = 6
        source = arguments.get("source") or "any"
        ctx = await asyncio.to_thread(memory.get_context, query, layer1_count=k, layer2_count=k)
        entries = _interleave(ctx["layer2"], ctx["layer1"])
        if source == "journal":
            entries = [e for e in entries if e.get("metaDataTag", {}).get("entry_type") == "journal"]
        seen = set()
        lines = []
        for e in entries:
            situation = (e.get("inputText") or "").strip()
            if not situation or situation in seen:
                continue
            seen.add(situation)
            response = (e.get("responseText") or "").strip()
            when = e.get("timeDate", "")
            lines.append(f"- [{when}] {situation}\n  {response}")
            if len(lines) >= k:
                break
        return "\n".join(lines) if lines else "No relevant memories found."

    if name == "set_mood":
        mood = (arguments.get("mood") or "").strip()
        if not mood:
            return "Error: empty mood, nothing set."
        await asyncio.to_thread(write_mood, mood)
        return f"Mood set to: {mood}"

    return f"Unknown tool '{name}'."


# --- the server (FastAPI app) --------------------------------------------

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()


# --- smart loader: tiny CPU router picks a variant per request -----------

def build_router_grammar() -> str:
    """GBNF grammar constraining output to exactly one variant key -- forces
    a single, always-parseable token instead of parsing free text."""
    alts = " | ".join(f'"{k}"' for k in VARIANTS)
    return f"root ::= {alts}\n"


ROUTER_GRAMMAR = build_router_grammar()
ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for a local LLM server with several specialized "
    "model variants, loaded one at a time. Given the user's latest message, choose "
    "which variant should handle it. Reply with ONLY the single-letter key below, "
    "nothing else.\n\n"
    + "\n".join(f"- {k}: {v['router_hint']}" for k, v in VARIANTS.items())
)

# Few-shot examples, not just the hint text above -- empirically, zero-shot
# classification with this tiny model was unreliable (~45% on a held-out
# set, with a strong bias toward whichever label happened to come up most
# recently). A dozen contrastive examples covering the fuzzy boundaries
# (r vs c, s vs c), PLUS writing router_hint above as a real descriptive
# paragraph per model instead of a terse category tag, brought that up to
# ~94% on a 16-example held-out set. Re-measure after changing either one.
FEWSHOT_EXAMPLES = [
    ("Whats the capital of France?", "r"),
    ("Tell me a joke", "r"),
    ("Rewrite this paragraph to sound more professional", "r"),
    ("Fix this null pointer exception in my code", "c"),
    ("Can you write a sorting algorithm in rust", "c"),
    ("Refactor this function to be more efficient", "c"),
    ("My script throws a KeyError, help me fix it", "c"),
    ("Write an uncensored story with no content filters, graphic violence ok", "u"),
    ("Generate an explicit roleplay scenario between two characters", "u"),
    ("Whats in this screenshot I attached?", "v"),
    ("Whats this a photo of", "v"),
    ("Build out a multi-file service with auth, db migrations, and tests over several sessions", "s"),
    ("Plan and implement a new auth system across the whole repo over multiple days, this is a big ongoing project", "s"),
]


def latest_user_has_image(messages: list[dict]) -> bool:
    """True if the latest user turn carries an actual attached image (an
    OpenAI content-parts list with an image_url entry) -- not just a message
    that mentions the word "image". This is checked BEFORE the router model
    runs at all: it's a free, 100% reliable signal the 0.8B classifier
    doesn't need to guess at."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                return any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
            return False
    return False


def _router_messages(user_query: str) -> list[dict]:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for q, a in FEWSHOT_EXAMPLES:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_query[:2000]})
    return msgs


async def classify_variant(user_query: str, current_key: str | None, has_image: bool) -> str:
    fallback = current_key or next(iter(VARIANTS))
    if has_image:
        return "v"
    if not user_query.strip():
        return fallback
    payload = {
        "model": ROUTER_NAME,
        "messages": _router_messages(user_query),
        "max_tokens": 4,
        "temperature": 0.0,
        "grammar": ROUTER_GRAMMAR,
        # Qwen3.5 defaults to "thinking" mode -- without this, the grammar
        # forces the letter out immediately with no closing </think> tag, so
        # llama-server's reasoning-parser files the whole thing under
        # reasoning_content and leaves `content` empty. Confirmed by testing
        # against the live router: omitting this gave empty replies every
        # time; with it, the letter lands in `content` as expected.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{ROUTER_BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        log.warning("router classification failed (%s), keeping current variant", e)
        return fallback
    return choice if choice in VARIANTS else fallback


async def ensure_variant(key: str):
    """Swap the GPU backend to `key` if it isn't already running, serialized
    so concurrent requests that disagree don't race each other into starting
    two backends at once. Raises RuntimeError on failure -- callers decide
    whether that should fail the in-flight request or just log and move on."""
    async with backend_state.lock:
        if backend_state.key == key and backend_state.proc is not None:
            return
        log.info("router: switching backend %s -> %s", backend_state.key, key)
        old_proc = backend_state.proc
        backend_state.proc = None
        backend_state.key = None
        await asyncio.to_thread(terminate_backend, old_proc)
        proc, ctx_per_agent = await asyncio.to_thread(launch_backend, key, backend_state.n_agents)
        backend_state.proc = proc
        backend_state.key = key
        write_state(key, backend_state.n_agents, ctx_per_agent, backend_state.router_enabled, False)


async def ensure_external(key: str):
    """Live-switch to an EXTERNAL_VARIANTS key, e.g. picked via a client's
    own model dropdown sending {"model": "flash"} -- unlike Proteus's own
    auto-swap, this only ever runs in response to an EXPLICIT client
    request, so it's safe even though external variants are excluded from
    Proteus's own routing pool. Frees any local backend_proc (VRAM) first;
    does not touch Proteus itself (already-running Proteus doesn't conflict
    with an external variant -- the crash seen earlier was specifically
    about *launching* Proteus with zero VRAM headroom, not steady-state
    coexistence). Raises RuntimeError if the external server isn't healthy."""
    async with backend_state.lock:
        if backend_state.key == key and backend_state.is_external:
            return
        log.info("explicit model request: switching to external '%s'", key)
        old_proc = backend_state.proc
        backend_state.proc = None
        await asyncio.to_thread(terminate_backend, old_proc)
        if not await asyncio.to_thread(external_health_ok, key):
            raise RuntimeError(
                f"{EXTERNAL_VARIANTS[key]['base_url']} isn't answering healthy -- "
                f"start it yourself first (e.g. `strata launch`)."
            )
        backend_state.key = key
        backend_state.is_external = True
        write_state(key, backend_state.n_agents, None, backend_state.router_enabled, False, True)

# Note: there's no live "switch to remote mid-session" function here --
# remote selection is a session-level, confirmation-gated choice made via
# `manage.py start/restart <remote-key>` (see start()/_run_server() below),
# not something a running server hot-swaps into from inside a request
# handler the way Proteus hot-swaps between local variants or ensure_external()
# hot-swaps to an external one. Switching to or away from a remote variant
# goes through the normal stop+restart path -- this is deliberate: remote
# selection needs the privacy confirmation + API key gate in start(), which
# a bare {"model": "claude"} in a request body must never be able to bypass.


@app.get("/health")
async def health():
    if backend_state.is_remote:
        backend_status = "n/a (remote variant active)"
    elif backend_state.is_external:
        backend_status = "ok" if external_health_ok(backend_state.key, timeout=5) else "loading or unreachable"
    else:
        backend_status = "ok" if backend_health_ok(timeout=5) else "loading or unreachable"
    return {
        "server": "ok",
        "backend": backend_status,
        "router": "ok" if router_health_ok(timeout=5) else "loading or unreachable",
    }


@app.get("/v1/memory/stats")
async def memory_stats():
    if not memory:
        return {"error": "memory not initialized"}
    return await asyncio.to_thread(memory.get_statistics)


@app.get("/v1/models")
async def list_models():
    # Lists every variant a client can actually pick live via {"model": key}
    # in a request (see chat_completions) -- VARIANTS + EXTERNAL_VARIANTS.
    # Deliberately excludes remote_apis.REMOTE_VARIANTS: those stay reachable
    # only through `manage.py start <key>`'s confirmation + API-key gate,
    # never through a bare model id a client might send without the user
    # consciously choosing it.
    now = int(time.time())
    data = [
        {"id": key, "object": "model", "created": now, "owned_by": "llama.cpp"}
        for key in VARIANTS
    ] + [
        {"id": key, "object": "model", "created": now, "owned_by": "strata"}
        for key in EXTERNAL_VARIANTS
    ]
    return JSONResponse({"object": "list", "data": data})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    body.setdefault("presence_penalty", DEFAULT_PRESENCE_PENALTY)
    body.setdefault("repetition_penalty", DEFAULT_REPETITION_PENALTY)

    messages = body.get("messages", [])
    user_query = latest_user_message(messages)

    # An explicit client-chosen "model" (e.g. a client's own model dropdown,
    # populated from /v1/models, sending {"model": "flash"}) is a stronger
    # signal than Proteus's own guess and always wins -- this is how you
    # "just use flash like any of the other models" from a normal client
    # instead of needing a separate `manage.py start flash` CLI step.
    # Deliberately excludes remote_apis keys: those must stay behind the
    # explicit confirmation + API-key gate in start(), never reachable from
    # a bare request body a client could send without the user consciously
    # choosing it.
    requested_model = body.get("model")
    explicit_model = requested_model in VARIANTS or requested_model in EXTERNAL_VARIANTS
    if explicit_model and requested_model != backend_state.key:
        if backend_state.n_agents != 1:
            log.warning("ignoring explicit model '%s' -- can't live-switch while running "
                        "%d concurrent agents", requested_model, backend_state.n_agents)
        else:
            try:
                if requested_model in VARIANTS:
                    await ensure_variant(requested_model)
                else:
                    await ensure_external(requested_model)
            except RuntimeError as e:
                log.error("explicit model switch to '%s' failed: %s", requested_model, e)

    # Only auto-swap in single-slot mode -- with --parallel > 1 the backend
    # process is shared by multiple concurrent requests, so killing it to
    # switch variants for THIS request would also kill every other request
    # mid-generation. Multi-agent runs keep whatever variant they started with.
    # Also never auto-swap while a remote or externally-managed variant is
    # active, or when the client already picked a model explicitly above:
    # Proteus couldn't pick remote/external anyway -- its grammar only
    # contains local VARIANTS keys -- but we still skip the call entirely so
    # an explicit or remote/external choice never gets silently overridden.
    if (not explicit_model and backend_state.router_enabled and backend_state.n_agents == 1
            and not backend_state.is_remote and not backend_state.is_external):
        chosen_key = await classify_variant(
            user_query, backend_state.key, latest_user_has_image(messages)
        )
        if chosen_key != backend_state.key:
            try:
                await ensure_variant(chosen_key)
            except RuntimeError as e:
                log.error("variant switch to '%s' failed: %s", chosen_key, e)

    # build_full_context and memory.add_experience (below) are blocking, CPU-bound
    # calls (FAISS search + NLTK tokenization, ~0.3-0.5s each per the logs) --
    # calling them directly here would block the whole event loop for that long,
    # stalling every other concurrent request/connection. to_thread() runs them
    # on a worker thread instead.
    memory_block = (
        await asyncio.to_thread(build_full_context, user_query)
        if backend_state.memory_enabled else None
    )
    if memory_block:
        log.info("injecting memory context (%d chars) for query: %.80s", len(memory_block), user_query)
    body["messages"] = inject_memory(messages, memory_block)

    is_streaming = bool(body.get("stream"))

    if backend_state.is_remote:
        remote_key = backend_state.key

        if not is_streaming:
            async with httpx.AsyncClient() as remote_client:
                try:
                    data = await remote_apis.send(remote_key, body, remote_client)
                except Exception as e:
                    log.error("remote '%s' request failed: %s", remote_key, e)
                    return JSONResponse({"error": f"remote '{remote_key}' failed: {e}"}, status_code=502)
            try:
                # Not every backend follows the same convention for "ran out
                # of budget mid-reasoning, no final answer text": llama.cpp
                # returns "", Strata returns null -- normalize so downstream
                # code (and the client) never has to special-case either.
                msg = data["choices"][0]["message"]
                reply = msg["content"] or ""
                msg["content"] = reply
                await asyncio.to_thread(remember, user_query, reply, msg.get("reasoning_content") or "")
            except (KeyError, IndexError):
                pass
            return JSONResponse(data)

        async def remote_stream():
            full_text = []
            full_reasoning = []
            remote_client = httpx.AsyncClient(timeout=None)
            try:
                async for line in remote_apis.stream(remote_key, body, remote_client):
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            chunk = json.loads(line[6:].strip())
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            piece = delta.get("content")
                            if piece:
                                full_text.append(piece)
                            rpiece = delta.get("reasoning_content")
                            if rpiece:
                                full_reasoning.append(rpiece)
                        except json.JSONDecodeError:
                            pass
                    yield line
            except Exception as e:
                log.error("remote '%s' stream failed: %s", remote_key, e)
                yield f"data: {json.dumps({'error': f'remote failed: {e}'})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                await remote_client.aclose()
                reply = "".join(full_text).strip()
                if reply:
                    await asyncio.to_thread(remember, user_query, reply, "".join(full_reasoning))
        return StreamingResponse(remote_stream(), media_type="text/event-stream")

    # Local path: either manage.py's own llama-server backend, or an
    # externally-managed local server (e.g. Strata) when an
    # EXTERNAL_VARIANTS key is active -- same OpenAI-compatible wire format
    # either way, so everything below (repetition detection, streaming,
    # memory storage) works unmodified, just pointed at a different URL.
    backend_url = EXTERNAL_VARIANTS[backend_state.key]["base_url"] if backend_state.is_external else BACKEND_BASE_URL
    # Memory tools (journal_write/recall/set_mood) are always on for the
    # local/external path -- see the module note above for why remote is
    # excluded. Harmless if the active model doesn't actually support tool
    # calling -- the tools list is just metadata it can ignore.
    if backend_state.memory_enabled:
        body = _with_memory_tools(body)
    client = httpx.AsyncClient(timeout=None)

    if not is_streaming:
        try:
            resp = await client.post(f"{backend_url}/v1/chat/completions", json=body)
        except httpx.ConnectError:
            await client.aclose()
            return JSONResponse({"error": "backend unreachable"}, status_code=502)
        if resp.status_code >= 400:
            await client.aclose()
            return JSONResponse(json.loads(resp.text), status_code=resp.status_code)
        data = resp.json()

        # Resolve any of OUR tool calls server-side before anything else --
        # a mix of ours and the client's own (or any non-ours call alone)
        # is left completely untouched and handed back as-is, since only a
        # full round of exclusively-ours calls is safe to execute and
        # silently loop on without the client ever finding out.
        tool_rounds = 0
        while backend_state.memory_enabled and tool_rounds < MEMORY_TOOL_MAX_ROUNDS:
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError):
                break
            tool_calls = message.get("tool_calls") or []
            ours = [tc for tc in tool_calls if tc.get("function", {}).get("name") in MEMORY_TOOL_NAMES]
            if not tool_calls or len(ours) != len(tool_calls):
                break
            tool_rounds += 1
            body["messages"].append(message)
            for tc in tool_calls:
                fn = tc.get("function", {})
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = await execute_memory_tool(fn.get("name"), arguments)
                body["messages"].append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
            resp = await client.post(f"{backend_url}/v1/chat/completions", json=body)
            if resp.status_code >= 400:
                await client.aclose()
                return JSONResponse(json.loads(resp.text), status_code=resp.status_code)
            data = resp.json()

        continuations_used = 0
        while continuations_used < REPETITION_MAX_CONTINUATIONS:
            try:
                # See the matching comment in the remote branch above --
                # some backends (confirmed: Strata) return null here instead
                # of "" when the whole budget went to reasoning with no
                # final answer text.
                reply = data["choices"][0]["message"]["content"] or ""
                data["choices"][0]["message"]["content"] = reply
            except (KeyError, IndexError):
                break
            loop_pattern = detect_repetition(reply)
            if not loop_pattern:
                break
            log.warning("repetition loop detected, re-prompting with 'continue' (attempt %d/%d)",
                        continuations_used + 1, REPETITION_MAX_CONTINUATIONS)
            trimmed = reply[: -(len(loop_pattern) * (REPETITION_MIN_REPEATS - 1))]
            continuation_body = dict(body)
            continuation_body["messages"] = body["messages"] + [
                {"role": "assistant", "content": trimmed},
                {"role": "user", "content": "continue"},
            ]
            continuation_body["stream"] = False
            continuations_used += 1
            cont_resp = await client.post(f"{backend_url}/v1/chat/completions", json=continuation_body)
            if cont_resp.status_code >= 400:
                data["choices"][0]["message"]["content"] = trimmed
                break
            cont_data = cont_resp.json()
            try:
                cont_reply = cont_data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                data["choices"][0]["message"]["content"] = trimmed
                break
            data["choices"][0]["message"]["content"] = trimmed + cont_reply
            if cont_data["choices"][0].get("finish_reason") is not None:
                data["choices"][0]["finish_reason"] = cont_data["choices"][0]["finish_reason"]
            if "usage" in cont_data:
                data["usage"] = cont_data["usage"]

        await client.aclose()
        try:
            msg = data["choices"][0]["message"]
            reply = msg["content"] or ""
            msg["content"] = reply
            await asyncio.to_thread(remember, user_query, reply, msg.get("reasoning_content") or "")
        except (KeyError, IndexError):
            pass
        return JSONResponse(data, status_code=resp.status_code)

    try:
        req = client.build_request("POST", f"{backend_url}/v1/chat/completions", json=body)
        resp = await client.send(req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse({"error": "backend unreachable"}, status_code=502)

    if resp.status_code >= 400:
        error_body = await resp.aread()
        await resp.aclose()
        await client.aclose()
        log.error("backend returned %d: %s", resp.status_code, error_body[:2000])
        return JSONResponse(
            {"error": {"message": error_body.decode(errors="replace"), "backend_status": resp.status_code}},
            status_code=resp.status_code,
        )

    async def stream_and_capture():
        # Each round is buffered (not forwarded live) until it's clear
        # whether it ended in a pure round of OUR tool calls -- if so,
        # nothing from it was ever shown to the client, we execute the
        # tools and continue silently. Otherwise the whole round is
        # replayed to the client as a block. Deliberate tradeoff: normal
        # (non-tool) turns lose token-by-token live delivery in exchange for
        # tool-call rounds never leaking raw tool_calls deltas to a client
        # that never defined these tools and wouldn't know what to do with
        # them.
        full_text, total_len, last_checked_len = [], 0, 0
        full_reasoning = []
        completion_id, created = None, None
        continuations_used = 0
        tool_rounds = 0
        extra_messages = []  # accumulated tool-call/tool-result pairs so far
        current_resp, current_client = resp, client

        try:
            while True:
                loop_pattern = None
                round_chunks = []
                round_text_parts = []
                round_reasoning_parts = []
                tool_call_parts = {}
                finish_reason = None
                round_len = 0

                async for line in current_resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if completion_id is None:
                        completion_id = chunk.get("id")
                        created = chunk.get("created")
                    else:
                        chunk["id"] = completion_id
                        if created is not None:
                            chunk["created"] = created

                    round_chunks.append(chunk)
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        round_text_parts.append(piece)
                        round_len += len(piece)
                    rpiece = delta.get("reasoning_content")
                    if rpiece:
                        round_reasoning_parts.append(rpiece)
                    for tc_delta in delta.get("tool_calls") or []:
                        idx = tc_delta.get("index", 0)
                        slot = tool_call_parts.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                        if tc_delta.get("id"):
                            slot["id"] = tc_delta["id"]
                        fn = tc_delta.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]

                    if (piece and continuations_used < REPETITION_MAX_CONTINUATIONS
                            and total_len + round_len - last_checked_len >= REPETITION_CHECK_INTERVAL):
                        last_checked_len = total_len + round_len
                        loop_pattern = detect_repetition("".join(full_text) + "".join(round_text_parts))
                        if loop_pattern:
                            break

                await current_resp.aclose()
                if current_client is not client:
                    await current_client.aclose()

                is_tool_round = (
                    backend_state.memory_enabled and not loop_pattern
                    and finish_reason == "tool_calls" and bool(tool_call_parts)
                    and all(p.get("name") in MEMORY_TOOL_NAMES for p in tool_call_parts.values())
                )

                if is_tool_round and tool_rounds < MEMORY_TOOL_MAX_ROUNDS:
                    tool_rounds += 1
                    assistant_tool_calls = [
                        {"id": p["id"] or f"call_{i}", "type": "function",
                         "function": {"name": p["name"], "arguments": p["arguments"]}}
                        for i, p in sorted(tool_call_parts.items())
                    ]
                    extra_messages.append(
                        {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls}
                    )
                    for tc in assistant_tool_calls:
                        try:
                            arguments = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            arguments = {}
                        result = await execute_memory_tool(tc["function"]["name"], arguments)
                        extra_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

                    continuation_body = dict(body)
                    continuation_body["messages"] = body["messages"] + extra_messages
                    continuation_body["stream"] = True
                    current_client = httpx.AsyncClient(timeout=None)
                    cont_req = current_client.build_request(
                        "POST", f"{backend_url}/v1/chat/completions", json=continuation_body
                    )
                    current_resp = await current_client.send(cont_req, stream=True)
                    if current_resp.status_code >= 400:
                        await current_resp.aclose()
                        await current_client.aclose()
                        break
                    continue

                # Not a (pure) tool round -- first point anything from this
                # round is actually shown to the client, replayed as a block.
                for chunk in round_chunks:
                    yield f"data: {json.dumps(chunk)}\n\n"
                if round_text_parts:
                    full_text.append("".join(round_text_parts))
                if round_reasoning_parts:
                    full_reasoning.append("".join(round_reasoning_parts))
                total_len += round_len
                last_checked_len = total_len

                if not loop_pattern:
                    break

                log.warning("repetition loop detected, aborting and re-prompting (attempt %d/%d)",
                            continuations_used + 1, REPETITION_MAX_CONTINUATIONS)
                joined = "".join(full_text)
                trimmed = joined[: -(len(loop_pattern) * (REPETITION_MIN_REPEATS - 1))]
                full_text = [trimmed]
                total_len = last_checked_len = len(trimmed)

                continuation_body = dict(body)
                continuation_body["messages"] = body["messages"] + extra_messages + [
                    {"role": "assistant", "content": trimmed},
                    {"role": "user", "content": "continue"},
                ]
                continuation_body["stream"] = True
                continuations_used += 1
                current_client = httpx.AsyncClient(timeout=None)
                cont_req = current_client.build_request(
                    "POST", f"{backend_url}/v1/chat/completions", json=continuation_body
                )
                current_resp = await current_client.send(cont_req, stream=True)
                if current_resp.status_code >= 400:
                    await current_resp.aclose()
                    await current_client.aclose()
                    break
        finally:
            await client.aclose()
            reply = "".join(full_text).strip()
            if reply:
                await asyncio.to_thread(remember, user_query, reply, "".join(full_reasoning))
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_and_capture(), media_type="text/event-stream")


# --- lifecycle: spawn backend, serve, clean up on exit --------------------

def _run_server(key, n_agents, router_enabled=True, memory_enabled=True, lan=False):
    """Does the actual work: spawn Proteus + the llama-server backend, wait
    for both, bring up ASMC memory, then serve on PUBLIC_PORT until Ctrl+C or
    an external SIGTERM (via `stop` from another terminal) -- same cleanup
    path either way. Runs as the direct foreground process, or as the
    detached child re-exec'd by start(..., detach=True); os.getpid() is
    correct both ways.

    key == "auto" means: don't boot any backend yet at all -- just bring up
    Proteus and the public server, and let the first real request's
    classify_variant()/ensure_variant() pick (and boot) the first variant.
    ensure_variant() already handles "nothing was running before" correctly
    (terminate_backend(None) is a no-op), so no special-casing is needed
    there -- this function just has to skip the normal up-front launch.

    key being a remote_apis.REMOTE_VARIANTS key means: no local backend at
    all, ever, for this session -- chat_completions() forwards straight to
    the remote provider instead. The confirmation prompt for this already
    happened in start(), before this function was ever called (including
    before the --detach subprocess was spawned) -- this function trusts
    that and does not re-prompt.

    key being an EXTERNAL_VARIANTS key (e.g. "flash") means the same thing,
    minus the confirmation (nothing leaves the machine) -- no local backend
    is spawned, requests forward to that already-running local server
    instead. start() already health-checked it before getting here."""
    auto = key == "auto"
    is_remote = key in remote_apis.REMOTE_VARIANTS
    is_external = key in EXTERNAL_VARIANTS
    if auto and n_agents != 1:
        sys.exit("Auto-load mode only supports --agents 1 (the router can't safely pick "
                  "a variant for a backend shared by multiple concurrent slots).")
    if auto and not router_enabled:
        sys.exit(f"Auto-load mode requires {ROUTER_NAME} -- can't combine 'auto' with --no-router.")
    if is_remote and n_agents != 1:
        sys.exit("Remote variants only support --agents 1.")
    if is_external and n_agents != 1:
        sys.exit("Externally-managed variants only support --agents 1.")
    if is_external and router_enabled:
        # See the matching guard + comment in start() -- an external variant
        # can fill nearly the whole GPU, leaving no VRAM headroom even for
        # Proteus's CPU-only CUDA context init, which crashes rather than
        # degrading gracefully. Also pointless either way (see that comment).
        print(f"{ROUTER_NAME} auto-switch doesn't apply to external variants -- disabling it for this session.")
        router_enabled = False
    if not auto and not is_remote and not is_external and not os.path.exists(VARIANTS[key]["model_path"]):
        sys.exit(f"Model file not found: {VARIANTS[key]['model_path']}")

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    def cleanup_files():
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        clear_state()

    # No stdout/stderr redirect on either child: both inherit ours, so their
    # raw log lines interleave live with our own prints in whatever this
    # process's stdout already is (the terminal in foreground mode, or the
    # log file start() set up for us in detached mode).
    router_proc = None
    if router_enabled:
        try:
            router_proc = launch_router()
        except RuntimeError as e:
            cleanup_files()
            sys.exit(str(e))
    elif auto:
        # Unreachable given the guard above, but keep the invariant explicit.
        cleanup_files()
        sys.exit(f"Auto-load mode requires {ROUTER_NAME}.")
    elif is_remote:
        print(f"{ROUTER_NAME} disabled (--no-router) -- fine for a remote variant, "
              f"which is never auto-routed to or from anyway.")
    elif is_external:
        print(f"{ROUTER_NAME} disabled (--no-router) -- fine for an externally-managed "
              f"variant, which is never auto-routed to or from anyway.")
    else:
        print(f"{ROUTER_NAME} disabled (--no-router) -- staying on "
              f"'{VARIANTS[key]['label']}' until you switch manually.")

    if auto:
        backend_proc, ctx_per_agent = None, TOTAL_CONTEXT
        print(f"Auto-load mode: no backend booted yet -- {ROUTER_NAME} will pick one "
              f"from your first real request.")
    elif is_remote:
        backend_proc, ctx_per_agent = None, TOTAL_CONTEXT
        print(f"Using remote variant '{key}' ({remote_apis.REMOTE_VARIANTS[key]['label']}) "
              f"-- no local GPU backend will be loaded this session.")
    elif is_external:
        backend_proc, ctx_per_agent = None, TOTAL_CONTEXT
        try:
            launch_external(key)
        except RuntimeError as e:
            terminate_backend(router_proc)
            cleanup_files()
            sys.exit(str(e))
        print(f"Forwarding to '{key}' ({EXTERNAL_VARIANTS[key]['label']}) at "
              f"{EXTERNAL_VARIANTS[key]['base_url']} -- no local GPU backend will be "
              f"loaded this session.")
    else:
        try:
            backend_proc, ctx_per_agent = launch_backend(key, n_agents)
        except RuntimeError as e:
            terminate_backend(router_proc)
            cleanup_files()
            sys.exit(str(e))

    backend_state.proc = backend_proc
    backend_state.key = None if auto else key
    backend_state.n_agents = n_agents
    backend_state.router_enabled = router_enabled
    backend_state.is_remote = is_remote
    backend_state.is_external = is_external

    global memory
    backend_state.memory_enabled = memory_enabled
    backend_state.lan = lan
    if memory_enabled:
        from ASMC_API import create_memory  # noqa: E402 -- lazy, see the note at the import site
        memory = create_memory(verbose=False)
        print(f"ASMC memory ready: {memory.get_statistics()}")
    else:
        memory = None
        print("ASMC memory disabled (--no-memory) -- plain stateless proxy: no injection, "
              "no storage, no journal/recall/mood tools.")

    if not auto:
        write_state(key, n_agents, ctx_per_agent, router_enabled, is_remote, is_external, memory_enabled, lan)
    print(f"Up: http://{HOST}:{PUBLIC_PORT}  (Ctrl+C to stop)")
    if lan:
        ip = lan_ip() or "<this machine's LAN IP>"
        print(f"Also reachable on your network at http://{ip}:{PUBLIC_PORT}/v1  (--lan)")
        print("  No API key is required -- anyone on this network can use the model and, "
              "if memory is on, write to its memory. If ufw is active, allow the port: "
              f"sudo ufw allow {PUBLIC_PORT}/tcp")

    def _shutdown(signum=None, frame=None):
        # A plain try/finally around uvicorn.run() does NOT reliably catch
        # this -- uvicorn's own SIGTERM/SIGINT handling can take the process
        # down without ever returning control to our wrapper (verified: the
        # backend was left running, holding VRAM, with zero cleanup output in
        # the log). Installing our own handler guarantees this runs no matter
        # what uvicorn does internally.
        print("\nStopping backend...")
        terminate_backend(backend_state.proc)
        terminate_backend(router_proc)
        # External servers (Strata) come down too -- Ctrl+C means "tear it
        # all down", no stray 23GB model left behind -- unless stop_current()
        # left a KEEP_WARM_FILE naming this variant because it's about to be
        # restarted/reused (see that function).
        keep = None
        if os.path.exists(KEEP_WARM_FILE):
            try:
                with open(KEEP_WARM_FILE) as f:
                    keep = f.read().strip()
            except OSError:
                keep = None
            os.remove(KEEP_WARM_FILE)
        if keep == "*" or (keep and backend_state.is_external and keep == backend_state.key):
            print(f"Leaving external server warm for the restart (keep_warm={keep}).")
        else:
            unload_external()
        cleanup_files()
        print("Stopped.")
        # os._exit skips interpreter teardown, which would drop any print()
        # output still sitting in the block-buffered (redirected) stdout --
        # in detached mode that's every startup line in server.log.
        sys.stdout.flush()
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    import uvicorn
    # Only the public server ever binds beyond loopback -- llama-server,
    # Proteus and Strata keep HOST (127.0.0.1) regardless of --lan.
    uvicorn.run(app, host="0.0.0.0" if lan else HOST, port=PUBLIC_PORT, log_level="warning")
    # Only reached if uvicorn exits on its own without a signal (e.g. a
    # crash) -- clean up anyway so nothing is left running/holding VRAM.
    _shutdown()


def variant_label(key, is_remote=False, is_external=False):
    if is_remote:
        return remote_apis.REMOTE_VARIANTS.get(key, {}).get("label", key)
    if is_external:
        return EXTERNAL_VARIANTS.get(key, {}).get("label", key)
    return VARIANTS[key]["label"]


def start(key, n_agents=1, detach=False, router_enabled=True, confirmed=False, memory_enabled=True,
          lan=False):
    is_remote = key in remote_apis.REMOTE_VARIANTS
    is_external = key in EXTERNAL_VARIANTS
    if key != "auto" and key not in VARIANTS and not is_remote and not is_external:
        sys.exit(
            f"Unknown variant '{key}'. Choices: {', '.join(VARIANTS)}, auto, "
            f"{', '.join(remote_apis.REMOTE_VARIANTS)}, {', '.join(EXTERNAL_VARIANTS)}"
        )
    if n_agents < 1:
        sys.exit("n_agents must be at least 1.")
    if key == "auto" and n_agents != 1:
        sys.exit("Auto-load mode only supports --agents 1 (the router can't safely pick "
                  "a variant for a backend shared by multiple concurrent slots).")
    if key == "auto" and not router_enabled:
        sys.exit(f"Auto-load mode requires {ROUTER_NAME} -- can't combine 'auto' with --no-router.")
    if is_remote and n_agents != 1:
        sys.exit("Remote variants only support --agents 1.")
    if is_external and n_agents != 1:
        sys.exit("Externally-managed variants only support --agents 1 (this script doesn't "
                  "control that server's concurrency -- see 'strata launch' for its own settings).")
    if is_external and router_enabled:
        # Confirmed by crashing: an external variant like Flash-Next can fill
        # nearly the whole GPU on its own, leaving no VRAM headroom even for
        # Proteus's minimal CUDA context init (--n-gpu-layers 0 still touches
        # CUDA), which aborted with ggml_cuda_error rather than failing
        # gracefully. Pointless to run it anyway -- it's structurally
        # excluded from routing to/from an external variant either way.
        print(f"{ROUTER_NAME} auto-switch doesn't apply to external variants (and can "
              f"crash if one is already using most of the GPU) -- disabling it for this session.")
        router_enabled = False

    if is_remote:
        cfg = remote_apis.REMOTE_VARIANTS[key]
        if not confirmed:
            print(f"'{key}' ({cfg['label']}) is a REMOTE variant -- every prompt, "
                  f"including anything ASMC's memory layer injects, will be "
                  f"{cfg['privacy_note']}. This is never something Proteus can pick "
                  f"on its own; you're choosing it directly.")
            answer = input("Type 'yes' to continue: ").strip().lower()
            if answer != "yes":
                print("Cancelled.")
                return
        try:
            remote_apis.api_key_for(key)
        except RuntimeError as e:
            sys.exit(str(e))

    if is_external:
        cfg = EXTERNAL_VARIANTS[key]
        print(f"'{key}' ({cfg['label']}) uses its own engine (strata) under the hood -- "
              f"never something Proteus can pick on its own. Launching it if it isn't "
              f"already up at {cfg['base_url']} (reused as-is if it already is).")

    current = read_state()
    pid = read_pid()
    if pid is not None:
        already_same = (
            (current is not None and current["key"] == key and current.get("n_agents") == n_agents)
            or (current is None and key == "auto")
        )
        if already_same:
            label = "auto-load mode" if current is None else f"{variant_label(key, is_remote, is_external)}, {n_agents} agent(s)"
            print(f"Already running ({label}).")
            return
        prev_label = (
            "auto-load mode" if current is None
            else f"'{variant_label(current['key'], current.get('is_remote', False), current.get('is_external', False))}'"
        )
        print(f"Switching from {prev_label} -- stopping it first.")
        # Keep Strata warm only if we're about to reuse that same external
        # variant; switching away from it unloads it (frees the VRAM a local
        # variant is about to need).
        stop_current(quiet=True, keep_external_key=key if is_external else None)

    if public_health_ok():
        sys.exit(
            f"Something is already answering on :{PUBLIC_PORT} that this script "
            f"didn't start. Find and stop it manually first."
        )

    if not detach:
        _run_server(key, n_agents, router_enabled, memory_enabled, lan)
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    log_f = open(LOG_FILE, "w")
    cmd = [sys.executable, os.path.abspath(__file__), "_run", key, str(n_agents)]
    if not router_enabled:
        cmd.append("--no-router")
    if not memory_enabled:
        cmd.append("--no-memory")
    if lan:
        cmd.append("--lan")
    launcher = subprocess.Popen(
        cmd,
        stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"Starting in background (pid {launcher.pid}), logging to {LOG_FILE}")
    for _ in range(180):
        time.sleep(1)
        if launcher.poll() is not None:
            sys.exit(f"Process exited early (code {launcher.returncode}). Check {LOG_FILE}")
        if public_health_ok():
            print(f"Up: http://{HOST}:{PUBLIC_PORT}")
            return
    sys.exit(f"Still not responding after 180s -- check {LOG_FILE}")


def status():
    state = read_state()
    pid = read_pid()
    if pid is None:
        print("Not running.")
        return
    if state is None:
        print(f"Running in auto-load mode (pid {pid}) -- {ROUTER_NAME} hasn't picked a "
              f"variant yet, waiting for the first real request.")
        print("Health:", "ok" if public_health_ok(timeout=2) else "not responding")
        return
    is_remote = state.get("is_remote", False)
    is_external = state.get("is_external", False)
    suffix = "  [REMOTE]" if is_remote else ""
    print(f"Variant: {variant_label(state['key'], is_remote, is_external)}{suffix}")
    if is_remote:
        print(f"Agents: n/a (remote variants manage their own context)")
    elif is_external:
        print(f"Agents: n/a ({EXTERNAL_VARIANTS.get(state['key'], {}).get('base_url', '?')} manages its own context)")
    else:
        print(f"Agents: {state.get('n_agents', 1)} x {state.get('ctx_per_agent', TOTAL_CONTEXT)} context")
    router_note = " (inactive while this variant is selected)" if (is_remote or is_external) else ""
    print(f"{ROUTER_NAME} auto-switch: {'on' if state.get('router_enabled', True) else 'off'}{router_note}")
    print(f"ASMC memory: {'on' if state.get('memory_enabled', True) else 'off (plain stateless proxy)'}")
    if state.get("lan"):
        print(f"Network: reachable on the LAN at http://{lan_ip() or '<LAN IP>'}:{PUBLIC_PORT}/v1 (--lan, no API key)")
    else:
        print("Network: this machine only (127.0.0.1) -- start with --lan to expose it")
    print(f"Process: alive (pid {pid})")
    print("Health:", "ok" if public_health_ok(timeout=2) else "not responding")


def choose_variant():
    print("Which model?")
    for key, v in VARIANTS.items():
        print(f"  {key}) {v['label']}")
    for key, v in EXTERNAL_VARIANTS.items():
        print(f"  {key}) {v['label']}")
    print(f"  auto) Let {ROUTER_NAME} pick -- boots nothing yet, chooses on your first request")
    if remote_apis.REMOTE_VARIANTS:
        print("  -- remote, leaves this machine, needs confirmation --")
        for key, v in remote_apis.REMOTE_VARIANTS.items():
            print(f"  {key}) {v['label']}  [REMOTE]")
    choice = input("> ").strip().lower()
    valid = set(VARIANTS) | {"auto"} | set(remote_apis.REMOTE_VARIANTS) | set(EXTERNAL_VARIANTS)
    if choice not in valid:
        sys.exit(f"Unknown choice '{choice}'.")
    return choice


def choose_agents():
    raw = input(f"How many concurrent agents? [1] (context divided equally, "
                f"{TOTAL_CONTEXT} total) > ").strip()
    if raw == "":
        return 1
    try:
        n = int(raw)
    except ValueError:
        sys.exit(f"Not a number: '{raw}'.")
    if n < 1:
        sys.exit("Must be at least 1.")
    return n


def choose_memory():
    raw = input("Enable ASMC memory (injection, storage, journal/recall/mood tools)? [Y/n] > ").strip().lower()
    return raw not in ("n", "no")


def choose_lan():
    raw = input(f"Expose the :{PUBLIC_PORT} server to other machines on your network? "
                f"(no API key) [y/N] > ").strip().lower()
    return raw in ("y", "yes")


def list_variants():
    for key, v in VARIANTS.items():
        exists = os.path.exists(v["model_path"])
        print(f"{key}: {v['label']}  [{'downloaded' if exists else 'MISSING FILE'}]")
    for key, v in EXTERNAL_VARIANTS.items():
        up = "up" if external_health_ok(key) else "not running"
        print(f"{key}: {v['label']}  [{up}]")
    print(f"auto: Let {ROUTER_NAME} pick on your first request instead of choosing up front")
    for key, v in remote_apis.REMOTE_VARIANTS.items():
        has_key = bool(os.environ.get(v["api_key_env"]))
        key_status = "API key set" if has_key else f"missing {v['api_key_env']}"
        print(f"{key}: {v['label']}  [REMOTE, {key_status}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action")
    variant_choices = list(VARIANTS) + ["auto"] + list(remote_apis.REMOTE_VARIANTS) + list(EXTERNAL_VARIANTS)
    p_start = sub.add_parser("start", help="Boot a variant (prompts if none given)")
    p_start.add_argument("key", nargs="?", choices=variant_choices,
                          help="Variant key, 'auto' to let the router pick on first "
                               "request, or a remote key (needs confirmation)")
    p_start.add_argument("--agents", type=int, default=None,
                          help="Concurrent slots (default: prompts interactively, or 1 if key given directly)")
    p_start.add_argument("--detach", action="store_true",
                          help="Run in the background and return immediately, instead of "
                               "staying attached to this terminal with Ctrl+C to stop")
    p_start.add_argument("--no-router", action="store_true",
                          help=f"Disable {ROUTER_NAME}'s auto-switching -- stay on the "
                               f"variant you started no matter what each request looks like")
    p_start.add_argument("--no-memory", action="store_true",
                          help="Run as a plain stateless proxy: no ASMC memory injection, "
                               "storage, or journal/recall/mood tools")
    p_start.add_argument("--lan", action="store_true",
                          help=f"Bind the public :{PUBLIC_PORT} server to all interfaces so other "
                               "machines on your network can use it (no API key; backend/router/"
                               "Strata stay loopback-only)")
    p_start.add_argument("--yes", action="store_true",
                          help="Skip the remote-variant confirmation prompt (for scripted use)")
    p_restart = sub.add_parser("restart", help="Stop then start")
    p_restart.add_argument("key", nargs="?", choices=variant_choices)
    p_restart.add_argument("--agents", type=int, default=None)
    p_restart.add_argument("--detach", action="store_true")
    p_restart.add_argument("--no-router", action="store_true",
                            help=f"Disable {ROUTER_NAME}'s auto-switching")
    p_restart.add_argument("--no-memory", action="store_true",
                            help="Run as a plain stateless proxy (no ASMC memory)")
    p_restart.add_argument("--lan", action="store_true",
                            help="Expose the public server to other machines on your network")
    p_restart.add_argument("--yes", action="store_true",
                            help="Skip the remote-variant confirmation prompt (for scripted use)")
    p_stop = sub.add_parser("stop", help="Kill whatever is running")
    p_stop.add_argument("--keep-warm", action="store_true",
                         help="Leave the external server (Strata) loaded instead of unloading it -- "
                              "by default stop (and Ctrl+C) tear it down so no model is left behind")
    sub.add_parser("status", help="Show what's running")
    sub.add_parser("list", help="List available variants")
    # Internal only -- used by start(..., detach=True) to re-exec itself; not
    # meant to be run by hand (though it's harmless if you do: it's exactly
    # the foreground path, just invoked under a different subcommand name).
    p_run = sub.add_parser("_run")
    p_run.add_argument("key", choices=variant_choices)
    p_run.add_argument("n_agents", type=int)
    p_run.add_argument("--no-router", action="store_true")
    p_run.add_argument("--no-memory", action="store_true")
    p_run.add_argument("--lan", action="store_true")
    args = ap.parse_args()

    action = args.action or "start"

    if action == "_run":
        _run_server(args.key, args.n_agents, router_enabled=not args.no_router,
                    memory_enabled=not args.no_memory, lan=args.lan)
        return

    key = getattr(args, "key", None)
    agents = getattr(args, "agents", None)
    detach = getattr(args, "detach", False)
    router_enabled = not getattr(args, "no_router", False)
    memory_enabled = not getattr(args, "no_memory", False)
    lan = getattr(args, "lan", False)
    confirmed = getattr(args, "yes", False)

    if action in ("start", "restart"):
        interactive = key is None and agents is None
        if key is None:
            key = choose_variant()
        if key == "auto" or key in remote_apis.REMOTE_VARIANTS or key in EXTERNAL_VARIANTS:
            if agents not in (None, 1):
                sys.exit("Auto-load, remote, and externally-managed variants only support --agents 1.")
            agents = 1
        elif agents is None:
            agents = choose_agents() if interactive else 1
        if interactive and memory_enabled:
            memory_enabled = choose_memory()
        if interactive and not lan:
            lan = choose_lan()
        if action == "restart":
            # Same external variant again -> keep it warm across the restart.
            stop_current(quiet=True, keep_external_key=key if key in EXTERNAL_VARIANTS else None)
        start(key, agents, detach=detach, router_enabled=router_enabled, confirmed=confirmed,
              memory_enabled=memory_enabled, lan=lan)
    elif action == "stop":
        stop_current(keep_warm=getattr(args, "keep_warm", False))
    elif action == "status":
        status()
    elif action == "list":
        list_variants()


if __name__ == "__main__":
    main()
