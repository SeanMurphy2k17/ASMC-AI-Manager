# ASMC AI Manager

A local, OpenAI-compatible LLM server with long-term memory built in.

One script, `manage.py`, gives every client (VS Code, Copilot-style extensions, scripts, `curl`)
a single endpoint, `http://127.0.0.1:18090/v1`. Behind it, it:

- runs and swaps local GGUF models through **llama.cpp**, one at a time on the GPU;
- can let a tiny CPU model, **Proteus**, pick the best model for each request and swap to it;
- can hand requests to an **external** OpenAI-compatible server you run yourself, or to a
  **remote API** (Anthropic, OpenAI, a Qwen endpoint), on your explicit choice;
- gives every conversation memory with **[ASMC](https://github.com/SeanMurphy2k17/AdvancedSemanticMemoryClustering-ASMC)**
  (Advanced Semantic Memory Clustering): recent turns and semantically related past exchanges
  are added to each request, every exchange is stored, and the model gets memory tools of its
  own (`recall`, `journal_write`, `set_mood`).

Memory stays on your machine. Run with `--no-memory` and it is a plain pass-through proxy.

## How a request flows

```
client ──POST /v1/chat/completions──▶ manage.py  (:18090, FastAPI)
                                        │  1. memory: recent turns + ASMC recall + mood,
                                        │     added to the system prompt
                                        │  2. model: the one asked for, or Proteus's pick,
                                        │     swapped in if another is loaded
                                        ▼
               llama-server (:18080)  │  external server  │  remote API
                                        │
                                        │  3. memory tool calls are answered here and the
                                        │     model continues; the reply streams back
                                        ▼  4. the exchange is stored in ASMC
client ◀────────────────────────────────┘
```

`llama-server` and Proteus (:18070) listen on loopback only; clients only ever talk to :18090.

## Requirements

- Python 3.10 or newer (tested with 3.12).
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with GPU support, for its
  `llama-server` binary (`-DGGML_CUDA=ON` for NVIDIA; Vulkan, ROCm and Metal builds work too).
- A GPU with enough memory for the models you choose (see "Picking models that fit").
- GGUF model files.
- Optional: an external OpenAI-compatible server (the default lineup includes one served by
  Strata), and API keys for any remote API you want to use.

## Install

```bash
git clone --recurse-submodules https://github.com/SeanMurphy2k17/ASMC-AI-Manager.git
cd ASMC-AI-Manager
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`, fetch ASMC with `git submodule update --init`.

`manage.py` re-runs itself inside `.venv/` when that folder exists, so after this
`python3 manage.py ...` is enough. The first start with memory on downloads a few small NLTK
datasets that ASMC uses (WordNet, SentiWordNet, a tokenizer), so it needs a network connection
once.

## Configure

Everything machine-specific is read from the environment, or from an optional
`local_settings.json` beside `manage.py` (kept out of git). A variable set in the environment
wins over the file.

```json
{
  "LLAMA_SERVER": "/path/to/llama.cpp/build/bin/llama-server",
  "ASMC_AI_MODELS_DIR": "/path/to/models"
}
```

| Setting | Default | What it is |
|---|---|---|
| `LLAMA_SERVER` | `llama-server` on your PATH | The llama.cpp server binary. |
| `ASMC_AI_MODELS_DIR` | `models/` | Where the GGUF files are. |
| `ASMC_PATH` | `asmc/` (the submodule) | Where ASMC is, if you keep it elsewhere. |
| `ASMC_STORE_DIR` | `memory/` | Where the memory store, journal and mood are kept. |
| `ANTHROPIC_API_KEY`, `CLAUDE_REMOTE_MODEL` | unset, `claude-sonnet-5-5` | The `claude` remote variant. |
| `OPENAI_API_KEY`, `OPENAI_REMOTE_MODEL` | unset, `gpt-5` | The `openai` remote variant. |
| `QWEN_REMOTE_API_KEY`, `QWEN_REMOTE_MODEL`, `QWEN_REMOTE_BASE_URL` | unset, `qwen-max`, DashScope | The `qwen-remote` variant (any OpenAI-compatible endpoint). |

Ports and the context budget are constants near the top of `manage.py`: `PUBLIC_PORT` (18090),
`BACKEND_PORT` (18080), `ROUTER_PORT` (18070) and `TOTAL_CONTEXT` (131072).

## Models

The lineup is the `VARIANTS` dictionary in `manage.py`. Each entry names a GGUF file in the
models folder, a label, and a `router_hint` that tells Proteus what the model is good at. The
default lineup:

| Key | Model | File in the models folder |
|---|---|---|
| `r` | Qwen3.6-35B-A3B, general purpose | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` |
| `u` | Qwen3.6-35B-A3B "Heretic" uncensored | `Qwen3.6-35B-A3B-uncensored-heretic-Native-MTP-Preserved-Q4_K_M.gguf` |
| `s` | CoderX, long multi-step coding | `Qwen3.6-27B-A3B-CoderX-Q5_K_M.gguf` |
| `v` | Qwen3.8-27B with a vision tower (images) | `Qwen3.8-27B-UD-Q5_K_M.gguf` and `mmproj-F16.gguf` |
| `c` | Coder390, the default coder | `coder390/GGUF/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-MTP.gguf` |
| Proteus | Qwen3.5-0.8B, the router (CPU) | `router/Qwen3.5-0.8B-Q8_0.gguf` |

Download with Hugging Face's tool, for example:

```bash
.venv/bin/hf download <repo> <file.gguf> --local-dir models/
```

Swap in your own models by editing `VARIANTS`; only the files you list need to exist.
`manage.py list` shows which ones are present.

Two more kinds of variant sit beside the local ones:

- **External** (`EXTERNAL_VARIANTS`, default `flash`): an OpenAI-compatible server that runs on
  this machine but is started and stopped by its own tool (the default entry launches
  Qwen3.8-Flash-Next with Strata on :8080). `manage.py` starts it if it is not running and
  unloads it on exit unless asked to keep it warm. It is never in Proteus's pool, since it needs
  the whole GPU.
- **Remote** (`remote_apis.py`: `claude`, `openai`, `qwen-remote`): an outside API. These are
  chosen only with `manage.py start <key>`, which asks for confirmation and needs the API key.
  A request cannot switch to one by naming it, because the conversation, including anything
  the memory layer adds, would be sent to that service.

## Run

```bash
python3 manage.py                    # interactive: choose a model, memory on or off, the router
python3 manage.py start c            # start Coder390 in the foreground (Ctrl+C stops everything)
python3 manage.py start auto         # let Proteus choose on the first request
python3 manage.py start r --agents 4 # four concurrent slots, each with a quarter of the context
python3 manage.py start c --detach   # in the background, logging to logs/server.log
python3 manage.py start r --no-router  # stay on this model; no per-request switching
python3 manage.py start r --no-memory  # a plain proxy: no memory added, stored or offered
python3 manage.py start r --lan      # listen on all interfaces (see the warning below)
python3 manage.py status             # what is running, with which settings
python3 manage.py list               # the lineup, and which files and keys are present
python3 manage.py restart [key]      # stop, then start again
python3 manage.py stop               # stop the server, the backend, the router and any external model
python3 manage.py stop --keep-warm   # stop, but leave an external model loaded
```

Only one GPU model runs at a time; starting another stops the first. `--yes` skips the
confirmation for a remote variant.

**`--lan` has no authentication.** Anyone who can reach the port can use the models and read
and add to the memory through them. Use it only on a network you trust.

## Use it

Point any OpenAI-compatible client at `http://127.0.0.1:18090/v1`.

```bash
curl http://127.0.0.1:18090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming or not. `"model"` may name a local or external variant to use it; otherwise the running one answers, or Proteus picks when the router is on. |
| `GET /v1/models` | The local and external variants a request can name. |
| `GET /v1/memory/stats` | The memory store's statistics. |
| `GET /health` | The server, backend and router state. |

## Memory

With memory on, each request gets, in its system prompt:

- the last 8 exchanges, for continuity;
- up to 5 recent and 6 long-term exchanges that ASMC finds related to the new message;
- the mood the model last set for itself.

How much is added is trimmed to the model's context. After the reply, the exchange is stored.
The model is offered three tools, which the server carries out itself before the model
continues:

| Tool | What it does |
|---|---|
| `recall` | Searches the memory on purpose. |
| `journal_write` | Writes a journal entry into the memory, mirrored to `journal.md` beside the store. |
| `set_mood` | Records a mood that carries forward as a default. |

The store, the journal and the mood live in `memory/` (or `ASMC_STORE_DIR`) and nowhere else.
Only one program should use a store at a time.

## Picking models that fit

`llama-server` uses quantized weights and, here, a quantized KV cache (`--cache-type-k/v q4_0`),
which is why a 27B to 35B model fits on one consumer GPU at long context. `BASE_ARGS` in
`manage.py` holds the flags applied to every local variant:

| Flag | Why |
|---|---|
| `--n-gpu-layers 999` | Everything on the GPU. If a model does not fit, use a smaller quant rather than spilling to the CPU. |
| `--cache-type-k/-v q4_0` | Quantizes the KV cache, the biggest lever for long context. |
| `--batch-size 512 --ubatch-size 256` | Sizes the compute buffers, separately from context. An out-of-memory error on the first request, not at load, means shrink these. |
| `--ctx-size` | Set per run as `TOTAL_CONTEXT / agents`: llama.cpp's context is per slot, so concurrency divides one budget. |

Rule of thumb: the model file plus 2 to 4 GB must fit in the GPU memory you have free
(`nvidia-smi --query-gpu=memory.total,memory.used --format=csv`). Reasoning stays on: the
models in the default lineup were benchmarked with it, so a single hard turn can use tens of
thousands of tokens; one agent with the full context suits that best.

## License

MIT. See [LICENSE](LICENSE). ASMC is its own project under its own license.
