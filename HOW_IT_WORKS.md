# How laya-api works

## What a checkpoint is

A **checkpoint** is one complete, trained copy of a model saved to disk: its learned weights plus the small files needed to use them. Laya publishes three separately trained models, and each one is a checkpoint.

In `main.py`, `CHECKPOINTS` maps each checkpoint's name to the folder it lives in:

```python
CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
```

On disk that is:

```
models/laya/                      ← "english" (None = the root folder itself)
├── model.safetensors   (843 MB)    the trained weights
├── rl_agent_config.json            settings: which base model, how many tokens, etc.
├── tokenizer/                      turns text into numbers the model reads
├── encoder/                        shape of the underlying model (e.g. ModernBERT-large)
├── multilingual/                 ← "multilingual" checkpoint, same 4 items (644 MB)
└── typed-decisions/              ← "typed-decisions" checkpoint, same 4 items (843 MB)
```

The three differ in what they were trained for:

| Checkpoint | Base model | Good at |
|---|---|---|
| `english` | ModernBERT-large | English text, general questions |
| `multilingual` | mmBERT-base | 100+ languages (Hindi, Spanish, …) |
| `typed-decisions` | ModernBERT-large, fine-tuned further | Support, security, invoice and agent-trace triage questions |

## What happens, step by step

### 1. Rebuild the model folder (once per clone)

`python3 model_chunks.py join` turns the 45 MiB pieces in `model_chunks/` back into `models/laya/`, checking SHA-256 checksums before and after. See "How the model is stored" in the README.

### 2. Start up (`uvicorn main:app`)

`main.py` does four things, top to bottom:

- **Blocks downloads.** It sets `HF_HUB_OFFLINE=1`, so if a file is missing the library fails instead of quietly downloading it from Hugging Face.
- **Checks the weights exist.** `build_router()` looks for each `model.safetensors` and stops with "run `model_chunks.py join` first" if one is missing.
- **Builds a Router.** A Router is Laya's dispatcher: it holds the three models and decides which one answers each request. Out of the box it would download them from Hugging Face; `main.py` points it at local folders instead:

  ```python
  models={name: (str(MODEL_DIR), sub) for name, sub in CHECKPOINTS.items()}
  # → {"english": ("/…/models/laya", None), "multilingual": ("/…/models/laya", "multilingual"), …}
  ```

- **Loads all three into memory.** `router.preload()` does this at startup so no request waits for a model to load. That is why startup takes 20–60 s. Set `LAYA_MODELS` (for example `LAYA_MODELS=english,multilingual`) to load fewer.

Finally, `create_app(router)` hands the Router to Laya's own web server. This repo does not define any endpoints; `/health`, `/v1/systemone` and `/v1/systemone/batch` come from the `laya` library (`laya.serve`).

### 3. Handle a request (`POST /v1/systemone`)

You send a text (`state`) and some typed questions:

- `choice`: pick one option (e.g. billing / technical / other)
- `score`: a level on a scale (e.g. not urgent → blocking)
- `noul`: the probability that the answer is yes

The Router then picks a checkpoint:

- If the request sets `"model"`, it uses that one.
- Otherwise it looks at the writing system and language: English goes to `english`, anything else goes to `multilingual`.
- `typed-decisions` is used only when you ask for it by name, or when the server runs with `LAYA_AUTO_TASK=1` and the questions match one of its four known workflows.

The chosen model answers **every question in a single pass** over the text. It does not generate text word by word like a chatbot, which is why it is fast (tens of milliseconds on a GPU). The response holds each answer with its probabilities, plus a `routing` field saying which checkpoint answered and why.

Example results from a local test:

| Input | Checkpoint picked | Department |
|---|---|---|
| English billing complaint | `english` | billing (churn risk 0.88) |
| Hindi billing text | `multilingual` | billing |
| Spanish "app crashes" | `multilingual` | technical |
| Outage text, `"model": "typed-decisions"` | `typed-decisions` | technical |

## In short

The repo stores three trained models in pieces, `join` reassembles them, and `main.py` loads all three and lets Laya's server send each request to the right one.
