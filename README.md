# Laya Decision API

A self-hosted HTTP API for [Laya](https://github.com/NandhaKishorM/laya). It answers typed questions about a text (`choice`, `score`, yes/no probability) in one forward pass, in 100+ languages. All three checkpoints from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) ship inside this repo, so it runs with no access to Hugging Face.

| Checkpoint | Used for | Weights |
|---|---|---:|
| `english` | English text (default) | 843 MB |
| `multilingual` | Non-English text, picked automatically by language | 644 MB |
| `typed-decisions` | Fine-tuned for triage workflows; request it with `"model": "typed-decisions"` | 843 MB |

Weights are pinned to Hugging Face revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Their SHA-256 values match laya's own `verify/checkpoints.json`.

## Quick start on a new machine

Requires Python 3.10–3.13 and git.

```bash
# 1. Clone the repo (includes the model, split into parts under model_chunks/)
git clone <this-repo-url> laya_api
cd laya_api

# 2. Rebuild the model: verifies every part, writes models/laya/, verifies the result
python3 model_chunks.py join

# 3. Install dependencies
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Start the API
uvicorn main:app --host 0.0.0.0 --port 8090
```

Startup loads all three checkpoints (about 20–60 s). Check it is up:

```bash
curl http://localhost:8090/health
# {"status":"ok","loaded":["english","multilingual","typed-decisions"],...}
```

Step 2 is needed only once per clone. If it prints `Verification FAILED`, a part is missing or corrupted; run `git pull` or clone again, then rerun `join`.

## Example

```bash
curl -s localhost:8090/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"body": "We were billed twice for March. Please refund it today or we will cancel."},
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}},
    "urgency":    {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
  }
}'
```

Non-English text is routed to `multilingual` automatically. Add `"model": "english" | "multilingual" | "typed-decisions"` to choose a checkpoint yourself.

Endpoints (laya's own server, `laya.serve`): `GET /health`, `POST /v1/systemone`, `POST /v1/systemone/batch` (up to 64 states, in a `states` array), and the Swagger UI at `/docs`.

Environment variables:
- `LAYA_DEVICE`: `cpu`, `cuda` or `mps` (default: best available)
- `LAYA_MODELS`: comma list of checkpoints to load at startup (default: all three)
- `LAYA_AUTO_TASK=1`: route requests that match a typed-decisions workflow to that checkpoint
- `LAYA_API_KEY`: if set, clients must send `Authorization: Bearer <key>`
- `MODEL_DIR`: model folder (default `models/laya`)

`main.py` sets `HF_HUB_OFFLINE=1`, so a missing local file fails at startup instead of being downloaded.

## How the model is stored

GitHub rejects files over 100 MB, so each `model.safetensors` is split into 45 MiB parts in `model_chunks/`. Small files (configs, tokenizers) are kept whole in `model_chunks/files/`. `model_chunks/manifest.json` records the SHA-256 of every part and every original file.

```bash
python3 model_chunks.py verify   # check the parts against the manifest
python3 model_chunks.py join     # model_chunks/ -> models/laya (checks before and after)
python3 model_chunks.py split    # models/laya -> model_chunks/ (after a model update)
```

To refresh the model from Hugging Face: `python download_model.py`, then `python3 model_chunks.py split`. Run `split` before the first server start, because laya can rewrite `tokenizer_config.json` when it loads.

## Pushing to GitHub

The parts total 2.2 GB and GitHub refuses a single push over 2 GB, so push one checkpoint at a time:

```bash
git remote add origin <your-repo-url>

git add .gitignore README.md requirements.txt main.py download_model.py model_chunks.py \
        model_chunks/manifest.json model_chunks/files model_chunks/model.safetensors.part*
git commit -m "Add Laya API and english checkpoint"
git push -u origin main

git add model_chunks/multilingual__*
git commit -m "Add multilingual checkpoint"
git push

git add model_chunks/typed-decisions__*
git commit -m "Add typed-decisions checkpoint"
git push
```

Until all three pushes are done, `join` on a fresh clone fails its check for the missing parts.
