"""Laya decision API, served from the local model folder with no Hugging Face access.

The checkpoints are read from ./models/laya (rebuilt by `python3 model_chunks.py join`).
The HTTP API itself is laya's own server (laya.serve): GET /health, POST /v1/systemone,
POST /v1/systemone/batch. Only the Router is built here, pointed at local folders.

    uvicorn main:app --host 0.0.0.0 --port 8090
"""

import os
from pathlib import Path

# Fail instead of silently downloading if a local file is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from laya.mcp.device import env_device  # noqa: E402
from laya.router import Router  # noqa: E402
from laya.serve import create_app  # noqa: E402

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models" / "laya")).resolve()
# Same layout as the hub repo: english at the root, the other two in subfolders.
CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


def build_router() -> Router:
    for sub in CHECKPOINTS.values():
        weights = MODEL_DIR / (sub or "") / "model.safetensors"
        if not weights.exists():
            raise RuntimeError(f"{weights} not found. Run `python3 model_chunks.py join` first.")

    router = Router(
        models={name: (str(MODEL_DIR), sub) for name, sub in CHECKPOINTS.items()},
        device=env_device(),
        auto_task_detection=os.getenv("LAYA_AUTO_TASK", "0") == "1",
    )
    # Load every checkpoint at startup so no request pays a cold load. LAYA_MODELS limits the set.
    names = [m.strip() for m in os.getenv("LAYA_MODELS", "").split(",") if m.strip()] or None
    router.preload(names)
    return router


app = create_app(build_router())
