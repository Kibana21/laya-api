"""Download the three Laya checkpoints into ./models/laya.

Run this once on a machine that can reach huggingface.co, then run
`python3 model_chunks.py split` to produce the parts that are committed.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "convaiinnovations/laya"
# Revision pinned by laya's own verify/checkpoints.json, whose SHA-256 values these weights match.
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
TARGET = Path(__file__).parent / "models" / "laya"
# Only what laya.Agent loads; skips images, eval output and the hub README.
FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
SUBFOLDERS = ["", "multilingual/", "typed-decisions/"]  # english, multilingual, typed-decisions

if __name__ == "__main__":
    patterns = [sub + f for sub in SUBFOLDERS for f in FILES]
    path = snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=TARGET, allow_patterns=patterns)
    print(f"Model saved to {path}")
