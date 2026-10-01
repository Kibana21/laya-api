"""Split the model folder into GitHub-sized chunks and join it back, with SHA-256 checks.

GitHub rejects files over 100 MB, so each model.safetensors (643-843 MB) is cut into
45 MiB parts. Small files (config, tokenizer) are copied as-is. A manifest
records the SHA-256 of every part and every original file, so a rebuilt
model is byte-for-byte identical to the one that was split.

    python model_chunks.py split    # models/laya -> model_chunks/
    python model_chunks.py verify   # check the parts in model_chunks/
    python model_chunks.py join     # model_chunks/ -> models/laya

Only the standard library is used, so it runs before requirements are installed.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "laya"
CHUNK_DIR = ROOT / "model_chunks"
MANIFEST = "manifest.json"
CHUNK_SIZE = 45 * 1024 * 1024  # under GitHub's 50 MB warning and 100 MB hard limit
SKIP_DIRS = {".cache"}  # huggingface download metadata, not needed to load the model
SKIP_FILES = {".gitattributes"}  # HF LFS rules; would turn the parts into LFS pointers in this repo
BUF = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(BUF):
            h.update(block)
    return h.hexdigest()


def model_files(model_dir: Path) -> list[Path]:
    files = []
    for p in sorted(model_dir.rglob("*")):
        rel = p.relative_to(model_dir)
        if p.is_file() and rel.parts[0] not in SKIP_DIRS and p.name not in SKIP_FILES:
            files.append(p)
    return files


def split(model_dir: Path, chunk_dir: Path, chunk_size: int) -> None:
    if not model_dir.is_dir():
        sys.exit(f"Model folder not found: {model_dir}")
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)

    manifest = {"chunk_size": chunk_size, "files": []}
    for src in model_files(model_dir):
        rel = src.relative_to(model_dir).as_posix()
        size = src.stat().st_size
        print(f"Hashing {rel} ({size:,} bytes)")
        entry = {"path": rel, "size": size, "sha256": sha256_file(src), "parts": []}

        if size <= chunk_size:
            dest = chunk_dir / "files" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            entry["parts"].append({"name": f"files/{rel}", "size": size, "sha256": entry["sha256"]})
        else:
            with open(src, "rb") as f:
                index = 0
                while data := f.read(chunk_size):
                    name = f"{rel.replace('/', '__')}.part{index:03d}"
                    (chunk_dir / name).write_bytes(data)
                    entry["parts"].append(
                        {"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    )
                    index += 1
            print(f"  -> {index} parts")
        manifest["files"].append(entry)

    (chunk_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {chunk_dir / MANIFEST}")
    verify(chunk_dir)


def load_manifest(chunk_dir: Path) -> dict:
    path = chunk_dir / MANIFEST
    if not path.exists():
        sys.exit(f"Manifest not found: {path}")
    return json.loads(path.read_text())


def verify(chunk_dir: Path) -> None:
    """Check every part's size and SHA-256 against the manifest."""
    manifest = load_manifest(chunk_dir)
    bad = []
    for entry in manifest["files"]:
        if sum(p["size"] for p in entry["parts"]) != entry["size"]:
            bad.append(f"{entry['path']}: part sizes do not add up to file size")
        for part in entry["parts"]:
            path = chunk_dir / part["name"]
            if not path.exists():
                bad.append(f"{part['name']}: missing")
            elif path.stat().st_size != part["size"]:
                bad.append(f"{part['name']}: size {path.stat().st_size} != {part['size']}")
            elif sha256_file(path) != part["sha256"]:
                bad.append(f"{part['name']}: checksum mismatch")
    if bad:
        print("Verification FAILED:")
        for line in bad:
            print(f"  {line}")
        sys.exit(1)
    n_parts = sum(len(e["parts"]) for e in manifest["files"])
    print(f"OK: {n_parts} parts for {len(manifest['files'])} files match the manifest")


def join(chunk_dir: Path, model_dir: Path) -> None:
    verify(chunk_dir)
    manifest = load_manifest(chunk_dir)
    for entry in manifest["files"]:
        dest = model_dir / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        h = hashlib.sha256()
        with open(tmp, "wb") as out:
            for part in entry["parts"]:
                with open(chunk_dir / part["name"], "rb") as f:
                    while block := f.read(BUF):
                        h.update(block)
                        out.write(block)
        if h.hexdigest() != entry["sha256"] or tmp.stat().st_size != entry["size"]:
            tmp.unlink()
            sys.exit(f"Rebuilt {entry['path']} does not match the original checksum")
        os.replace(tmp, dest)
        print(f"Restored {entry['path']} ({entry['size']:,} bytes, sha256 {entry['sha256'][:16]}...)")
    print(f"Model ready in {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["split", "join", "verify"])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--chunk-mb", type=int, default=CHUNK_SIZE // (1024 * 1024))
    args = parser.parse_args()

    if args.command == "split":
        split(args.model_dir, args.chunk_dir, args.chunk_mb * 1024 * 1024)
    elif args.command == "verify":
        verify(args.chunk_dir)
    else:
        join(args.chunk_dir, args.model_dir)


if __name__ == "__main__":
    main()
