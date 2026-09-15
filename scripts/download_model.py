"""download_model.py — fetch model weights into the model directory.

Prerequisites: a machine with internet access (on many clusters, a login node;
    compute nodes often have none), and `source ./env.sh` beforehand, which
    sets MODELS_DIR. huggingface_hub must be importable by whichever
    interpreter runs this.
Outputs:       $MODELS_DIR/<repo with / replaced by _>, and nothing else.

Contract: the target directory name is the repository id with `/` replaced by a
    single `_`, because that is what the serving script looks for —
    `unsloth/Qwen3.8-27B-FP8` becomes `unsloth_Qwen3.8-27B-FP8`. A different
    separator produces a directory the server cannot find.

Why: MODELS_DIR comes from the environment, so the destination follows the
    installation rather than a constant written into the file. The resolved
    target is printed before anything is fetched, and a target outside
    MODELS_DIR's parent has to be confirmed.

    No token is stored here. Public repositories need none, and a token in a
    file is a token in Git, in a backup and in a paste.

Usage:
    source ./env.sh
    python scripts/download_model.py
    python scripts/download_model.py unsloth/Qwen3.8-27B-FP8 Qwen/Qwen3.8-27B
    python scripts/download_model.py --output-dir /somewhere/else
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# The checkpoint the reference configuration serves.
DEFAULT_MODELS = ["unsloth/Qwen3.8-27B-FP8"]

# vLLM loads safetensors. A GGUF copy in the same repository is a second full
# set of weights for llama.cpp and doubles the download for nothing.
IGNORE = ["*.gguf", "*.pth", "original/*"]


def target_for(repo_id: str, out_root: Path) -> Path:
    return out_root / repo_id.replace("/", "_")


def missing_files(repo_id: str, target: Path, token: str | None) -> list[str]:
    """Remote files this download did not produce.

    An interrupted transfer leaves a directory that looks plausible: config
    and a few shards land first, so `test -f config.json` passes while the
    tokenizer is simply absent. vLLM then accepts the path and the server
    dies at config validation because it cannot tokenize the reasoning
    delimiters.
    Compare against the repository listing instead of trusting one file.
    """
    if not target.is_dir():
        return ["<nothing downloaded>"]
    remote = {name for name in HfApi(token=token).list_repo_files(repo_id)
              if not any(fnmatch.fnmatch(name, pat) for pat in IGNORE)}
    local = set()
    for path in target.rglob("*"):
        if path.is_file():
            name = str(path.relative_to(target))
            if not name.startswith(".cache/"):
                local.add(name)
    return sorted(remote - local)


def resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    models_dir = os.environ.get("MODELS_DIR")
    if not models_dir:
        raise SystemExit(
            "MODELS_DIR is not set. Run `source ./env.sh` from the "
            "repository root first, or pass --output-dir.")
    return Path(models_dir)


def confirm_owner(out_root: Path, assume_yes: bool) -> None:
    """Stop before writing outside the configured storage root.

    MODELS_DIR follows env.sh, but --output-dir does not, and a stale shell can
    still carry an old value. A large download in the wrong place is expensive
    and hard to notice.
    """
    storage = os.environ.get("RLM_STORAGE_ROOT")
    if not storage:
        return
    expected = Path(storage)
    try:
        out_root.resolve().relative_to(expected.resolve())
        return
    except ValueError:
        pass
    print(f"\nWARNING: {out_root}")
    print(f"         is outside the storage root {expected}.")
    if assume_yes:
        print("         --yes given; continuing.")
        return
    if input("         Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        raise SystemExit("aborted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", default=None,
                    help=f"repository ids (default: {' '.join(DEFAULT_MODELS)})")
    ap.add_argument("--output-dir", default=None,
                    help="override MODELS_DIR")
    ap.add_argument("--yes", action="store_true",
                    help="do not ask before writing outside the storage root")
    ap.add_argument("--verify", action="store_true",
                    help="only report missing files; download nothing")
    args = ap.parse_args()

    repos = args.repos or DEFAULT_MODELS
    out_root = resolve_output_dir(args.output_dir)

    print(f"destination : {out_root}")
    for repo_id in repos:
        print(f"  {repo_id}  ->  {target_for(repo_id, out_root).name}")
    confirm_owner(out_root, args.yes)
    out_root.mkdir(parents=True, exist_ok=True)

    # Gated repositories only. Set HF_TOKEN in the environment; do not put it
    # in this file.
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN") or None
    if token:
        print("using a token from the environment")

    failed = []
    for repo_id in repos:
        target = target_for(repo_id, out_root)
        if args.verify:
            absent = missing_files(repo_id, target, token)
            print(f"{repo_id}: " + (f"INCOMPLETE — {len(absent)} missing, "
                                    f"e.g. {absent[:6]}" if absent
                                    else "complete"))
            if absent:
                failed.append(repo_id)
            continue
        print(f"\n=== {repo_id} -> {target} ===", flush=True)
        try:
            # snapshot_download resumes an interrupted transfer, so a dropped
            # session costs the current file rather than the download.
            snapshot_download(repo_id=repo_id, local_dir=str(target),
                              token=token, repo_type="model",
                              ignore_patterns=IGNORE)
        except Exception as error:            # noqa: BLE001 — report, continue
            print(f"FAILED {repo_id}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            failed.append(repo_id)
            continue
        absent = missing_files(repo_id, target, token)
        if absent:
            print(f"FAILED {repo_id}: {len(absent)} file(s) missing, e.g. "
                  f"{absent[:6]}", file=sys.stderr)
            print("Re-run this command; snapshot_download resumes.",
                  file=sys.stderr)
            failed.append(repo_id)
            continue
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"ok: {target}  ({size / 1e9:.1f} GB)")

    if failed:
        print(f"\nincomplete: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
    print("\nall downloads complete")


if __name__ == "__main__":
    main()
