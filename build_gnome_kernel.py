#!/usr/bin/env python3
"""Build GNOmE large-scale Kaggle kernel."""
import base64, json, os, pathlib, zlib

WORK = pathlib.Path(__file__).resolve().parent

# Collect all gnome source files
src_files = {}
for root, dirs, files in os.walk(str(WORK / "gnome")):
    for f in files:
        if f.endswith(".py") and "__pycache__" not in root:
            rel = os.path.relpath(os.path.join(root, f), str(WORK)).replace("\\", "/")
            src_files[rel] = open(os.path.join(root, f), encoding="utf-8").read()

# Add the experiment script
src_files["exp/gnome_large_scale.py"] = open(WORK / "kernel" / "gnome_large_scale.py", encoding="utf-8").read()

print("Packing", len(src_files), "files")
compressed = zlib.compress(json.dumps(src_files).encode("utf-8"))
b64 = base64.b64encode(compressed).decode("ascii")
print("Blob:", len(b64), "chars")

SCRIPT = r'''"""GNOmE Large-Scale kernel -- built by build_gnome_kernel.py"""
import base64, json, os, pathlib, subprocess, sys, zlib

BLOB = "''' + b64 + r'''"
ARGS = []
WORK = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle").exists() else pathlib.Path(".").resolve()
SRCDIR = WORK / "src"
files = json.loads(zlib.decompress(base64.b64decode(BLOB)).decode("utf-8"))
for rel, txt in files.items():
    p = SRCDIR / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")
print("unpacked", len(files), "files", flush=True)

import subprocess as _sp
print("using pre-installed torch (no pip install)", flush=True)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "device count:", torch.cuda.device_count(), flush=True)

env = dict(os.environ, PYTHONPATH=str(SRCDIR), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
cmd = [sys.executable, str(SRCDIR / "exp/gnome_large_scale.py")] + ARGS
print("run", " ".join(cmd), flush=True)
os.execve(cmd[0], cmd, env)
'''

OUT = WORK / "kernel" / "gnome_large_scale"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "gnome_large_scale.py").write_text(SCRIPT, encoding="utf-8")

meta = {
    "id": "sehajrsingh/gnome-large-scale",
    "title": "GNOmE Large-Scale (6-8 layer transformer, modular addition)",
    "code_file": "gnome_large_scale.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Built:", OUT)
