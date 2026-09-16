"""Resolve local data and isolate each experiment's generated files."""

from datetime import datetime, timezone
from pathlib import Path
import os
import re
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def create_run(name):
    """Create a fresh output directory and return (data_dir, output_dir).

    Existing data is only read. Legacy relative output paths are contained by
    changing the working directory to a new experiment directory.
    """
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError("Use a lowercase experiment name with letters, numbers, underscores, or hyphens.")
    data_dir = Path(os.environ.get("SPECTRAL_DATA_DIR", ROOT / "data" / "local")).resolve()
    output_root = Path(os.environ.get("SPECTRAL_OUTPUT_DIR", ROOT / "results")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / f"{name}_{stamp}_{uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=False)
    for child in ("processed", "checkpoints/ckpt", "checkpoints/ckpt_initial", "export"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    return data_dir, output_dir
