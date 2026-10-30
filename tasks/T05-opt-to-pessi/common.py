"""Models, labels, inputs and output checks shared by the T05 analysis scripts."""

import csv
import io
import json
from pathlib import Path

from zero_shot_rl.latent import archive

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data" / "inputs"  # latents, rollouts, traces and missing targets
OUTPUTS = ROOT / "data"
FIGURES = ROOT / "figures"
MODELS = json.loads((ROOT / "configs" / "sources.json").read_text())["models"]
TASKS = ("stand", "walk", "run", "flip")
SOURCES = tuple((model["training_seed"], Path(model["latents"]).name) for model in MODELS)
METHODS = {
    "CQL": "CQL implicit-reward projection",
    "ReBRAC": "ReBRAC action-deviation penalty",
    "ReBRAC+E2E": "ReBRAC penalty plus norm-matched constant reward",
    "E2E": "Positive constant reward",
    "E2E-bi": "Deterministic-policy next-state bonus (bilevel E2E-style)",
}
REPORT_LABELS = {
    METHODS["CQL"]: "CQL 價值偏好動作分布修正",
    METHODS["ReBRAC"]: "ReBRAC 動作偏離懲罰",
    METHODS["ReBRAC+E2E"]: "ReBRAC 動作偏離懲罰＋長度配平的正常數 reward",
    METHODS["E2E"]: "正常數 reward",
    METHODS["E2E-bi"]: "確定性 policy 下一狀態 bonus（bilevel E2E-style）",
}
# These two targets exist for every method, task and training seed.
MATCHED_ANGLES = (5.0, 10.0)


def validate(data, source):
    archive.validate(data, source, TASKS, 50)


def csv_text(rows):
    if not rows:
        raise ValueError("Refusing to write an empty table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: format(v, ".10g") if isinstance(v, float) else v
                         for k, v in row.items()})
    return stream.getvalue()


def write_or_check(path, content, check):
    """Write `content` (bytes) to `path`, or with `check` only verify that `path` holds it."""
    if check:
        if not path.exists() or path.read_bytes() != content:
            raise SystemExit(f"Stale or missing output: {path}")
        print(f"Verified {path.name}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"Wrote {path}")
