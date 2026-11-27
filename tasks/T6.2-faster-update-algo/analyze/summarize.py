"""Summarize T6.2-faster-update-algo Table-6 IQM across OOD dynamics, algorithms, and search spaces.

Reads per-seed ``table6.csv`` under ``--inputs_dir`` (default ``tasks/T6.2-faster-update-algo/data/inputs``)
and optionally includes T6.1 reference rows (Frozen FB z_r and REINFORCE at 800k / 3.2M env steps) from
``--t61_summary``.

Usage (repo root):
    PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.summarize
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.table6_aggregate import TASKS as TABLE6_TASKS
from analysis.table6_aggregate import aggregate as table6_aggregate

DYNAMICS_ORDER: Tuple[str, ...] = (
    "friction_mild",
    "friction_high_mild",
    "actuator_mild",
    "actuator_high_mild",
)

# (condition_dir, algo_label, search_space_label)
T62_CONDITIONS: Tuple[Tuple[str, str, str], ...] = (
    ("cmaes_z", "CMA-ES (full-cov)", "z"),
    ("lola_z", "LoLA", "z"),
    ("rela_z", "ReLA", "z"),
    ("cem_z", "CEM", "z"),
    ("cmaes_anfis", "sep-CMA-ES", "anfis (4 sensors)"),
    ("cmaes_anfis_custom", "sep-CMA-ES", "anfis (custom sensors)"),
    ("cem_anfis", "CEM", "anfis (4 sensors)"),
    ("cem_anfis_custom", "CEM", "anfis (custom sensors)"),
    ("lola_anfis", "LoLA", "anfis (4 sensors)"),
    ("lola_anfis_custom", "LoLA", "anfis (custom sensors)"),
)


def _load_steps(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path).set_index("step")[TABLE6_TASKS]


def _load_t61_references(t61_summary: Optional[Path]) -> List[Dict[str, object]]:
    if t61_summary is None or not t61_summary.exists():
        return []
    t61 = pd.read_csv(t61_summary)
    rows: List[Dict[str, object]] = []
    for dyn in DYNAMICS_ORDER:
        sub = t61[t61["dynamics"] == dyn]
        if sub.empty:
            continue
        by_cond = sub.set_index("condition")
        if "clean_fb_zr" in by_cond.index:
            r = by_cond.loc["clean_fb_zr"]
            rows.append({
                "dynamics": dyn,
                "condition": "t61_frozen_fb_zr",
                "algo": "Frozen FB z_r (T6.1)",
                "search_space": "z (zero-shot)",
                "n_seeds": int(r["n_seeds"]),
                "selected_step": 0,
                "select_iqm": float(r["table6_iqm"]),
                "final_step_iqm": float(r["final_iqm"]),
                **{t: float(r[t]) for t in TABLE6_TASKS},
            })
        for cond_key, space_label in (
            ("ood_4sensor", "anfis (4 sensors)"),
            ("ood_custom", "anfis (custom sensors)"),
        ):
            if cond_key not in by_cond.index:
                continue
            r = by_cond.loc[cond_key]
            # T6.1 stage 1 (step 50 = 800k env steps; no-selection snapshot in summary.csv)
            rows.append({
                "dynamics": dyn,
                "condition": f"t61_reinforce_800k_{cond_key}",
                "algo": "REINFORCE 800k (T6.1)",
                "search_space": space_label,
                "n_seeds": int(r["n_seeds"]),
                "selected_step": 800000,
                "select_iqm": float(r["iqm_step50"]),
                "final_step_iqm": float(r["iqm_step50"]),
                **{t: np.nan for t in TABLE6_TASKS},
            })
            # T6.1 full 200 iters (3.2M env steps)
            rows.append({
                "dynamics": dyn,
                "condition": f"t61_reinforce_3p2m_{cond_key}",
                "algo": "REINFORCE 3.2M (T6.1)",
                "search_space": space_label,
                "n_seeds": int(r["n_seeds"]),
                "selected_step": int(r["step"]) * 16000 if not pd.isna(r["step"]) else np.nan,
                "select_iqm": float(r["table6_iqm"]),
                "final_step_iqm": float(r["final_iqm"]),
                **{t: float(r[t]) for t in TABLE6_TASKS},
            })
    return rows


def summarize(
    inputs_dir: Path,
    t61_summary: Optional[Path] = None,
    min_seeds: int = 10,
    require_all_envs: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = _load_t61_references(t61_summary)

    for cond, algo_label, space_label in T62_CONDITIONS:
        cond_rows: List[Dict[str, object]] = []
        for dyn in DYNAMICS_ORDER:
            seed_dirs = sorted((inputs_dir / cond / dyn).glob("ft*"))
            runs = [_load_steps(d / "table6.csv") for d in seed_dirs if (d / "table6.csv").exists()]
            if len(runs) < min_seeds:
                continue
            steps = runs[0].index
            assert all(r.index.equals(steps) for r in runs), f"{cond}/{dyn}: runs must share eval steps"
            arr = np.stack([r.to_numpy(dtype=np.float64) for r in runs])  # [seeds, steps, tasks]
            per_step = table6_aggregate(arr)
            best = int(np.argmax(per_step))
            row: Dict[str, object] = {
                "dynamics": dyn,
                "condition": cond,
                "algo": algo_label,
                "search_space": space_label,
                "n_seeds": len(runs),
                "selected_step": int(steps[best]),
                "select_iqm": float(per_step[best]),
                "final_step_iqm": float(per_step[-1]),
            }
            for i, t in enumerate(TABLE6_TASKS):
                row[t] = float(arr[:, best, i].mean())
            cond_rows.append(row)
        if require_all_envs and len(cond_rows) != len(DYNAMICS_ORDER):
            continue
        rows.extend(cond_rows)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep canonical ordering: per dynamics, then condition order (T6.1 refs then T6.2 conditions).
    cond_order = [
        "t61_frozen_fb_zr",
        "t61_reinforce_800k_ood_4sensor",
        "t61_reinforce_800k_ood_custom",
        "t61_reinforce_3p2m_ood_4sensor",
        "t61_reinforce_3p2m_ood_custom",
        *[c[0] for c in T62_CONDITIONS],
    ]
    dyn_rank = {d: i for i, d in enumerate(DYNAMICS_ORDER)}
    cond_rank = {c: i for i, c in enumerate(cond_order)}
    df["_dr"] = df["dynamics"].map(dyn_rank)
    df["_cr"] = df["condition"].map(lambda c: cond_rank.get(c, 999))
    df = df.sort_values(["_dr", "_cr"]).drop(columns=["_dr", "_cr"]).reset_index(drop=True)

    # 4-environment mean rows for every condition that has all 4 environments populated.
    mean_rows: List[Dict[str, object]] = []
    for cond in cond_order:
        sub = df[df["condition"] == cond]
        if len(sub) != len(DYNAMICS_ORDER):
            continue
        mean_rows.append({
            "dynamics": "mean_of_4_dynamics",
            "condition": cond,
            "algo": sub["algo"].iloc[0],
            "search_space": sub["search_space"].iloc[0],
            "n_seeds": int(sub["n_seeds"].min()),
            "selected_step": np.nan,
            "select_iqm": float(sub["select_iqm"].mean()),
            "final_step_iqm": float(sub["final_step_iqm"].mean()),
            **{t: float(sub[t].mean()) for t in TABLE6_TASKS},
        })
    if mean_rows:
        df = pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)
    return df


def to_markdown(df: pd.DataFrame) -> str:
    lines = [
        "| OOD env | algo | searching space | select IQM | final step IQM |",
        "|---|---|---|---:|---:|",
    ]
    for _, r in df.iterrows():
        env_label = "4-env mean" if r["dynamics"] == "mean_of_4_dynamics" else str(r["dynamics"])
        lines.append(
            f"| {env_label} | {r['algo']} | {r['search_space']} | "
            f"{r['select_iqm']:.1f} | {r['final_step_iqm']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs_dir",
        type=Path,
        default=Path("tasks/T6.2-faster-update-algo/data/inputs"),
    )
    parser.add_argument(
        "--t61_summary",
        type=Path,
        default=Path("tasks/T6.1-ood-fuzzy/data/summary.csv"),
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("tasks/T6.2-faster-update-algo/data/summary.csv"),
    )
    parser.add_argument(
        "--min_seeds",
        type=int,
        default=10,
        help="Minimum completed seeds required for a (condition, dynamics) entry to be included",
    )
    args = parser.parse_args()

    df = summarize(
        args.inputs_dir,
        args.t61_summary if args.t61_summary.exists() else None,
        min_seeds=args.min_seeds,
    )
    if df.empty:
        raise SystemExit(f"No inputs found under {args.inputs_dir}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, float_format="%.2f")
    md_path = args.out_csv.with_suffix(".md")
    md_path.write_text(to_markdown(df))
    print(f"Wrote {args.out_csv} and {md_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
