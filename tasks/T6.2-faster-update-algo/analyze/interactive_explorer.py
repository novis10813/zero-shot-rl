"""Build a self-contained interactive HTML explorer for T6.2 ANFIS reward functions.

Supports real-time slider manipulation of fixed sensors with:
  - 1D Slice view (single sensor, 10-seed mean +- 1 std, optional individual seeds & data density)
  - 2D Heatmap view (any pair of sensors: Original r(s), ANFIS 10-seed mean, diff, std + real-data overlay)
  - All-Sensors Grid view (live synchronized 1D slices across all sensors)

Usage:
    MUJOCO_GL=egl PYTHONPATH=.:tasks/T6.2-faster-update-algo uv run python -m analyze.interactive_explorer \
        --artifacts_dir artifacts/T6.2-faster-update-algo \
        --dataset_path /mnt/raid1/novis/zero-shot-rl/datasets/walker/rnd/dataset.npz \
        --out tasks/T6.2-faster-update-algo/figures/cmaes_800k/anfis_explorer.html
"""

import base64
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from analysis.table6_aggregate import TASKS
from analyze.reward_figs import ALL_WALKER_SENSORS, discover_artifact_dirs
from zero_shot_rl.finetune.anfis_z import collect_reference_set
from zero_shot_rl.latent.anfis import ANFIS, load_anfis

TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = TASK_DIR / "figures" / "cmaes_800k" / "anfis_explorer.html"

CONDITION_SPECS = [
    ("actuator_mild_custom", "cmaes_anfis_custom", "actuator_mild", "Actuator Mild (76%) — Custom 8-sensor"),
    ("actuator_mild", "cmaes_anfis", "actuator_mild", "Actuator Mild (76%) — 4-sensor"),
    ("friction_mild_custom", "cmaes_anfis_custom", "friction_mild", "Friction Mild (0.1) — Custom 7-sensor"),
    ("friction_mild", "cmaes_anfis", "friction_mild", "Friction Mild (0.1) — 4-sensor"),
    ("friction_high_mild_custom", "cmaes_anfis_custom", "friction_high_mild", "Friction High (1.5) — Custom 7-sensor"),
    ("friction_high_mild", "cmaes_anfis", "friction_high_mild", "Friction High (1.5) — 4-sensor"),
    ("actuator_high_mild_custom", "cmaes_anfis_custom", "actuator_high_mild", "Actuator High (130%) — Custom 8-sensor"),
    ("actuator_high_mild", "cmaes_anfis", "actuator_high_mild", "Actuator High (130%) — 4-sensor"),
]

SENSOR_LABELS = {
    "torso_height": "torso_height (m)",
    "torso_upright": "torso_upright",
    "horizontal_velocity": "CoM_vx (m/s)",
    "angmomentum": "angmom_y",
}

BASE_SLICE_RANGES = {
    "torso_height": (0.10, 1.55),
    "torso_upright": (-1.0, 1.0),
    "horizontal_velocity": (-3.0, 8.5),
    "angmomentum": (-35.0, 55.0),
}


def _f32_b64(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def export_condition_spec(model: ANFIS) -> Dict[str, Any]:
    """Export static architecture and input-normalization stats shared across seeds."""
    if model.t_norm != "product":
        raise ValueError(f"Interactive explorer supports t_norm='product', got {model.t_norm!r}")
    if not model.normalize_firing:
        raise ValueError("Interactive explorer requires normalize_firing=True")
    if model.output_activation != "identity":
        raise ValueError(f"Interactive explorer supports identity output, got {model.output_activation!r}")
    if any(n not in ("none", "zscore") for n in model.input_normalizers):
        raise ValueError(f"Unsupported input normalizers: {model.input_normalizers}")

    return {
        "sensor_names": list(model.sensor_names),
        "term_counts": [int(m) for m in model.term_counts],
        "num_rules": int(model.num_rules),
        "in_dim": int(model.in_dim),
        "temperature": float(model.temperature),
        "normalize_inputs": bool(model.normalize_inputs),
        "input_normalizers": list(model.input_normalizers),
        "x_mean": [float(v) for v in model.x_mean.detach().cpu().numpy()],
        "x_std": [float(v) for v in model.x_std.detach().cpu().numpy()],
    }


def export_model_weights(model: ANFIS) -> Dict[str, str]:
    """Export one trained ANFIS seed's parameters as compact base64 float32 arrays."""
    centers_parts = []
    sigmas_parts = []
    for mem in model.memberships:
        centers_parts.append(mem.centers.detach().cpu().numpy().reshape(-1))
        sigmas_parts.append(mem.sigma.detach().cpu().numpy().reshape(-1))
    centers = np.concatenate(centers_parts, axis=0)
    sigmas = np.concatenate(sigmas_parts, axis=0)
    # FirstOrderConsequent: weight [R, 1, D] -> [R, D], bias [R, 1] -> [R]
    weight = model.consequent.weight.detach().cpu().numpy()[:, 0, :]
    bias = model.consequent.bias.detach().cpu().numpy()[:, 0]
    return {
        "centers_b64": _f32_b64(centers),
        "sigmas_b64": _f32_b64(sigmas),
        "weight_b64": _f32_b64(weight),
        "bias_b64": _f32_b64(bias),
    }


def build_explorer_payload(
    artifacts_dir: Path,
    dataset_path: Path,
    tag: str = "final",
    max_ref_samples: int = 2000,
) -> Dict[str, Any]:
    """Extract all 8 conditions x 4 tasks x 10 seeds and reference pool statistics."""
    _, feats_t, _, _ = collect_reference_set(
        dataset_path=dataset_path,
        domain="walker",
        sensor_names=ALL_WALKER_SENSORS,
        tasks=TASKS,
        dataset_transitions=100000,
        z_inference_steps=10000,
        seed=42,
        discount=0.98,
        device=torch.device("cpu"),
    )
    assert feats_t is not None
    feats_np = feats_t.numpy().astype(np.float32)  # [10000, 11]

    # Per-sensor histograms and quantiles over all 10,000 real reference states
    sensor_stats: Dict[str, Dict[str, Any]] = {}
    for idx, sname in enumerate(ALL_WALKER_SENSORS):
        col = feats_np[:, idx]
        q = np.percentile(col, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
        if sname in BASE_SLICE_RANGES:
            lo, hi = BASE_SLICE_RANGES[sname]
        else:
            span = max(float(q[7] - q[1]), 1e-3)
            lo = float(q[1] - 0.1 * span)
            hi = float(q[7] + 0.1 * span)
        hist_counts, hist_edges = np.histogram(col, bins=40, range=(lo, hi))
        sensor_stats[sname] = {
            "label": SENSOR_LABELS.get(sname, sname),
            "mean": float(col.mean()),
            "std": float(max(col.std(), 1e-6)),
            "min": float(lo),
            "max": float(hi),
            "p01": float(q[1]),
            "p05": float(q[2]),
            "p50": float(q[4]),
            "p95": float(q[6]),
            "p99": float(q[7]),
            "hist_counts": [int(c) for c in hist_counts],
            "hist_edges": [round(float(e), 4) for e in hist_edges],
        }

    # Deterministic subsample of real reference states for fast live support/overlay in browser
    rng = np.random.default_rng(42)
    n_sub = min(max_ref_samples, len(feats_np))
    sub_idx = rng.choice(len(feats_np), size=n_sub, replace=False)
    ref_sub = feats_np[sub_idx]  # [n_sub, 11]

    conditions: Dict[str, Any] = {}
    condition_order: List[str] = []
    for key, cond_prefix, env_name, label in CONDITION_SPECS:
        dirs = discover_artifact_dirs(artifacts_dir, f"{cond_prefix}_{env_name}")
        if not dirs:
            continue
        first_model, _ = load_anfis(dirs[0] / f"{TASKS[0]}_{tag}.pt")
        spec = export_condition_spec(first_model)
        tasks_data: Dict[str, List[Dict[str, str]]] = {}
        for task in TASKS:
            seeds_list = []
            for md in dirs:
                m, _ = load_anfis(md / f"{task}_{tag}.pt")
                seeds_list.append(export_model_weights(m))
            tasks_data[task] = seeds_list
        conditions[key] = {
            "key": key,
            "label": label,
            "env": env_name,
            "n_seeds": len(dirs),
            "spec": spec,
            "tasks": tasks_data,
        }
        condition_order.append(key)

    return {
        "tasks": list(TASKS),
        "condition_order": condition_order,
        "conditions": conditions,
        "sensor_stats": sensor_stats,
        "ref_sensor_names": list(ALL_WALKER_SENSORS),
        "ref_n": int(n_sub),
        "ref_total": int(len(feats_np)),
        "ref_samples_b64": _f32_b64(ref_sub),
    }


JS_CORE_MATH = r"""
function b64ToFloat32(b64) {
  if (typeof atob === 'function') {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function buildRuleTable(termCounts) {
  const D = termCounts.length;
  let R = 1;
  for (let d = 0; d < D; d++) R *= termCounts[d];
  const strides = new Int32Array(D);
  let s = 1;
  for (let d = D - 1; d >= 0; d--) {
    strides[d] = s;
    s *= termCounts[d];
  }
  const table = new Int32Array(R * D);
  for (let r = 0; r < R; r++) {
    for (let d = 0; d < D; d++) {
      table[r * D + d] = Math.floor(r / strides[d]) % termCounts[d];
    }
  }
  return table;
}

function decodeSeedWeights(wObj, spec) {
  const centersFlat = b64ToFloat32(wObj.centers_b64);
  const sigmasFlat = b64ToFloat32(wObj.sigmas_b64);
  const weight = b64ToFloat32(wObj.weight_b64);
  const bias = b64ToFloat32(wObj.bias_b64);
  const D = spec.in_dim;
  const centers = [];
  const invSigmas = [];
  let off = 0;
  for (let d = 0; d < D; d++) {
    const M = spec.term_counts[d];
    const c = new Float64Array(M);
    const isig = new Float64Array(M);
    for (let m = 0; m < M; m++) {
      c[m] = centersFlat[off + m];
      isig[m] = 1.0 / sigmasFlat[off + m];
    }
    centers.push(c);
    invSigmas.push(isig);
    off += M;
  }
  if (!spec._ruleTable) {
    spec._ruleTable = buildRuleTable(spec.term_counts);
  }
  return { centers, invSigmas, weight, bias };
}

function normalizeX(xRaw, spec) {
  const D = spec.in_dim;
  const xn = new Float64Array(D);
  const norms = spec.input_normalizers;
  for (let d = 0; d < D; d++) {
    const kind = norms ? norms[d] : (spec.normalize_inputs ? 'zscore' : 'none');
    xn[d] = kind === 'none' ? xRaw[d] : (xRaw[d] - spec.x_mean[d]) / spec.x_std[d];
  }
  return xn;
}

function evalAnfisSingle(xRaw, decoded, spec) {
  const D = spec.in_dim;
  const R = spec.num_rules;
  const rt = spec._ruleTable || (spec._ruleTable = buildRuleTable(spec.term_counts));
  const xn = normalizeX(xRaw, spec);
  const logMu = new Array(D);
  for (let d = 0; d < D; d++) {
    const M = spec.term_counts[d];
    const arr = new Float64Array(M);
    const c = decoded.centers[d];
    const isig = decoded.invSigmas[d];
    const v = xn[d];
    for (let m = 0; m < M; m++) {
      const z = (v - c[m]) * isig[m];
      arr[m] = -0.5 * z * z;
    }
    logMu[d] = arr;
  }
  const invT = 1.0 / spec.temperature;
  const logW = new Float64Array(R);
  let maxLog = -Infinity;
  for (let r = 0; r < R; r++) {
    let acc = 0.0;
    const base = r * D;
    for (let d = 0; d < D; d++) {
      acc += logMu[d][rt[base + d]];
    }
    const lw = acc * invT;
    logW[r] = lw;
    if (lw > maxLog) maxLog = lw;
  }
  let sumW = 0.0;
  let num = 0.0;
  const W = decoded.weight;
  const B = decoded.bias;
  for (let r = 0; r < R; r++) {
    const w = Math.exp(logW[r] - maxLog);
    sumW += w;
    const base = r * D;
    let f = B[r];
    for (let d = 0; d < D; d++) f += W[base + d] * xn[d];
    num += w * f;
  }
  return num / sumW;
}

function evalAnfis1DSweep(baseRaw, sweepIdx, xs, decoded, spec) {
  const D = spec.in_dim;
  const R = spec.num_rules;
  const N = xs.length;
  const rt = spec._ruleTable || (spec._ruleTable = buildRuleTable(spec.term_counts));
  const xnBase = normalizeX(baseRaw, spec);
  const logMuFixed = new Array(D);
  for (let d = 0; d < D; d++) {
    if (d === sweepIdx) continue;
    const M = spec.term_counts[d];
    const arr = new Float64Array(M);
    const c = decoded.centers[d];
    const isig = decoded.invSigmas[d];
    const v = xnBase[d];
    for (let m = 0; m < M; m++) {
      const z = (v - c[m]) * isig[m];
      arr[m] = -0.5 * z * z;
    }
    logMuFixed[d] = arr;
  }
  const baseLog = new Float64Array(R);
  const baseF = new Float64Array(R);
  const sweepW = new Float64Array(R);
  const sweepTerm = new Int32Array(R);
  const W = decoded.weight;
  const B = decoded.bias;
  for (let r = 0; r < R; r++) {
    const off = r * D;
    let acc = 0.0;
    let f = B[r];
    for (let d = 0; d < D; d++) {
      if (d === sweepIdx) {
        sweepW[r] = W[off + d];
        sweepTerm[r] = rt[off + d];
      } else {
        acc += logMuFixed[d][rt[off + d]];
        f += W[off + d] * xnBase[d];
      }
    }
    baseLog[r] = acc;
    baseF[r] = f;
  }
  const Ms = spec.term_counts[sweepIdx];
  const cs = decoded.centers[sweepIdx];
  const isigs = decoded.invSigmas[sweepIdx];
  const normKind = spec.input_normalizers ? spec.input_normalizers[sweepIdx] : (spec.normalize_inputs ? 'zscore' : 'none');
  const xm = spec.x_mean[sweepIdx];
  const xsStd = spec.x_std[sweepIdx];
  const invT = 1.0 / spec.temperature;
  const out = new Float64Array(N);
  const logMuS = new Float64Array(Ms);
  const logW = new Float64Array(R);
  for (let i = 0; i < N; i++) {
    const xns = normKind === 'none' ? xs[i] : (xs[i] - xm) / xsStd;
    for (let m = 0; m < Ms; m++) {
      const z = (xns - cs[m]) * isigs[m];
      logMuS[m] = -0.5 * z * z;
    }
    let maxLog = -Infinity;
    for (let r = 0; r < R; r++) {
      const lw = (baseLog[r] + logMuS[sweepTerm[r]]) * invT;
      logW[r] = lw;
      if (lw > maxLog) maxLog = lw;
    }
    let sumW = 0.0;
    let num = 0.0;
    for (let r = 0; r < R; r++) {
      const w = Math.exp(logW[r] - maxLog);
      sumW += w;
      num += w * (baseF[r] + sweepW[r] * xns);
    }
    out[i] = num / sumW;
  }
  return out;
}

function evalAnfis2DGrid(baseRaw, ix, iy, xs, ys, decoded, spec) {
  const D = spec.in_dim;
  const R = spec.num_rules;
  const nx = xs.length;
  const ny = ys.length;
  const rt = spec._ruleTable || (spec._ruleTable = buildRuleTable(spec.term_counts));
  const xnBase = normalizeX(baseRaw, spec);
  const logMuFixed = new Array(D);
  for (let d = 0; d < D; d++) {
    if (d === ix || d === iy) continue;
    const M = spec.term_counts[d];
    const arr = new Float64Array(M);
    const c = decoded.centers[d];
    const isig = decoded.invSigmas[d];
    const v = xnBase[d];
    for (let m = 0; m < M; m++) {
      const z = (v - c[m]) * isig[m];
      arr[m] = -0.5 * z * z;
    }
    logMuFixed[d] = arr;
  }
  const baseLog = new Float64Array(R);
  const baseF = new Float64Array(R);
  const wxArr = new Float64Array(R);
  const wyArr = new Float64Array(R);
  const txArr = new Int32Array(R);
  const tyArr = new Int32Array(R);
  const W = decoded.weight;
  const B = decoded.bias;
  for (let r = 0; r < R; r++) {
    const off = r * D;
    let acc = 0.0;
    let f = B[r];
    for (let d = 0; d < D; d++) {
      if (d === ix) {
        wxArr[r] = W[off + d];
        txArr[r] = rt[off + d];
      } else if (d === iy) {
        wyArr[r] = W[off + d];
        tyArr[r] = rt[off + d];
      } else {
        acc += logMuFixed[d][rt[off + d]];
        f += W[off + d] * xnBase[d];
      }
    }
    baseLog[r] = acc;
    baseF[r] = f;
  }
  const normX = spec.input_normalizers ? spec.input_normalizers[ix] : (spec.normalize_inputs ? 'zscore' : 'none');
  const normY = spec.input_normalizers ? spec.input_normalizers[iy] : (spec.normalize_inputs ? 'zscore' : 'none');
  const Mx = spec.term_counts[ix];
  const My = spec.term_counts[iy];
  const cx = decoded.centers[ix], isx = decoded.invSigmas[ix];
  const cy = decoded.centers[iy], isy = decoded.invSigmas[iy];
  const xnx = new Float64Array(nx);
  const logMuX = new Float64Array(nx * Mx);
  for (let i = 0; i < nx; i++) {
    const v = normX === 'none' ? xs[i] : (xs[i] - spec.x_mean[ix]) / spec.x_std[ix];
    xnx[i] = v;
    for (let m = 0; m < Mx; m++) {
      const z = (v - cx[m]) * isx[m];
      logMuX[i * Mx + m] = -0.5 * z * z;
    }
  }
  const xny = new Float64Array(ny);
  const logMuY = new Float64Array(ny * My);
  for (let j = 0; j < ny; j++) {
    const v = normY === 'none' ? ys[j] : (ys[j] - spec.x_mean[iy]) / spec.x_std[iy];
    xny[j] = v;
    for (let m = 0; m < My; m++) {
      const z = (v - cy[m]) * isy[m];
      logMuY[j * My + m] = -0.5 * z * z;
    }
  }
  const invT = 1.0 / spec.temperature;
  const out = new Float64Array(ny * nx);
  const logW = new Float64Array(R);
  const rowBaseLog = new Float64Array(R);
  const rowBaseF = new Float64Array(R);
  for (let j = 0; j < ny; j++) {
    const yOff = j * My;
    const vy = xny[j];
    for (let r = 0; r < R; r++) {
      rowBaseLog[r] = baseLog[r] + logMuY[yOff + tyArr[r]];
      rowBaseF[r] = baseF[r] + wyArr[r] * vy;
    }
    for (let i = 0; i < nx; i++) {
      const xOff = i * Mx;
      const vx = xnx[i];
      let maxLog = -Infinity;
      for (let r = 0; r < R; r++) {
        const lw = (rowBaseLog[r] + logMuX[xOff + txArr[r]]) * invT;
        logW[r] = lw;
        if (lw > maxLog) maxLog = lw;
      }
      let sumW = 0.0;
      let num = 0.0;
      for (let r = 0; r < R; r++) {
        const w = Math.exp(logW[r] - maxLog);
        sumW += w;
        num += w * (rowBaseF[r] + wxArr[r] * vx);
      }
      out[j * nx + i] = num / sumW;
    }
  }
  return out;
}

function dmTolerance(x, lower, upper, margin, sigmoid, valueAtMargin) {
  if (x >= lower && x <= upper) return 1.0;
  if (margin === 0) return 0.0;
  const d = (x < lower ? lower - x : x - upper) / margin;
  if (sigmoid === 'gaussian') {
    const scale = Math.sqrt(-2.0 * Math.log(valueAtMargin));
    const t = d * scale;
    return Math.exp(-0.5 * t * t);
  }
  if (sigmoid === 'linear') {
    const scale = 1.0 - valueAtMargin;
    const t = d * scale;
    return Math.abs(t) < 1.0 ? 1.0 - t : 0.0;
  }
  return 0.0;
}

function trueReward(task, h, up, vx, ang) {
  const standing = dmTolerance(h, 1.2, Infinity, 0.6, 'gaussian', 0.1);
  const upright = (1.0 + up) / 2.0;
  const sr = (3.0 * standing + upright) / 4.0;
  if (task === 'stand') return sr;
  let mv = 0.0;
  if (task === 'walk') {
    mv = dmTolerance(vx, 1.0, Infinity, 0.5, 'linear', 0.5);
  } else if (task === 'run') {
    mv = dmTolerance(vx, 8.0, Infinity, 4.0, 'linear', 0.5);
  } else if (task === 'flip') {
    mv = dmTolerance(ang, 5.0, Infinity, 5.0, 'linear', 0.0);
  }
  return sr * (5.0 * mv + 1.0) / 6.0;
}
"""


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>T6.2 ANFIS Interactive Reward Explorer (sep-CMA-ES 800k, 10 seeds)</title>
<style>
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #dd8452;
    --blue: #4c72b0;
    --red: #c44e52;
    --green: #16a34a;
    --amber: #d97706;
  }
  * { box-sizing: border-box; }
  html, body {
    overscroll-behavior: none;
  }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
  }
  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    justify-content: space-between;
  }
  header h1 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
  }
  .subtitle { color: var(--muted); font-size: 12px; }
  .top-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }
  select, button {
    font-family: inherit;
    font-size: 12.5px;
    padding: 5px 9px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    background: #fff;
    color: var(--text);
    cursor: pointer;
  }
  select:focus, button:focus { outline: 2px solid #93c5fd; outline-offset: 1px; }
  .layout {
    display: grid;
    grid-template-columns: 330px 1fr;
    gap: 14px;
    padding: 14px 18px;
    max-width: 1700px;
    margin: 0 auto;
  }
  @media (max-width: 980px) {
    .layout { grid-template-columns: 1fr; }
  }
  .panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }
  .section-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin: 0 0 8px 0;
  }
  .preset-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 10px;
  }
  .preset-btn {
    padding: 3px 8px;
    font-size: 11.5px;
    background: #f1f5f9;
    border-color: #cbd5e1;
  }
  .preset-btn:hover { background: #e2e8f0; }
  .readout-card {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 10px;
    font-variant-numeric: tabular-nums;
  }
  .readout-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin: 2px 0;
  }
  .readout-val { font-weight: 700; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .support-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
  }
  .sup-good { background: #dcfce7; color: #166534; }
  .sup-warn { background: #fef3c7; color: #92400e; }
  .sup-bad  { background: #fee2e2; color: #991b1b; }
  .slider-item {
    padding: 7px 0;
    border-top: 1px solid #f1f5f9;
  }
  .slider-item.swept {
    background: #fff7ed;
    margin: 0 -8px;
    padding: 7px 8px;
    border-radius: 6px;
    border-top-color: transparent;
  }
  .slider-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    margin-bottom: 3px;
  }
  .sensor-name { font-weight: 600; font-size: 12.5px; }
  .sensor-name.extra { color: var(--red); }
  .axis-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    background: var(--accent);
    color: #fff;
  }
  .axis-tag.ytag { background: #2563eb; }
  .slider-num {
    width: 66px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    padding: 2px 5px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    text-align: right;
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .slider-row input[type="range"] {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    touch-action: none;
  }
  .slider-meta {
    display: flex;
    justify-content: space-between;
    font-size: 10.5px;
    color: var(--muted);
    margin-top: 1px;
  }
  .tabs {
    display: flex;
    gap: 6px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
  }
  .tab-group { display: flex; gap: 6px; }
  .tab-btn {
    padding: 6px 12px;
    font-weight: 600;
    background: #f1f5f9;
    border: 1px solid transparent;
  }
  .tab-btn.active {
    background: var(--text);
    color: #fff;
  }
  .view-toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
    padding: 8px 10px;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  .view-toolbar label {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    cursor: pointer;
  }
  .canvas-wrap {
    position: relative;
    width: 100%;
  }
  canvas {
    display: block;
    width: 100%;
    border-radius: 6px;
    background: #fff;
  }
  .hm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  @media (max-width: 1100px) {
    .hm-grid { grid-template-columns: 1fr; }
  }
  .all-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 10px;
  }
  .mini-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 8px;
    cursor: pointer;
    transition: border-color 0.12s;
  }
  .mini-card:hover { border-color: var(--accent); }
  .hover-bar {
    margin-top: 8px;
    padding: 7px 10px;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    color: var(--text);
    min-height: 30px;
  }
  .legend-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 11.5px;
    color: var(--muted);
    margin-top: 6px;
  }
  .swatch {
    display: inline-block;
    width: 14px;
    height: 3px;
    vertical-align: middle;
    margin-right: 4px;
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>T6.2 ANFIS Interactive Reward Explorer</h1>
    <div class="subtitle">sep-CMA-ES 800k (10 seeds mean &plusmn; 1&sigma;) &middot; 即時拖動感測器拉桿觀察 1D 切面、2D 熱圖與真實參考資料支撐度</div>
  </div>
  <div class="top-controls">
    <label>環境與感測器:
      <select id="sel-cond"></select>
    </label>
    <label>任務:
      <select id="sel-task">
        <option value="stand">stand</option>
        <option value="walk">walk</option>
        <option value="run">run</option>
        <option value="flip">flip</option>
      </select>
    </label>
  </div>
</header>

<div class="layout">
  <aside class="panel">
    <div class="section-title">快速基準姿態</div>
    <div class="preset-grid" id="preset-btns"></div>

    <div class="readout-card">
      <div class="readout-row">
        <span>目前拉桿點 Original r(s)</span>
        <span class="readout-val" style="color:var(--blue)" id="ro-true">-</span>
      </div>
      <div class="readout-row">
        <span>ANFIS r~(s) (10-seed)</span>
        <span class="readout-val" style="color:var(--accent)" id="ro-anfis">-</span>
      </div>
      <div class="readout-row" style="margin-top:5px;">
        <span id="ro-sup-label">固定軸真實樣本支撐度</span>
        <span id="ro-sup-badge" class="support-badge sup-good">-</span>
      </div>
      <div style="font-size:10.5px; color:var(--muted); margin-top:2px;" id="ro-sup-detail"></div>
    </div>

    <div class="section-title">感測器拉桿（紅字為客製新增）</div>
    <div id="sliders-box"></div>
  </aside>

  <main class="panel">
    <div class="tabs">
      <div class="tab-group">
        <button class="tab-btn active" data-mode="1d">1D 切面</button>
        <button class="tab-btn" data-mode="2d">2D 熱圖</button>
        <button class="tab-btn" data-mode="grid">全感測器同步</button>
      </div>
      <div style="font-size:11.5px; color:var(--muted);" id="perf-info"></div>
    </div>

    <!-- 1D View -->
    <div id="view-1d">
      <div class="view-toolbar">
        <label>X 軸感測器: <select id="sel-1d-x"></select></label>
        <label><input type="checkbox" id="chk-1d-seeds" /> 顯示 10 條個別 seed</label>
        <label><input type="checkbox" id="chk-1d-density" checked /> 顯示資料密度直方圖</label>
        <label><input type="checkbox" id="chk-1d-alltasks" /> 同時顯示 4 任務</label>
      </div>
      <div class="canvas-wrap">
        <canvas id="cv-1d" height="440"></canvas>
      </div>
      <div class="legend-row">
        <span><span class="swatch" style="border-top:2px dashed var(--blue);height:0;"></span>Original r(s)</span>
        <span><span class="swatch" style="background:var(--accent);"></span>ANFIS 10-seed mean</span>
        <span><span class="swatch" style="background:rgba(221,132,82,0.3);height:10px;"></span>&plusmn; 1 std</span>
        <span><span class="swatch" style="background:rgba(196,78,82,0.2);height:10px;"></span>負權重區 (&lt; 0)</span>
        <span><span class="swatch" style="background:rgba(148,163,184,0.35);height:10px;"></span>真實參考資料密度 (N=10,000)</span>
      </div>
      <div class="hover-bar" id="hover-1d">將滑鼠移到圖上可檢視精確數值；拖動左側拉桿即時更新。</div>
    </div>

    <!-- 2D Heatmap View -->
    <div id="view-2d" style="display:none;">
      <div class="view-toolbar">
        <label>X 軸: <select id="sel-2d-x"></select></label>
        <label>Y 軸: <select id="sel-2d-y"></select></label>
        <label>右側顯示:
          <select id="sel-2d-right">
            <option value="anfis">ANFIS 10-seed Mean</option>
            <option value="diff">差值 (ANFIS - Original)</option>
            <option value="std">跨 10 Seeds 標準差 (&sigma;)</option>
          </select>
        </label>
        <label>解析度:
          <select id="sel-2d-res">
            <option value="44">44&times;44 (極速)</option>
            <option value="64" selected>64&times;64 (清晰)</option>
          </select>
        </label>
        <label><input type="checkbox" id="chk-2d-pts" checked /> 疊加真實樣本點</label>
        <label><input type="checkbox" id="chk-2d-sync" /> 兩圖共用色階</label>
      </div>
      <div class="hm-grid">
        <div><canvas id="cv-hm-left" height="410"></canvas></div>
        <div><canvas id="cv-hm-right" height="410"></canvas></div>
      </div>
      <div class="hover-bar" id="hover-2d">將滑鼠移到任一熱圖上檢視 (X, Y) 座標與兩側數值；白色/深色點為真實離線資料分布（越靠近目前其餘拉桿值的樣本越亮）。</div>
    </div>

    <!-- All Sensors Grid View -->
    <div id="view-grid" style="display:none;">
      <div class="view-toolbar">
        <label><input type="checkbox" id="chk-grid-seeds" /> 顯示個別 seed</label>
        <label><input type="checkbox" id="chk-grid-density" checked /> 顯示資料密度</label>
        <span style="color:var(--muted);font-size:12px;">點擊任意小圖可直接切換到該感測器的放大 1D 切面</span>
      </div>
      <div class="all-grid" id="all-grid-box"></div>
    </div>
  </main>
</div>

<script>
const __ANFIS_DATA__ = __DATA_JSON_PLACEHOLDER__;
__JS_CORE_MATH_PLACEHOLDER__

(function initApp() {
  const DATA = __ANFIS_DATA__;
  const refSensorNames = DATA.ref_sensor_names;
  const refNameToIdx = {};
  refSensorNames.forEach((n, i) => refNameToIdx[n] = i);
  const refFlat = b64ToFloat32(DATA.ref_samples_b64);
  const refN = DATA.ref_n;
  const refD = refSensorNames.length;

  // Decode cache: decodedCache[condKey][task] = array of decoded seed objects
  const decodedCache = {};
  function getSeeds(condKey, task) {
    if (!decodedCache[condKey]) decodedCache[condKey] = {};
    if (!decodedCache[condKey][task]) {
      const cond = DATA.conditions[condKey];
      decodedCache[condKey][task] = cond.tasks[task].map(w => decodeSeedWeights(w, cond.spec));
    }
    return decodedCache[condKey][task];
  }

  const PRESETS = [
    { name: 'Stand', vals: { torso_height: 1.25, torso_upright: 1.0, horizontal_velocity: 0.0, angmomentum: 0.0 } },
    { name: 'Walk',  vals: { torso_height: 1.25, torso_upright: 1.0, horizontal_velocity: 1.5, angmomentum: 0.0 } },
    { name: 'Run',   vals: { torso_height: 1.25, torso_upright: 1.0, horizontal_velocity: 4.0, angmomentum: 0.0 } },
    { name: 'Flip',  vals: { torso_height: 1.20, torso_upright: 1.0, horizontal_velocity: 0.5, angmomentum: 5.0 } },
    { name: 'Crouch', vals: { torso_height: 0.70, torso_upright: 0.5, horizontal_velocity: 0.0, angmomentum: 0.0 } },
    { name: 'Fallen', vals: { torso_height: 0.25, torso_upright: -0.5, horizontal_velocity: 0.0, angmomentum: 0.0 } },
    { name: 'Data Mean', vals: null },
  ];

  const STORAGE_KEY = 't62_anfis_explorer_state_v1';

  const state = {
    condKey: DATA.condition_order[0],
    task: 'stand',
    mode: '1d',
    x1d: 0,
    x2d: 0,
    y2d: 1,
    vals: [],
    hover1dX: null,
    hover2d: null,
    last1dCache: null,
    last2dCache: null,
    isDraggingSlider: false,
  };

  function savePersistedState() {
    try {
      const snap = {
        condKey: state.condKey,
        task: state.task,
        mode: state.mode,
        x1d: state.x1d,
        x2d: state.x2d,
        y2d: state.y2d,
        vals: state.vals,
      };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(snap));
    } catch (_) {}
  }

  function loadPersistedState() {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      const snap = JSON.parse(raw);
      if (!snap || !DATA.conditions[snap.condKey]) return null;
      return snap;
    } catch (_) {
      return null;
    }
  }

  // DOM refs
  const selCond = document.getElementById('sel-cond');
  const selTask = document.getElementById('sel-task');
  const sel1dX = document.getElementById('sel-1d-x');
  const sel2dX = document.getElementById('sel-2d-x');
  const sel2dY = document.getElementById('sel-2d-y');
  const sel2dRight = document.getElementById('sel-2d-right');
  const sel2dRes = document.getElementById('sel-2d-res');
  const chk1dSeeds = document.getElementById('chk-1d-seeds');
  const chk1dDensity = document.getElementById('chk-1d-density');
  const chk1dAllTasks = document.getElementById('chk-1d-alltasks');
  const chk2dPts = document.getElementById('chk-2d-pts');
  const chk2dSync = document.getElementById('chk-2d-sync');
  const chkGridSeeds = document.getElementById('chk-grid-seeds');
  const chkGridDensity = document.getElementById('chk-grid-density');
  const slidersBox = document.getElementById('sliders-box');
  const presetBtns = document.getElementById('preset-btns');
  const cv1d = document.getElementById('cv-1d');
  const cvHmLeft = document.getElementById('cv-hm-left');
  const cvHmRight = document.getElementById('cv-hm-right');
  const allGridBox = document.getElementById('all-grid-box');

  // Populate condition select
  DATA.condition_order.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = DATA.conditions[k].label;
    selCond.appendChild(opt);
  });

  // Populate presets
  PRESETS.forEach(p => {
    const b = document.createElement('button');
    b.className = 'preset-btn';
    b.textContent = p.name;
    b.addEventListener('click', () => applyPreset(p.vals));
    presetBtns.appendChild(b);
  });

  function currentCond() { return DATA.conditions[state.condKey]; }

  function defaultValsForCond(cond, task) {
    const basePreset = PRESETS.find(p => p.name.toLowerCase() === task) || PRESETS[0];
    return cond.spec.sensor_names.map((sname, i) => {
      if (basePreset.vals && sname in basePreset.vals) return basePreset.vals[sname];
      return DATA.sensor_stats[sname].mean;
    });
  }

  function applyPreset(presetMap) {
    const cond = currentCond();
    state.vals = cond.spec.sensor_names.map((sname) => {
      if (presetMap && sname in presetMap) return presetMap[sname];
      return DATA.sensor_stats[sname].mean;
    });
    syncSliderInputs();
    savePersistedState();
    scheduleRender();
  }

  function rebuildConditionUI(restoredVals) {
    const cond = currentCond();
    const names = cond.spec.sensor_names;
    if (Array.isArray(restoredVals) && restoredVals.length === names.length) {
      state.vals = restoredVals.map((v, i) => {
        const st = DATA.sensor_stats[names[i]];
        const num = Number(v);
        return Number.isFinite(num) ? Math.max(st.min, Math.min(st.max, num)) : st.mean;
      });
    } else {
      state.vals = defaultValsForCond(cond, state.task);
    }
    if (state.x1d >= names.length) state.x1d = 0;
    if (state.x2d >= names.length) state.x2d = 0;
    if (state.y2d >= names.length || state.y2d === state.x2d) state.y2d = state.x2d === 0 ? 1 : 0;

    [sel1dX, sel2dX, sel2dY].forEach(sel => {
      sel.innerHTML = '';
      names.forEach((n, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = (i >= 4 ? '[+] ' : '') + DATA.sensor_stats[n].label;
        sel.appendChild(opt);
      });
    });
    sel1dX.value = state.x1d;
    sel2dX.value = state.x2d;
    sel2dY.value = state.y2d;

    buildSliders();
    savePersistedState();
    scheduleRender();
  }

  let sharpenTimer = null;

  function buildSliders() {
    const cond = currentCond();
    const names = cond.spec.sensor_names;
    slidersBox.innerHTML = '';
    names.forEach((sname, idx) => {
      const st = DATA.sensor_stats[sname];
      const step = (st.max - st.min) / 400;
      const div = document.createElement('div');
      div.className = 'slider-item';
      div.id = `sitem-${idx}`;
      div.innerHTML = `
        <div class="slider-head">
          <div>
            <span class="sensor-name ${idx >= 4 ? 'extra' : ''}">${st.label}</span>
            <span class="axis-tag" id="stag-${idx}" style="display:none;"></span>
          </div>
          <input class="slider-num" type="number" id="snum-${idx}" step="any" min="${st.min}" max="${st.max}" value="${state.vals[idx].toFixed(3)}" />
        </div>
        <div class="slider-row">
          <input type="range" id="srng-${idx}" min="${st.min}" max="${st.max}" step="${step}" value="${state.vals[idx]}" />
        </div>
        <div class="slider-meta">
          <span>[${st.min.toFixed(2)}, ${st.max.toFixed(2)}]</span>
          <span id="spct-${idx}"></span>
        </div>
      `;
      slidersBox.appendChild(div);
      const rng = div.querySelector(`#srng-${idx}`);
      const num = div.querySelector(`#snum-${idx}`);
      rng.addEventListener('pointerdown', () => { state.isDraggingSlider = true; });
      rng.addEventListener('pointerup', () => {
        state.isDraggingSlider = false;
        if (state.mode === '2d') scheduleRender();
      });
      rng.addEventListener('input', () => {
        const v = parseFloat(rng.value);
        state.vals[idx] = v;
        num.value = v.toFixed(3);
        updateSliderBadges();
        savePersistedState();
        scheduleRender();
        if (state.mode === '2d') {
          if (sharpenTimer) clearTimeout(sharpenTimer);
          sharpenTimer = setTimeout(() => {
            state.isDraggingSlider = false;
            scheduleRender();
          }, 140);
        }
      });
      rng.addEventListener('change', () => {
        state.isDraggingSlider = false;
        savePersistedState();
        if (state.mode === '2d') scheduleRender();
      });
      num.addEventListener('change', () => {
        let v = parseFloat(num.value);
        if (!Number.isFinite(v)) return;
        v = Math.max(st.min, Math.min(st.max, v));
        state.vals[idx] = v;
        rng.value = v;
        num.value = v.toFixed(3);
        updateSliderBadges();
        savePersistedState();
        scheduleRender();
      });
    });
    updateSliderBadges();
  }

  function syncSliderInputs() {
    const names = currentCond().spec.sensor_names;
    names.forEach((_, idx) => {
      const rng = document.getElementById(`srng-${idx}`);
      const num = document.getElementById(`snum-${idx}`);
      if (rng) rng.value = state.vals[idx];
      if (num) num.value = state.vals[idx].toFixed(3);
    });
    updateSliderBadges();
  }

  function approxPercentile(sname, val) {
    const st = DATA.sensor_stats[sname];
    const edges = st.hist_edges;
    const counts = st.hist_counts;
    let total = 0, below = 0;
    for (let i = 0; i < counts.length; i++) total += counts[i];
    for (let i = 0; i < counts.length; i++) {
      if (val >= edges[i + 1]) below += counts[i];
      else if (val > edges[i]) below += counts[i] * ((val - edges[i]) / (edges[i + 1] - edges[i]));
    }
    return Math.round(100 * below / Math.max(total, 1));
  }

  function updateSliderBadges() {
    const names = currentCond().spec.sensor_names;
    names.forEach((sname, idx) => {
      const item = document.getElementById(`sitem-${idx}`);
      const tag = document.getElementById(`stag-${idx}`);
      const pct = document.getElementById(`spct-${idx}`);
      if (pct) pct.textContent = `ref ~p${approxPercentile(sname, state.vals[idx])} (mean ${DATA.sensor_stats[sname].mean.toFixed(2)})`;
      if (!item || !tag) return;
      item.classList.remove('swept');
      tag.style.display = 'none';
      tag.classList.remove('ytag');
      if (state.mode === '1d' && idx === state.x1d) {
        item.classList.add('swept');
        tag.style.display = 'inline-block';
        tag.textContent = 'X 軸掃描中';
      } else if (state.mode === '2d' && idx === state.x2d) {
        item.classList.add('swept');
        tag.style.display = 'inline-block';
        tag.textContent = 'X 軸';
      } else if (state.mode === '2d' && idx === state.y2d) {
        item.classList.add('swept');
        tag.style.display = 'inline-block';
        tag.classList.add('ytag');
        tag.textContent = 'Y 軸';
      }
    });
  }

  function computeSupport(excludeSet) {
    const cond = currentCond();
    const names = cond.spec.sensor_names;
    const active = [];
    for (let d = 0; d < names.length; d++) {
      if (excludeSet.has(d)) continue;
      const sname = names[d];
      const st = DATA.sensor_stats[sname];
      active.push({ refIdx: refNameToIdx[sname], val: state.vals[d], invStd: 1.0 / st.std });
    }
    const k = active.length;
    if (k === 0) return { minRms: 0, nearCount: refN, k: 0 };
    let minD2 = Infinity;
    let nearCount = 0;
    const threshD2 = 0.5 * 0.5 * k; // RMS z-dist <= 0.5
    for (let n = 0; n < refN; n++) {
      const base = n * refD;
      let d2 = 0.0;
      for (let j = 0; j < k; j++) {
        const a = active[j];
        const diff = (refFlat[base + a.refIdx] - a.val) * a.invStd;
        d2 += diff * diff;
      }
      if (d2 < minD2) minD2 = d2;
      if (d2 <= threshD2) nearCount++;
    }
    return { minRms: Math.sqrt(minD2 / k), nearCount, k };
  }

  function updateReadout() {
    const cond = currentCond();
    const seeds = getSeeds(state.condKey, state.task);
    const rTrue = trueReward(state.task, state.vals[0], state.vals[1], state.vals[2], state.vals[3]);
    let sum = 0, sum2 = 0;
    for (let s = 0; s < seeds.length; s++) {
      const y = evalAnfisSingle(state.vals, seeds[s], cond.spec);
      sum += y;
      sum2 += y * y;
    }
    const mean = sum / seeds.length;
    const std = Math.sqrt(Math.max(0, sum2 / seeds.length - mean * mean));
    document.getElementById('ro-true').textContent = rTrue.toFixed(3);
    document.getElementById('ro-anfis').textContent = `${mean.toFixed(3)} ± ${std.toFixed(3)}`;

    const excl = new Set();
    if (state.mode === '1d') excl.add(state.x1d);
    else if (state.mode === '2d') { excl.add(state.x2d); excl.add(state.y2d); }
    const sup = computeSupport(excl);
    const badge = document.getElementById('ro-sup-badge');
    const detail = document.getElementById('ro-sup-detail');
    const lbl = document.getElementById('ro-sup-label');
    lbl.textContent = excl.size > 0 ? '固定軸真實樣本支撐度' : '全軸真實樣本支撐度';
    const pct = (100 * sup.nearCount / refN).toFixed(1);
    badge.className = 'support-badge ' + (sup.nearCount >= 20 ? 'sup-good' : (sup.minRms < 0.9 ? 'sup-warn' : 'sup-bad'));
    if (sup.nearCount >= 20) badge.textContent = `良好 (${sup.nearCount}/${refN})`;
    else if (sup.minRms < 0.9) badge.textContent = `稀疏 (${sup.nearCount}/${refN})`;
    else badge.textContent = `空區外推! (0/${refN})`;
    detail.textContent = `固定 ${sup.k} 維：最近真實樣本 RMS z-距離 = ${sup.minRms.toFixed(2)}σ，0.5σ 內樣本佔 ${pct}%`;
  }

  function prepCanvas(cv, heightPx) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(240, Math.floor(cv.parentElement.clientWidth));
    const h = heightPx;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.height = h + 'px';
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function linspace(lo, hi, n) {
    const arr = new Float64Array(n);
    const step = (hi - lo) / (n - 1);
    for (let i = 0; i < n; i++) arr[i] = lo + i * step;
    return arr;
  }

  function compute1DCurve(cond, task, xIdx, nPts) {
    const sname = cond.spec.sensor_names[xIdx];
    const st = DATA.sensor_stats[sname];
    const xs = linspace(st.min, st.max, nPts);
    const seeds = getSeeds(cond.key, task);
    const seedYs = seeds.map(dec => evalAnfis1DSweep(state.vals, xIdx, xs, dec, cond.spec));
    const meanY = new Float64Array(nPts);
    const stdY = new Float64Array(nPts);
    const trueY = new Float64Array(nPts);
    const S = seeds.length;
    for (let i = 0; i < nPts; i++) {
      let s1 = 0, s2 = 0;
      for (let k = 0; k < S; k++) {
        const v = seedYs[k][i];
        s1 += v;
        s2 += v * v;
      }
      const m = s1 / S;
      meanY[i] = m;
      stdY[i] = Math.sqrt(Math.max(0, s2 / S - m * m));
      const h = xIdx === 0 ? xs[i] : state.vals[0];
      const up = xIdx === 1 ? xs[i] : state.vals[1];
      const vx = xIdx === 2 ? xs[i] : state.vals[2];
      const ang = xIdx === 3 ? xs[i] : state.vals[3];
      trueY[i] = trueReward(task, h, up, vx, ang);
    }
    return { xs, trueY, meanY, stdY, seedYs, sname, st };
  }

  function draw1DPlot(cv, heightPx, curves, opts) {
    const { ctx, w, h } = prepCanvas(cv, heightPx);
    const pad = opts.mini
      ? { l: 38, r: 10, t: 24, b: 26 }
      : { l: 56, r: 22, t: 30, b: 42 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const primary = curves[0];
    const xMin = primary.st.min, xMax = primary.st.max;
    let yMin = Infinity, yMax = -Infinity;
    curves.forEach(c => {
      for (let i = 0; i < c.xs.length; i++) {
        yMin = Math.min(yMin, c.trueY[i], c.meanY[i] - c.stdY[i]);
        yMax = Math.max(yMax, c.trueY[i], c.meanY[i] + c.stdY[i]);
        if (opts.showSeeds) {
          for (let k = 0; k < c.seedYs.length; k++) {
            yMin = Math.min(yMin, c.seedYs[k][i]);
            yMax = Math.max(yMax, c.seedYs[k][i]);
          }
        }
      }
    });
    yMin = Math.min(yMin, -0.2);
    yMax = Math.max(yMax, 1.15);
    const yPad = 0.08 * (yMax - yMin);
    yMin -= yPad; yMax += yPad;

    const sx = x => pad.l + ((x - xMin) / (xMax - xMin)) * pw;
    const sy = y => pad.t + ph - ((y - yMin) / (yMax - yMin)) * ph;

    // Grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.font = opts.mini ? '10px sans-serif' : '11px sans-serif';
    ctx.fillStyle = '#64748b';
    const nYTicks = opts.mini ? 4 : 6;
    for (let i = 0; i <= nYTicks; i++) {
      const yv = yMin + (i / nYTicks) * (yMax - yMin);
      const yp = sy(yv);
      ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(pad.l + pw, yp); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(yv.toFixed(1), pad.l - 5, yp);
    }
    const nXTicks = opts.mini ? 4 : 6;
    for (let i = 0; i <= nXTicks; i++) {
      const xv = xMin + (i / nXTicks) * (xMax - xMin);
      const xp = sx(xv);
      ctx.beginPath(); ctx.moveTo(xp, pad.t); ctx.lineTo(xp, pad.t + ph); ctx.stroke();
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(Math.abs(xv) >= 10 ? xv.toFixed(0) : xv.toFixed(2), xp, pad.t + ph + 5);
    }

    // Background reference density histogram
    if (opts.showDensity) {
      const counts = primary.st.hist_counts;
      const edges = primary.st.hist_edges;
      const maxC = Math.max(...counts, 1);
      ctx.fillStyle = 'rgba(148, 163, 184, 0.25)';
      for (let b = 0; b < counts.length; b++) {
        const x0 = sx(edges[b]), x1 = sx(edges[b + 1]);
        const bh = (counts[b] / maxC) * (ph * 0.36);
        ctx.fillRect(x0, pad.t + ph - bh, Math.max(1, x1 - x0 - 0.5), bh);
      }
    }

    // Zero line
    const y0 = sy(0);
    if (y0 >= pad.t && y0 <= pad.t + ph) {
      ctx.save();
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(pad.l + pw, y0); ctx.stroke();
      ctx.restore();
    }

    const taskColors = { stand: '#dd8452', walk: '#16a34a', run: '#9333ea', flip: '#dc2626' };

    curves.forEach((c, ci) => {
      const color = curves.length > 1 ? (taskColors[c.task] || '#dd8452') : '#dd8452';
      // Negative fill for single curve
      if (curves.length === 1 && y0 >= pad.t && y0 <= pad.t + ph) {
        ctx.fillStyle = 'rgba(196, 78, 82, 0.14)';
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < c.xs.length; i++) {
          const xp = sx(c.xs[i]);
          const yp = Math.max(y0, sy(c.meanY[i]));
          if (!started) { ctx.moveTo(xp, y0); started = true; }
          ctx.lineTo(xp, yp);
        }
        ctx.lineTo(sx(c.xs[c.xs.length - 1]), y0);
        ctx.closePath();
        ctx.fill();
      }

      // +- 1 std shaded band
      ctx.fillStyle = curves.length > 1 ? color + '22' : 'rgba(221, 132, 82, 0.22)';
      ctx.beginPath();
      for (let i = 0; i < c.xs.length; i++) {
        const xp = sx(c.xs[i]), yp = sy(c.meanY[i] + c.stdY[i]);
        if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
      }
      for (let i = c.xs.length - 1; i >= 0; i--) {
        ctx.lineTo(sx(c.xs[i]), sy(c.meanY[i] - c.stdY[i]));
      }
      ctx.closePath();
      ctx.fill();

      // Individual seeds
      if (opts.showSeeds) {
        ctx.strokeStyle = curves.length > 1 ? color + '55' : 'rgba(221, 132, 82, 0.42)';
        ctx.lineWidth = 1;
        c.seedYs.forEach(arr => {
          ctx.beginPath();
          for (let i = 0; i < c.xs.length; i++) {
            const xp = sx(c.xs[i]), yp = sy(arr[i]);
            if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
          }
          ctx.stroke();
        });
      }

      // Original r(s) dashed blue (only for single-task mode)
      if (curves.length === 1) {
        ctx.save();
        ctx.strokeStyle = '#4c72b0';
        ctx.lineWidth = opts.mini ? 1.7 : 2.2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        for (let i = 0; i < c.xs.length; i++) {
          const xp = sx(c.xs[i]), yp = sy(c.trueY[i]);
          if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
        }
        ctx.stroke();
        ctx.restore();
      }

      // ANFIS mean curve
      ctx.strokeStyle = color;
      ctx.lineWidth = opts.mini ? 2.0 : 2.6;
      ctx.beginPath();
      for (let i = 0; i < c.xs.length; i++) {
        const xp = sx(c.xs[i]), yp = sy(c.meanY[i]);
        if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
      }
      ctx.stroke();
    });

    // Current slider marker vertical line
    const curVal = state.vals[opts.xIdx];
    const xCur = sx(curVal);
    if (xCur >= pad.l && xCur <= pad.l + pw) {
      ctx.save();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(xCur, pad.t); ctx.lineTo(xCur, pad.t + ph); ctx.stroke();
      ctx.restore();
    }

    // Hover crosshair
    if (opts.hoverX != null && opts.hoverX >= xMin && opts.hoverX <= xMax) {
      const hx = sx(opts.hoverX);
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(hx, pad.t); ctx.lineTo(hx, pad.t + ph); ctx.stroke();
    }

    // Border & title
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, pw, ph);

    ctx.fillStyle = opts.xIdx >= 4 ? '#c44e52' : '#0f172a';
    ctx.font = opts.mini ? 'bold 11.5px sans-serif' : 'bold 13px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(opts.title, pad.l + pw / 2, 6);

    if (!opts.mini) {
      ctx.fillStyle = '#475569';
      ctx.font = '12px sans-serif';
      ctx.fillText(primary.st.label, pad.l + pw / 2, h - 16);
      ctx.save();
      ctx.translate(14, pad.t + ph / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText('reward / weight', 0, 0);
      ctx.restore();
    }

    // Multi-task inline legend
    if (curves.length > 1) {
      let lx = pad.l + 10;
      const ly = pad.t + 14;
      ctx.font = 'bold 11.5px sans-serif';
      ctx.textAlign = 'left';
      curves.forEach(c => {
        ctx.fillStyle = taskColors[c.task];
        ctx.fillRect(lx, ly - 4, 12, 3);
        ctx.fillText(c.task, lx + 16, ly - 7);
        lx += 64;
      });
    }

    return { pad, pw, ph, xMin, xMax, yMin, yMax };
  }

  // Diverging RdBu-like colormap (negative blue -> zero white/light -> positive orange/red) or Viridis
  function colorDiverging(t) {
    // t in [0, 1], 0.5 is neutral
    const clamped = Math.max(0, Math.min(1, t));
    if (clamped < 0.5) {
      const u = clamped * 2.0; // 0=deep blue, 1=light neutral
      return [
        Math.round(33 + u * (247 - 33)),
        Math.round(102 + u * (247 - 102)),
        Math.round(172 + u * (247 - 172)),
      ];
    }
    const u = (clamped - 0.5) * 2.0; // 0=neutral, 1=deep red-orange
    return [
      Math.round(247 + u * (178 - 247)),
      Math.round(247 + u * (24 - 247)),
      Math.round(247 + u * (43 - 247)),
    ];
  }

  function colorViridis(t) {
    const stops = [
      [68, 1, 84],
      [59, 82, 139],
      [33, 145, 140],
      [94, 201, 98],
      [253, 231, 37],
    ];
    const u = Math.max(0, Math.min(1, t)) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(u));
    const f = u - i;
    return [
      Math.round(stops[i][0] + f * (stops[i + 1][0] - stops[i][0])),
      Math.round(stops[i][1] + f * (stops[i + 1][1] - stops[i][1])),
      Math.round(stops[i][2] + f * (stops[i + 1][2] - stops[i][2])),
    ];
  }

  function drawHeatmap(cv, grid, nx, ny, meta, opts) {
    const { ctx, w, h } = prepCanvas(cv, 410);
    const pad = { l: 54, r: 62, t: 30, b: 42 };
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);

    // Offscreen image for crisp pixel heatmap
    const off = document.createElement('canvas');
    off.width = nx;
    off.height = ny;
    const octx = off.getContext('2d');
    const img = octx.createImageData(nx, ny);
    const data = img.data;
    const vMin = opts.vMin, vMax = opts.vMax;
    const span = Math.max(vMax - vMin, 1e-6);
    const cmap = opts.diverging ? colorDiverging : colorViridis;

    for (let j = 0; j < ny; j++) {
      // Canvas y=0 is top = high Y value
      const rowY = ny - 1 - j;
      for (let i = 0; i < nx; i++) {
        const val = grid[rowY * nx + i];
        let t;
        if (opts.diverging) {
          const absMax = Math.max(Math.abs(vMin), Math.abs(vMax), 1e-6);
          t = 0.5 + 0.5 * (val / absMax);
        } else {
          t = (val - vMin) / span;
        }
        const rgb = cmap(t);
        const idx = (j * nx + i) * 4;
        data[idx] = rgb[0];
        data[idx + 1] = rgb[1];
        data[idx + 2] = rgb[2];
        data[idx + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, pad.l, pad.t, pw, ph);

    const sx = x => pad.l + ((x - meta.xMin) / (meta.xMax - meta.xMin)) * pw;
    const sy = y => pad.t + ph - ((y - meta.yMin) / (meta.yMax - meta.yMin)) * ph;

    // Overlay real reference dataset points
    if (opts.showPoints) {
      const ixRef = refNameToIdx[meta.xName];
      const iyRef = refNameToIdx[meta.yName];
      const cond = currentCond();
      const names = cond.spec.sensor_names;
      const fixedSpecs = [];
      for (let d = 0; d < names.length; d++) {
        if (d === state.x2d || d === state.y2d) continue;
        fixedSpecs.push({ rIdx: refNameToIdx[names[d]], val: state.vals[d], invStd: 1.0 / DATA.sensor_stats[names[d]].std });
      }
      const k = fixedSpecs.length;
      for (let n = 0; n < refN; n++) {
        const base = n * refD;
        const xv = refFlat[base + ixRef];
        const yv = refFlat[base + iyRef];
        if (xv < meta.xMin || xv > meta.xMax || yv < meta.yMin || yv > meta.yMax) continue;
        let near = false;
        if (k > 0) {
          let d2 = 0;
          for (let f = 0; f < k; f++) {
            const diff = (refFlat[base + fixedSpecs[f].rIdx] - fixedSpecs[f].val) * fixedSpecs[f].invStd;
            d2 += diff * diff;
          }
          near = (d2 / k) <= 0.36; // RMS <= 0.6 sigma
        }
        ctx.fillStyle = near ? 'rgba(15, 23, 42, 0.65)' : 'rgba(15, 23, 42, 0.10)';
        ctx.fillRect(sx(xv) - 1, sy(yv) - 1, near ? 2.6 : 1.6, near ? 2.6 : 1.6);
      }
    }

    // Current slider point crosshair
    const cx = sx(state.vals[state.x2d]);
    const cy = sy(state.vals[state.y2d]);
    if (cx >= pad.l && cx <= pad.l + pw && cy >= pad.t && cy <= pad.t + ph) {
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();
    }

    // Hover marker
    if (opts.hover) {
      const hx = sx(opts.hover.x), hy = sy(opts.hover.y);
      ctx.strokeStyle = 'rgba(15,23,42,0.75)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hx, pad.t); ctx.lineTo(hx, pad.t + ph);
      ctx.moveTo(pad.l, hy); ctx.lineTo(pad.l + pw, hy);
      ctx.stroke();
    }

    // Axes ticks & labels
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l, pad.t, pw, ph);
    ctx.fillStyle = '#475569';
    ctx.font = '11px sans-serif';
    for (let i = 0; i <= 5; i++) {
      const xv = meta.xMin + (i / 5) * (meta.xMax - meta.xMin);
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(Math.abs(xv) >= 10 ? xv.toFixed(0) : xv.toFixed(2), sx(xv), pad.t + ph + 5);
      const yv = meta.yMin + (i / 5) * (meta.yMax - meta.yMin);
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(Math.abs(yv) >= 10 ? yv.toFixed(0) : yv.toFixed(2), pad.l - 5, sy(yv));
    }

    // Colorbar
    const cbX = pad.l + pw + 10;
    const cbW = 12;
    for (let py = 0; py < ph; py++) {
      const frac = 1.0 - py / Math.max(ph - 1, 1);
      const rgb = cmap(frac);
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(cbX, pad.t + py, cbW, 1.5);
    }
    ctx.strokeStyle = '#94a3b8';
    ctx.strokeRect(cbX, pad.t, cbW, ph);
    ctx.fillStyle = '#475569';
    ctx.font = '10.5px monospace';
    ctx.textAlign = 'left';
    const loLbl = opts.diverging ? -Math.max(Math.abs(vMin), Math.abs(vMax)) : vMin;
    const hiLbl = opts.diverging ? Math.max(Math.abs(vMin), Math.abs(vMax)) : vMax;
    ctx.textBaseline = 'top'; ctx.fillText(hiLbl.toFixed(2), cbX + cbW + 4, pad.t);
    ctx.textBaseline = 'middle'; ctx.fillText(((loLbl + hiLbl) / 2).toFixed(2), cbX + cbW + 4, pad.t + ph / 2);
    ctx.textBaseline = 'bottom'; ctx.fillText(loLbl.toFixed(2), cbX + cbW + 4, pad.t + ph);

    // Titles
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 12.5px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText(opts.title, pad.l + pw / 2, 7);
    ctx.fillStyle = '#475569';
    ctx.font = '11.5px sans-serif';
    ctx.fillText(meta.xLabel, pad.l + pw / 2, h - 16);
    ctx.save();
    ctx.translate(13, pad.t + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(meta.yLabel, 0, 0);
    ctx.restore();

    return { pad, pw, ph };
  }

  function redraw1DFromCache() {
    const cache = state.last1dCache;
    if (!cache) return;
    const cond = currentCond();
    const title = chk1dAllTasks.checked
      ? `${cond.label} | All 4 Tasks vs ${cache.curves[0].st.label}`
      : `${cond.label} | ${state.task} vs ${cache.curves[0].st.label}`;
    cache.geom = draw1DPlot(cv1d, 440, cache.curves, {
      mini: false,
      xIdx: state.x1d,
      title,
      showSeeds: chk1dSeeds.checked,
      showDensity: chk1dDensity.checked,
      hoverX: state.hover1dX,
    });
  }

  function render1D() {
    const t0 = performance.now();
    const cond = currentCond();
    const tasksToPlot = chk1dAllTasks.checked ? DATA.tasks : [state.task];
    const curves = tasksToPlot.map(t => {
      const c = compute1DCurve(cond, t, state.x1d, 200);
      c.task = t;
      return c;
    });
    state.last1dCache = { curves, geom: null };
    redraw1DFromCache();
    document.getElementById('perf-info').textContent = `計算耗時 ${(performance.now() - t0).toFixed(1)} ms`;
  }

  function redraw2DFromCache() {
    const cache = state.last2dCache;
    if (!cache) return;
    const { res, trueGrid, meanGrid, stdGrid, diffGrid, meta, stats } = cache;
    const rightMode = sel2dRight.value;
    const syncScale = chk2dSync.checked && rightMode === 'anfis';
    const leftMin = syncScale ? Math.min(stats.tMin, stats.aMin) : 0.0;
    const leftMax = syncScale ? Math.max(stats.tMax, stats.aMax) : 1.0;

    cache.gLeft = drawHeatmap(cvHmLeft, trueGrid, res, res, meta, {
      title: `Original r(s) — ${state.task}`,
      vMin: leftMin, vMax: leftMax,
      diverging: syncScale && leftMin < 0,
      showPoints: chk2dPts.checked,
      hover: state.hover2d,
    });

    let rightGrid = meanGrid, rTitle = `ANFIS r~(s) 10-seed Mean — ${state.task}`, rMin = stats.aMin, rMax = stats.aMax, rDiv = true;
    if (rightMode === 'diff') {
      rightGrid = diffGrid; rTitle = `Diff (ANFIS - Original) — ${state.task}`; rMin = stats.dMin; rMax = stats.dMax; rDiv = true;
    } else if (rightMode === 'std') {
      rightGrid = stdGrid; rTitle = `10-Seed Std σ — ${state.task}`; rMin = 0; rMax = Math.max(stats.sMax, 0.1); rDiv = false;
    } else if (syncScale) {
      rMin = leftMin; rMax = leftMax; rDiv = leftMin < 0;
    }

    cache.gRight = drawHeatmap(cvHmRight, rightGrid, res, res, meta, {
      title: rTitle,
      vMin: rMin, vMax: rMax,
      diverging: rDiv,
      showPoints: chk2dPts.checked,
      hover: state.hover2d,
    });
  }

  function render2D() {
    const t0 = performance.now();
    const cond = currentCond();
    const ix = state.x2d, iy = state.y2d;
    const xName = cond.spec.sensor_names[ix];
    const yName = cond.spec.sensor_names[iy];
    const stX = DATA.sensor_stats[xName];
    const stY = DATA.sensor_stats[yName];
    const fullRes = parseInt(sel2dRes.value, 10);
    const res = state.isDraggingSlider ? Math.min(28, fullRes) : fullRes;
    const xs = linspace(stX.min, stX.max, res);
    const ys = linspace(stY.min, stY.max, res);

    const seeds = getSeeds(state.condKey, state.task);
    const S = seeds.length;
    const N = res * res;
    const sum = new Float64Array(N);
    const sum2 = new Float64Array(N);
    for (let s = 0; s < S; s++) {
      const g = evalAnfis2DGrid(state.vals, ix, iy, xs, ys, seeds[s], cond.spec);
      for (let k = 0; k < N; k++) { sum[k] += g[k]; sum2[k] += g[k] * g[k]; }
    }
    const trueGrid = new Float64Array(N);
    const meanGrid = new Float64Array(N);
    const stdGrid = new Float64Array(N);
    const diffGrid = new Float64Array(N);
    let tMin = Infinity, tMax = -Infinity, aMin = Infinity, aMax = -Infinity;
    let dMin = Infinity, dMax = -Infinity, sMax = 0;
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const k = j * res + i;
        const m = sum[k] / S;
        const sd = Math.sqrt(Math.max(0, sum2[k] / S - m * m));
        const h = ix === 0 ? xs[i] : (iy === 0 ? ys[j] : state.vals[0]);
        const up = ix === 1 ? xs[i] : (iy === 1 ? ys[j] : state.vals[1]);
        const vx = ix === 2 ? xs[i] : (iy === 2 ? ys[j] : state.vals[2]);
        const ang = ix === 3 ? xs[i] : (iy === 3 ? ys[j] : state.vals[3]);
        const r = trueReward(state.task, h, up, vx, ang);
        trueGrid[k] = r;
        meanGrid[k] = m;
        stdGrid[k] = sd;
        diffGrid[k] = m - r;
        if (r < tMin) tMin = r; if (r > tMax) tMax = r;
        if (m < aMin) aMin = m; if (m > aMax) aMax = m;
        if (m - r < dMin) dMin = m - r; if (m - r > dMax) dMax = m - r;
        if (sd > sMax) sMax = sd;
      }
    }

    const meta = {
      xName, yName, xLabel: stX.label, yLabel: stY.label,
      xMin: stX.min, xMax: stX.max, yMin: stY.min, yMax: stY.max,
    };
    state.last2dCache = {
      res, xs, ys, trueGrid, meanGrid, stdGrid, diffGrid, meta,
      stats: { tMin, tMax, aMin, aMax, dMin, dMax, sMax },
      gLeft: null, gRight: null,
    };
    redraw2DFromCache();
    document.getElementById('perf-info').textContent = `${res}×${res} × ${S} seeds 耗時 ${(performance.now() - t0).toFixed(1)} ms`;
  }

  function renderAllGrid() {
    const t0 = performance.now();
    const cond = currentCond();
    const names = cond.spec.sensor_names;
    if (allGridBox.childElementCount !== names.length) {
      allGridBox.innerHTML = '';
      names.forEach((sname, idx) => {
        const card = document.createElement('div');
        card.className = 'mini-card';
        card.innerHTML = `<canvas id="cv-mini-${idx}" height="215"></canvas>`;
        card.addEventListener('click', () => {
          state.x1d = idx;
          sel1dX.value = idx;
          setMode('1d');
        });
        allGridBox.appendChild(card);
      });
    }
    names.forEach((sname, idx) => {
      const c = compute1DCurve(cond, state.task, idx, 120);
      c.task = state.task;
      const cv = document.getElementById(`cv-mini-${idx}`);
      draw1DPlot(cv, 215, [c], {
        mini: true,
        xIdx: idx,
        title: c.st.label,
        showSeeds: chkGridSeeds.checked,
        showDensity: chkGridDensity.checked,
        hoverX: null,
      });
    });
    document.getElementById('perf-info').textContent = `全 ${names.length} 感測器同步更新 ${(performance.now() - t0).toFixed(1)} ms`;
  }

  let rafId = null;
  function scheduleRender() {
    if (rafId != null) return;
    rafId = requestAnimationFrame(() => {
      rafId = null;
      updateReadout();
      if (state.mode === '1d') render1D();
      else if (state.mode === '2d') render2D();
      else renderAllGrid();
    });
  }

  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
    document.getElementById('view-1d').style.display = mode === '1d' ? '' : 'none';
    document.getElementById('view-2d').style.display = mode === '2d' ? '' : 'none';
    document.getElementById('view-grid').style.display = mode === 'grid' ? '' : 'none';
    updateSliderBadges();
    savePersistedState();
    scheduleRender();
  }

  // Event listeners
  document.querySelectorAll('.tab-btn').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  selCond.addEventListener('change', () => { state.condKey = selCond.value; rebuildConditionUI(null); });
  selTask.addEventListener('change', () => { state.task = selTask.value; savePersistedState(); scheduleRender(); });
  sel1dX.addEventListener('change', () => { state.x1d = parseInt(sel1dX.value, 10); updateSliderBadges(); savePersistedState(); scheduleRender(); });
  sel2dX.addEventListener('change', () => {
    const v = parseInt(sel2dX.value, 10);
    if (v === state.y2d) { state.y2d = state.x2d; sel2dY.value = state.y2d; }
    state.x2d = v; updateSliderBadges(); savePersistedState(); scheduleRender();
  });
  sel2dY.addEventListener('change', () => {
    const v = parseInt(sel2dY.value, 10);
    if (v === state.x2d) { state.x2d = state.y2d; sel2dX.value = state.x2d; }
    state.y2d = v; updateSliderBadges(); savePersistedState(); scheduleRender();
  });
  [sel2dRight, chk2dPts, chk2dSync].forEach(el => el.addEventListener('change', redraw2DFromCache));
  [sel2dRes, chk1dSeeds, chk1dDensity, chk1dAllTasks, chkGridSeeds, chkGridDensity]
    .forEach(el => el.addEventListener('change', scheduleRender));
  window.addEventListener('resize', scheduleRender);

  // 1D hover crosshair (uses cached curves — no recomputation)
  cv1d.addEventListener('mousemove', e => {
    const cache = state.last1dCache;
    if (!cache || !cache.geom) return;
    const rect = cv1d.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const { pad, pw, xMin, xMax } = cache.geom;
    if (px < pad.l || px > pad.l + pw) { state.hover1dX = null; redraw1DFromCache(); return; }
    const xVal = xMin + ((px - pad.l) / pw) * (xMax - xMin);
    state.hover1dX = xVal;
    const c0 = cache.curves[0];
    const idx = Math.max(0, Math.min(c0.xs.length - 1, Math.round(((xVal - xMin) / (xMax - xMin)) * (c0.xs.length - 1))));
    const bar = document.getElementById('hover-1d');
    if (cache.curves.length === 1) {
      bar.textContent = `${c0.st.label} = ${c0.xs[idx].toFixed(3)} | Original r(s) = ${c0.trueY[idx].toFixed(3)} | ANFIS = ${c0.meanY[idx].toFixed(3)} ± ${c0.stdY[idx].toFixed(3)}`;
    } else {
      bar.textContent = `${c0.st.label} = ${c0.xs[idx].toFixed(3)} | ` +
        cache.curves.map(c => `${c.task}: ${c.meanY[idx].toFixed(2)}±${c.stdY[idx].toFixed(2)}`).join(' | ');
    }
    redraw1DFromCache();
  });
  cv1d.addEventListener('mouseleave', () => { state.hover1dX = null; redraw1DFromCache(); });

  // 2D hover crosshair on either canvas (uses cached grid — no recomputation)
  [cvHmLeft, cvHmRight].forEach((cv, which) => {
    cv.addEventListener('mousemove', e => {
      const cache = state.last2dCache;
      if (!cache) return;
      const g = which === 0 ? cache.gLeft : cache.gRight;
      if (!g) return;
      const rect = cv.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      if (px < g.pad.l || px > g.pad.l + g.pw || py < g.pad.t || py > g.pad.t + g.ph) {
        state.hover2d = null; redraw2DFromCache(); return;
      }
      const fx = (px - g.pad.l) / g.pw;
      const fy = 1.0 - (py - g.pad.t) / g.ph;
      const xv = cache.meta.xMin + fx * (cache.meta.xMax - cache.meta.xMin);
      const yv = cache.meta.yMin + fy * (cache.meta.yMax - cache.meta.yMin);
      state.hover2d = { x: xv, y: yv };
      const i = Math.max(0, Math.min(cache.res - 1, Math.round(fx * (cache.res - 1))));
      const j = Math.max(0, Math.min(cache.res - 1, Math.round(fy * (cache.res - 1))));
      const k = j * cache.res + i;
      document.getElementById('hover-2d').textContent =
        `${cache.meta.xLabel}=${xv.toFixed(3)}, ${cache.meta.yLabel}=${yv.toFixed(3)} | ` +
        `Original r(s)=${cache.trueGrid[k].toFixed(3)} | ANFIS=${cache.meanGrid[k].toFixed(3)} ± ${cache.stdGrid[k].toFixed(3)} | ` +
        `Diff=${cache.diffGrid[k] >= 0 ? '+' : ''}${cache.diffGrid[k].toFixed(3)}`;
      redraw2DFromCache();
    });
    cv.addEventListener('mouseleave', () => { state.hover2d = null; redraw2DFromCache(); });
  });

  // Boot (restore previous slider state if an external live-reload refreshed the tab)
  const saved = loadPersistedState();
  if (saved) {
    state.condKey = saved.condKey;
    state.task = DATA.tasks.includes(saved.task) ? saved.task : 'stand';
    state.mode = ['1d', '2d', 'grid'].includes(saved.mode) ? saved.mode : '1d';
    state.x1d = Number.isInteger(saved.x1d) ? saved.x1d : 0;
    state.x2d = Number.isInteger(saved.x2d) ? saved.x2d : 0;
    state.y2d = Number.isInteger(saved.y2d) ? saved.y2d : 1;
    selCond.value = state.condKey;
    selTask.value = state.task;
    rebuildConditionUI(saved.vals);
    setMode(state.mode);
  } else {
    rebuildConditionUI(null);
  }
})();
</script>
</body>
</html>
"""


def build_html(payload: Dict[str, Any]) -> str:
    """Embed exported model payload and pure-JS ANFIS engine into a standalone HTML page."""
    data_json = json.dumps(payload, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA_JSON_PLACEHOLDER__", data_json).replace(
        "__JS_CORE_MATH_PLACEHOLDER__", JS_CORE_MATH
    )


def extract_payload_from_html(html_path: Path) -> Dict[str, Any]:
    """Read the embedded __ANFIS_DATA__ JSON back out of a previously built explorer HTML."""
    import re

    text = html_path.read_text(encoding="utf-8")
    match = re.search(r"const __ANFIS_DATA__ = (\{.*?\});\n", text, re.S)
    if match is None:
        raise ValueError(f"Could not find embedded __ANFIS_DATA__ in {html_path}")
    return json.loads(match.group(1))


def main() -> None:
    parser = ArgumentParser(description="Build standalone interactive ANFIS explorer HTML")
    parser.add_argument("--artifacts_dir", type=Path, default=None)
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument(
        "--rebuild_from_html",
        type=Path,
        default=None,
        help="Reuse embedded model payload from an existing HTML to refresh UI/JS instantly",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tag", type=str, default="final", choices=["final", "best"])
    parser.add_argument("--max_ref_samples", type=int, default=2000)
    args = parser.parse_args()

    if args.rebuild_from_html is not None:
        payload = extract_payload_from_html(args.rebuild_from_html)
    else:
        if args.artifacts_dir is None or args.dataset_path is None:
            raise SystemExit("Provide --artifacts_dir and --dataset_path (or --rebuild_from_html)")
        payload = build_explorer_payload(
            artifacts_dir=args.artifacts_dir,
            dataset_path=args.dataset_path,
            tag=args.tag,
            max_ref_samples=args.max_ref_samples,
        )
    html = build_html(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024.0
    print(
        f"Wrote {args.out} ({size_kb:.1f} KiB, {len(payload['condition_order'])} conditions, "
        f"{payload['ref_n']} reference points)"
    )


if __name__ == "__main__":
    main()
