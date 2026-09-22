"""V6.3 retina propagation diagnostic.

This program is deliberately diagnostic only.  It does not capture the screen,
send HTTP triggers, run a motor decoder, or change connectome signs or gains.
It uses the same ContinuousBrainSession source injection path as live_brain.py
and compares a small retina train with the known LPLC2/LC4 control.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
W_SYN_DIAGNOSTIC = 0.110  # existing project value; this script does not tune it
CONTROL_RATE_HZ = 40.0     # same short stimulus rate used by live_brain.py
CHUNK_MS = 20.0
CONTROL_SECONDS = 2.0
RETINA_SAMPLE_SIZE = 100
RNG_SEED = 101


def block(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def safe_series(df: pd.DataFrame, name: str, default: str = "unknown") -> pd.Series:
    if name in df.columns:
        return df[name].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def load_inputs(root: Path):
    from lif_engine_realtime import RealtimeMaleCNSLIF

    map_path = root / "data" / "processed" / "retina_spatial_map_v1.npz"
    neurons_path = root / "data" / "processed" / "simulation-neurons.parquet"
    edges_path = root / "data" / "processed" / "simulation-edges.parquet"
    annotation_path = root / "data" / "raw" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    if not map_path.exists():
        raise FileNotFoundError(map_path)
    if not neurons_path.exists():
        raise FileNotFoundError(neurons_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    spatial = np.load(map_path, allow_pickle=False)
    source_indices = np.asarray(spatial["neuron_idx"], dtype=np.int32)
    source_indices = np.unique(source_indices)
    neurons = pd.read_parquet(neurons_path)
    # Annotation is used only to enrich direct-target labels.  Missing optional
    # annotation columns never stop the diagnostic.
    try:
        annotations = pd.read_feather(annotation_path) if annotation_path.exists() else pd.DataFrame()
    except Exception as exc:
        print(f"annotation load skipped: {type(exc).__name__}: {exc}")
        annotations = pd.DataFrame()
    brain = RealtimeMaleCNSLIF(
        matrix_file=str(root / "data" / "processed" / "lif-connectome-csr.npz"),
        neurons_file=str(neurons_path),
    )
    brain.w_syn = W_SYN_DIAGNOSTIC
    return map_path, neurons_path, edges_path, spatial, source_indices, neurons, annotations, brain


def read_retina_edges(path: Path, source_set: set[int]) -> pd.DataFrame:
    """Read only relevant rows in batches; never loads a raw syn-points file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # The project runtime normally has pyarrow.  This fallback keeps the
        # diagnostic readable if an alternate parquet engine is configured.
        all_edges = pd.read_parquet(path)
        return all_edges[all_edges["pre_idx"].isin(source_set)].copy()

    pf = pq.ParquetFile(path)
    wanted = ["pre_idx", "post_idx", "pre_nt", "nt_role", "fast_weight"]
    cols = [c for c in wanted if c in pf.schema_arrow.names]
    parts: List[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        frame = batch.to_pandas()
        part = frame[frame["pre_idx"].isin(source_set)]
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def print_distribution(label: str, series: pd.Series) -> None:
    print(f"{label}:")
    if series.empty:
        print("  (no rows)")
        return
    for key, value in series.fillna("unknown").astype(str).value_counts(dropna=False).items():
        print(f"  {key}: {int(value)}")


def edge_analysis(edges: pd.DataFrame, source_indices: np.ndarray, neurons: pd.DataFrame, annotations: pd.DataFrame):
    block("RETINA TRANSMITTERS")
    print(f"retina source neurons: {len(source_indices)}")
    print(f"total outgoing edge rows: {len(edges)}")
    if edges.empty:
        print("No matching outgoing edges were found.")
        return {}
    print(f"unique postsynaptic targets: {edges['post_idx'].nunique()}")
    print_distribution("pre_nt distribution", edges.get("pre_nt", pd.Series(dtype=object)))
    print_distribution("nt_role distribution", edges.get("nt_role", pd.Series(dtype=object)))

    weights = pd.to_numeric(edges.get("fast_weight", pd.Series(dtype=float)), errors="coerce").dropna()
    block("RETINA EDGE SIGNS")
    if weights.empty:
        print("fast_weight: unavailable")
    else:
        print(f"positive: {int((weights > 0).sum())}")
        print(f"negative: {int((weights < 0).sum())}")
        print(f"zero: {int((weights == 0).sum())}")
        print(f"min={weights.min()} max={weights.max()} mean={weights.mean()} median={weights.median()}")
    nt_text = edges.astype(str).agg(" ".join, axis=1).str.lower()
    hist = nt_text.str.contains("histamine", na=False)
    print(f"histamine-labelled edge rows: {int(hist.sum())} / {len(edges)} ({100.0 * hist.mean():.4f}%)")
    per_source = edges.groupby("pre_idx").size().reindex(source_indices, fill_value=0)
    print("outgoing edges/source: min=%d max=%d mean=%.3f median=%.3f" %
          (per_source.min(), per_source.max(), per_source.mean(), per_source.median()))

    post = edges[["post_idx"]].drop_duplicates().copy()
    post["post_idx"] = post["post_idx"].astype(np.int64)
    sim = neurons.copy()
    idx_col = "neuron_idx" if "neuron_idx" in sim.columns else "index"
    sim[idx_col] = pd.to_numeric(sim[idx_col], errors="coerce")
    post = post.merge(sim, left_on="post_idx", right_on=idx_col, how="left")
    if not annotations.empty and "bodyId" in post.columns and "bodyId" in annotations.columns:
        ann_cols = [c for c in ("bodyId", "class", "subclass", "superclass", "system", "type") if c in annotations.columns]
        if len(ann_cols) > 1:
            ann = annotations[ann_cols].drop_duplicates("bodyId")
            post = post.merge(ann, on="bodyId", how="left", suffixes=("", "_annotation"))
    block("DIRECT TARGETS")
    for col in ("superclass", "class", "subclass", "system"):
        if col in post.columns:
            print_distribution(col, post[col])
    return {"post_targets": post, "per_source": per_source, "weights": weights}


def population_masks(brain, neurons: pd.DataFrame, source_indices: np.ndarray) -> Dict[str, np.ndarray]:
    n = brain.N
    masks: Dict[str, np.ndarray] = {"retina source": np.zeros(n, dtype=bool)}
    valid = source_indices[(source_indices >= 0) & (source_indices < n)]
    masks["retina source"][valid] = True
    if "superclass" in neurons.columns and "neuron_idx" in neurons.columns:
        sc = safe_series(neurons, "superclass").str.lower()
        idx = pd.to_numeric(neurons["neuron_idx"], errors="coerce").fillna(-1).astype(int)
        for name in ("ol_intrinsic", "visual_projection", "ol_sensory", "descending_neuron"):
            mask = np.zeros(n, dtype=bool)
            ids = idx[sc.eq(name)].to_numpy()
            ids = ids[(ids >= 0) & (ids < n)]
            mask[ids] = True
            masks[name] = mask
    masks["motor"] = np.asarray(brain.motor_mask, dtype=bool)
    claimed = np.zeros(n, dtype=bool)
    for m in masks.values():
        claimed |= m
    masks["other"] = ~claimed
    return masks


def configure_session(brain, source_indices: np.ndarray):
    from brain_session import ContinuousBrainSession

    session = ContinuousBrainSession(brain, source_types=[], seed=RNG_SEED)
    session.source_indices = np.asarray(source_indices, dtype=np.int32)
    session.source_mask = np.zeros(brain.N, dtype=bool)
    valid = session.source_indices[(session.source_indices >= 0) & (session.source_indices < brain.N)]
    session.source_mask[valid] = True
    return session


def run_train(brain, source_indices: np.ndarray, duration: float, seed: int, masks):
    session = configure_session(brain, source_indices)
    session.reset(seed=seed)
    # Use the actual public session.step/create_schedule path.  No private
    # simulator function is called and no sign/gain is altered.
    chunks = max(1, int(math.ceil(duration * 1000.0 / CHUNK_MS)))
    totals = Counter()
    source_spikes = 0
    state_check = None
    first_targets = None
    valid_targets = np.flatnonzero(~masks["retina source"])
    for i in range(chunks):
        if i == 0:
            session.step(0.0, 1.0)
            targets = valid_targets[: min(200, len(valid_targets))]
            before_v = session.v[targets].copy()
            before_g = session.g[targets].copy()
        result = session.step(CONTROL_RATE_HZ, CHUNK_MS)
        counts = np.asarray(result["spike_counts"], dtype=np.int64)
        source_spikes += int(counts[masks["retina source"]].sum())
        totals["forced_source_events"] += int(result["stimulus_events"])
        totals["network_spikes"] += int(result["total_spikes"])
        totals["descending_spikes"] += int(result["descending_spikes"])
        totals["motor_spikes"] += int(result["motor_spikes"])
        for name, mask in masks.items():
            totals[name] += int(counts[mask].sum())
        if i == 0:
            after_v = session.v[targets].copy()
            after_g = session.g[targets].copy()
            dv = after_v - before_v
            dg = after_g - before_g
            first_targets = targets
            state_check = {
                "target_count": int(len(targets)),
                "nonzero_dv": int(np.count_nonzero(np.abs(dv) > 1e-8)),
                "nonzero_dg": int(np.count_nonzero(np.abs(dg) > 1e-8)),
                "max_abs_dv": float(np.max(np.abs(dv))) if len(dv) else 0.0,
                "max_abs_dg": float(np.max(np.abs(dg))) if len(dg) else 0.0,
            }
    totals["source_output_spikes"] = source_spikes
    totals["secondary_network_spikes"] = totals["network_spikes"] - source_spikes
    return totals, state_check, session


def print_population_totals(totals: Counter) -> None:
    for name in ("retina source", "ol_intrinsic", "visual_projection", "ol_sensory", "descending_neuron", "motor", "other"):
        print(f"{name}: {int(totals[name])}")


def main() -> int:
    root = PROJECT_ROOT
    print(f"V6.3 diagnostic root: {root}")
    try:
        map_path, neurons_path, edges_path, spatial, retina_indices, neurons, annotations, brain = load_inputs(root)
    except Exception as exc:
        print(f"INPUT ERROR: {type(exc).__name__}: {exc}")
        return 2

    print(f"map: {map_path}")
    print(f"neurons: {neurons_path}")
    print(f"edges: {edges_path}")
    print(f"brain N={brain.N}, w_syn={brain.w_syn} (existing value, unchanged)")
    valid_retina = retina_indices[(retina_indices >= 0) & (retina_indices < brain.N)]
    print(f"retina indices: {len(retina_indices)} total, {len(valid_retina)} valid")
    edge_info = edge_analysis(read_retina_edges(edges_path, set(map(int, valid_retina))), valid_retina, neurons, annotations)
    masks = population_masks(brain, neurons, valid_retina)

    block("SOURCE INJECTION CHECK")
    rng = np.random.default_rng(RNG_SEED)
    sample = rng.choice(valid_retina, size=min(RETINA_SAMPLE_SIZE, len(valid_retina)), replace=False).astype(np.int32)
    retina_totals, retina_state, _ = run_train(brain, sample, CONTROL_SECONDS, RNG_SEED, masks)
    print(f"selected retina sources: {len(sample)}")
    print(f"forced source events: {retina_totals['forced_source_events']}")
    print(f"source output spikes: {retina_totals['source_output_spikes']}")
    print(f"network spikes: {retina_totals['network_spikes']}")

    block("POSTSYNAPTIC STATE CHECK")
    if retina_state is None:
        print("No state sample was collected.")
    else:
        for key, value in retina_state.items():
            print(f"{key}: {value}")
        if retina_state["nonzero_dg"] or retina_state["nonzero_dv"]:
            print("Interpretation: at least one public target state changed during the first injected chunk.")
        else:
            print("Interpretation: no sampled public target state changed during the first injected chunk.")

    block("KNOWN POSITIVE CONTROL")
    positive_indices = np.flatnonzero(brain.types.isin(["LPLC2", "LC4"]).to_numpy()).astype(np.int32)
    if len(positive_indices):
        positive_totals, positive_state, _ = run_train(brain, positive_indices, CONTROL_SECONDS, RNG_SEED, masks)
        print(f"LPLC2/LC4 source neurons: {len(positive_indices)}")
        print(f"positive-control forced source events: {positive_totals['forced_source_events']}")
        print(f"positive-control source spikes: {positive_totals['source_output_spikes']}")
        print(f"positive-control secondary network spikes: {positive_totals['secondary_network_spikes']}")
        print(f"positive-control descending spikes: {positive_totals['descending_spikes']}")
        print(f"positive-control target state: {positive_state}")
    else:
        positive_totals = Counter()
        print("No LPLC2/LC4 neurons were found; positive control unavailable.")

    block("SECONDARY SPIKES")
    print(f"forced_source_spikes (stimulus events): {retina_totals['forced_source_events']}")
    print(f"retina source output spikes: {retina_totals['source_output_spikes']}")
    print(f"secondary_network_spikes: {retina_totals['secondary_network_spikes']}")
    print_population_totals(retina_totals)

    block("DIAGNOSIS")
    if not len(positive_indices) or positive_totals["secondary_network_spikes"] == 0:
        diagnosis = "DIAGNOSIS C: Known-positive control also fails; injection/API integration problem."
    elif retina_totals["secondary_network_spikes"] > 0:
        diagnosis = "DIAGNOSIS D: Secondary retinal spikes exist but previous accounting hid them."
    elif retina_state and (retina_state["nonzero_dg"] > 0 or retina_state["nonzero_dv"] > 0):
        diagnosis = "DIAGNOSIS A: Source injection propagates, but retinal synaptic effects do not drive postsynaptic spikes."
    elif retina_totals["source_output_spikes"] > 0:
        diagnosis = "DIAGNOSIS B: Source spikes are not actually reaching outgoing edges."
    else:
        diagnosis = "DIAGNOSIS E: Insufficient evidence; no retina source output spike was observed."
    print(diagnosis)

    report_dir = root / "data" / "processed" / "retina_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, totals in (("retina", retina_totals), ("positive_control", positive_totals)):
        for metric, value in totals.items():
            rows.append({"test": label, "metric": metric, "value": int(value)})
    pd.DataFrame(rows).to_csv(report_dir / "retina_v6_3_diagnostic.csv", index=False)
    print(f"diagnostic report: {report_dir / 'retina_v6_3_diagnostic.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
