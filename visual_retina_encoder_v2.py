"""Inspect annotated spatial assignments only; no WORLD transform or live input.

Run in MaleCNS: python visual_retina_encoder_v2.py
Dependencies: numpy, pandas, pyarrow (same as V1).
Scalar pairs and positional tuples are exploratory candidates, not proof of hex
geometry. Only explicit named x/y or q/r pairs can pass the structural NPZ gate.
Even such an NPZ is annotation-space only: WORLD calibration is always unknown.
"""
from pathlib import Path
from collections import Counter
import argparse
import ast
import json
import math
import numbers
import sys

import numpy as np
import pandas as pd


HEX = ("assignedOlHex1", "assignedOlHex2")
V1_FILES = ("visual_retina_population.csv", "visual_retina_type_summary.csv",
            "visual_retina_laterality_summary.csv", "visual_retina_downstream_summary.csv",
            "visual_retina_downstream_top_targets.csv")


def block(title):
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def missing(v):
    if v is None or v is pd.NA:
        return True
    if isinstance(v, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(v))
    except (ValueError, TypeError):
        return False


def finite_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_)) and math.isfinite(float(v))


def parse(v, depth=0):
    """Bounded literal parsing; never eval. Return format, numeric value, status."""
    if missing(v):
        return "null", None, "missing"
    if isinstance(v, str):
        if not v.strip() or v.strip().lower() in ("null", "none", "nan", "unknown", "<na>"):
            return "string_empty_or_missing", None, "missing"
        if depth > 1 or len(v) > 4096:
            return "string", None, "unparsed"
        try:
            value = json.loads(v)
        except (ValueError, TypeError):
            try:
                value = ast.literal_eval(v)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                return "string", None, "unparsed"
        if isinstance(value, str):
            return "string", None, "unparsed"
        kind, value, status = parse(value, depth + 1)
        return "string:" + kind, value, status
    if isinstance(v, dict):
        for keys in (("q", "r"), ("x", "y")):
            if set(v) == set(keys) and all(finite_number(v[k]) for k in keys):
                return "named_" + "_".join(keys), tuple(v[k] for k in keys), "parsed"
        return "dict", None, "unparsed"
    if isinstance(v, (tuple, list, np.ndarray)):
        if len(v) == 2 and all(finite_number(x) for x in v):
            return "positional_pair", tuple(v), "parsed"
        return "sequence", None, "unparsed"
    if finite_number(v):
        return "integer" if float(v).is_integer() else "numeric", v, "parsed"
    return type(v).__name__, None, "unparsed"


def plain(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    return None if missing(v) else v


def canonical(v):
    return json.dumps(plain(v), ensure_ascii=True, sort_keys=True, allow_nan=False, default=str)


def numeric_key(v):
    vals = v if isinstance(v, tuple) else (v,)
    return canonical([int(x) if float(x).is_integer() else float(x) for x in vals])


def require_ids(frame, col):
    if col not in frame:
        raise ValueError(f"Missing required field: {col}")
    raw = frame[col].astype(str)
    if not raw.str.fullmatch(r"\d+").all():
        raise ValueError(f"Invalid/missing integer identifier in {col}")
    frame[col] = raw.astype("int64")
    if frame[col].duplicated().any():
        raise ValueError(f"Duplicate {col}: mapping would be ambiguous")


def main(root=None, output=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    src_reports = root / "data/processed/retina_reports"
    processed = Path(output) if output else root / "data/processed"
    dest = processed / "retina_reports"
    annotation_path = root / "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    simulation_path = root / "data/processed/simulation-neurons.parquet"
    block("INPUT FILES")
    print(annotation_path); print(simulation_path)
    prior = {}
    for name in V1_FILES:
        p = src_reports / name
        print(p)
        prior[name] = pd.read_csv(p, dtype=str, keep_default_na=False)
        print("  rows:", len(prior[name]))
    a = pd.read_feather(annotation_path)
    n = pd.read_parquet(simulation_path, columns=["bodyId", "neuron_idx"])
    require_ids(a, "bodyId"); require_ids(n, "bodyId"); require_ids(n, "neuron_idx")
    v1 = prior[V1_FILES[0]].copy()
    require_ids(v1, "bodyId")
    if "in_simulation" not in v1:
        raise ValueError("V1 population is missing in_simulation")
    if not v1.in_simulation.str.lower().isin(["true", "false"]).all():
        raise ValueError("Invalid V1 in_simulation values")
    v1 = v1[v1.in_simulation.str.lower().eq("true")].copy()
    require_ids(v1, "neuron_idx")
    checks = v1[["bodyId", "neuron_idx"]].merge(n, on="bodyId", suffixes=("_v1", "_current"), how="left", validate="one_to_one")
    if checks.neuron_idx_current.isna().any() or not checks.neuron_idx_v1.eq(checks.neuron_idx_current).all():
        raise ValueError("V1/current simulation mapping disagreement; rerun V1 first")
    supercol = next((c for c in a if c.casefold() == "superclass"), None)
    if not supercol:
        raise ValueError("Cannot revalidate ol_sensory without annotation superclass")
    expected = set(a.loc[a[supercol].astype(str).str.casefold().eq("ol_sensory"), "bodyId"]) & set(n.bodyId)
    if expected != set(v1.bodyId):
        raise ValueError("V1 population differs from current annotation/simulation ol_sensory intersection")
    cols = [c for c in ("bodyId", "type", "instance", "somaSide", *HEX) if c in a]
    population = v1[["bodyId", "neuron_idx"]].merge(a[cols], on="bodyId", how="left", validate="one_to_one")
    population["laterality"] = v1.set_index("bodyId").reindex(population.bodyId).get("side_consensus", pd.Series("UNKNOWN", index=population.bodyId)).to_numpy()
    population["retina_identity"] = v1.set_index("bodyId").reindex(population.bodyId).get("retina_identity", pd.Series("unknown", index=population.bodyId)).to_numpy()
    for f in ("type", "instance", "somaSide"):
        if f not in population:
            population[f] = "unknown"
        population[f] = population[f].map(lambda x: "unknown" if missing(x) else str(x))
    population["laterality"] = population.laterality.where(population.laterality.isin(["LEFT", "RIGHT"]), "UNKNOWN")
    block("TOTAL SIMULATION OL_SENSORY")
    print(len(population), "(4114 is prior reference, not a hard-coded population)")
    parsed = {}
    for f in HEX:
        block(f + " COMPLETENESS / FORMATS / EXAMPLES")
        exists = f in a
        if f not in population:
            population[f] = None
        series = population[f]
        parsed[f] = [parse(v) for v in series]
        nonnull = [v for v in series if not missing(v)]
        filled = sum(status != "missing" for _, _, status in parsed[f])
        print("Column exists:", exists, "annotation dtype:", str(a[f].dtype) if exists else "unknown")
        print("Null count:", len(series) - len(nonnull), "filled:", filled, "fill rate:", f"{filled / max(len(series), 1):.2%}")
        print("Unique non-null raw values:", len({canonical(v) for v in nonnull}))
        print("Python value types:", dict(Counter(type(v).__name__ for v in nonnull)))
        print("Parsed formats:", dict(Counter(k for k, _, _ in parsed[f])))
        print("Examples:", list(dict.fromkeys(canonical(v) for v in nonnull))[:10] or "unknown (no non-null values)")
        population[f + "_format"] = [k for k, _, _ in parsed[f]]
        population[f + "_parse_status"] = [st for _, _, st in parsed[f]]
        population[f + "_parsed"] = [canonical(v) if st == "parsed" else "unknown" for _, v, st in parsed[f]]
        population[f] = [canonical(v) if not missing(v) else "unknown" for v in series]
        print("Whole annotation non-null count:", int(a[f].notna().sum()) if exists else 0)
    # Keep each field independent; scalar pairs are explicitly marked exploratory.
    candidates = []
    trusted = []
    for i, row in population.iterrows():
        parsed_fields = [(f, *parsed[f][i]) for f in HEX]
        named = []
        for f, fmt, value, status in parsed_fields:
            if status != "parsed":
                continue
            scheme = f + ":" + fmt.removeprefix("string:")
            candidates.append((scheme, numeric_key(value), i))
            if fmt.removeprefix("string:").startswith("named_"):
                named.append((fmt.removeprefix("string:"), value))
        v, w = parsed[HEX[0]][i], parsed[HEX[1]][i]
        if v[2] == w[2] == "parsed" and finite_number(v[1]) and finite_number(w[1]):
            candidates.append(("EXPLORATORY_two_scalar_fields", numeric_key((v[1], w[1])), i))
        # Both populated fields must agree and explicitly name coordinate axes.
        populated = sum(st != "missing" for _, _, _, st in parsed_fields)
        if named and len(named) == populated and all(x == named[0] for x in named):
            trusted.append((i, named[0][0], named[0][1]))
    block("COORDINATE INTERPRETATION / CONFIDENCE")
    print("Field names alone do not establish whether these are axes or separate assignments.")
    print("Scalar pairing / positional tuples: exploratory only; geometry and units unknown.")
    print("Unknown coordinates are NOT grouped into a fake shared coordinate.")
    groups = {}
    for scheme, coord, i in candidates:
        groups.setdefault((scheme, coord), set()).add(i)
    summary, type_rows, lateral = [], [], []
    for (scheme, coord), indices in groups.items():
        frame = population.loc[sorted(indices)]
        types = frame.retina_identity.value_counts()
        r16, r7, r8 = (int(types.get(t, 0)) for t in ("R1-R6_group_unresolved", "R7", "R8"))
        other = len(frame) - r16 - r7 - r8
        left, right = int(frame.laterality.eq("LEFT").sum()), int(frame.laterality.eq("RIGHT").sum())
        summary.append(dict(scheme=scheme, coordinate=coord, total_neurons=len(frame), R1_R6=r16, R7=r7, R8=r8, unknown_or_other=other, LEFT=left, RIGHT=right, UNKNOWN=int(frame.laterality.eq("UNKNOWN").sum()), all_three_present=bool(r16 and r7 and r8), both_sides=bool(left and right), neuron_indices=canonical(frame.neuron_idx.tolist()), body_ids=canonical(frame.bodyId.tolist())))
        for label, count in types.items():
            type_rows.append(dict(scheme=scheme, coordinate=coord, retina_identity=label, neuron_count=int(count)))
        # A shared numeric label across sides is not evidence of shared physical space.
    scols = ["scheme", "coordinate", "total_neurons", "R1_R6", "R7", "R8", "unknown_or_other", "LEFT", "RIGHT", "UNKNOWN", "all_three_present", "both_sides", "neuron_indices", "body_ids"]
    summary = pd.DataFrame(summary, columns=scols)
    block("UNIQUE COORDINATES / NEURONS PER COORDINATE / TYPE GROUPING")
    if summary.empty:
        print("Parseable coordinate candidates: 0. Spatial group sizes / R combinations: unknown, not zero biological occupancy.")
    for scheme, frame in summary.groupby("scheme"):
        print(scheme, "unique:", len(frame), "singleton:", int(frame.total_neurons.eq(1).sum()), "multiple:", int(frame.total_neurons.gt(1).sum()), "R1-R6+R7+R8:", int(frame.all_three_present.sum()))
        print("Neurons-per-coordinate distribution:", frame.total_neurons.value_counts().sort_index().to_dict())
        coords = [json.loads(x) for x in frame.coordinate]
        numeric = np.asarray(coords, dtype=float)
        lo, hi = numeric.min(axis=0), numeric.max(axis=0)
        print("Numeric min/max/range:", lo.tolist(), hi.tolist(), (hi-lo).tolist())
        if np.equal(numeric, np.floor(numeric)).all():
            denominator = math.prod(int(b-a+1) for a, b in zip(lo, hi))
            print("Integer bounding-box occupancy:", len(frame) / denominator, "(not hex-lattice or retina coverage)")
        else:
            print("Occupancy density: unknown (no discrete lattice spacing established)")
    block("LEFT/RIGHT COORDINATE DISTRIBUTION")
    for side in ("LEFT", "RIGHT", "UNKNOWN"):
        lateral.append(dict(scheme="population", laterality=side, neuron_count=int(population.laterality.eq(side).sum()), unique_coordinates="unknown", overlap_coordinates="unknown"))
    for scheme, frame in summary.groupby("scheme"):
        overlap = int(frame.both_sides.sum())
        for side in ("LEFT", "RIGHT", "UNKNOWN"):
            lateral.append(dict(scheme=scheme, laterality=side, neuron_count=int(frame[side].sum()), unique_coordinates=int(frame[side].gt(0).sum()), overlap_coordinates=overlap))
    lateral = pd.DataFrame(lateral)
    print(lateral.to_string(index=False))
    print("Cross-side equal values are label overlap only, not physical or WORLD co-location.")
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for name, frame in (("retina_spatial_population.csv", population), ("retina_spatial_coordinate_summary.csv", summary), ("retina_spatial_type_by_coordinate.csv", pd.DataFrame(type_rows, columns=["scheme", "coordinate", "retina_identity", "neuron_count"])), ("retina_spatial_laterality_summary.csv", lateral)):
        path = dest / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(path)
    block("MAPPING CONFIDENCE / LIMITATIONS")
    # Conservative: no partial map masquerading as a complete V3 input map.
    reliable = bool(len(population)) and len(trusted) == len(population) and len({x[1] for x in trusted}) == 1 and population.laterality.ne("UNKNOWN").all()
    path = processed / "retina_spatial_map.npz"
    if reliable:
        rows = population.loc[[x[0] for x in trusted]]
        coords = np.asarray([x[2] for x in trusted])
        np.savez_compressed(path, coordinates=coords, neuron_indices=rows.neuron_idx.to_numpy(dtype=np.int64), body_ids=rows.bodyId.to_numpy(dtype=np.int64), cell_type_code=rows.retina_identity.map({"R1-R6_group_unresolved": 1, "R7": 7, "R8": 8}).fillna(0).to_numpy(dtype=np.int8), laterality_code=rows.laterality.map({"LEFT": -1, "RIGHT": 1}).to_numpy(dtype=np.int8), coordinate_axes=np.asarray(trusted[0][1].split("_")[1:]), coordinate_space=np.asarray("annotation_only_WORLD_transform_unknown"), cell_type_legend=np.asarray(["0=unknown_or_other", "1=R1-R6_unresolved", "7=R7", "8=R8"]), laterality_legend=np.asarray(["-1=LEFT", "1=RIGHT"]))
        files.append(path)
        print("Structurally explicit named-axis map saved. Optic-lobe geometry/units and WORLD calibration still unvalidated.")
    else:
        print("NPZ NOT generated: no complete, unambiguous named-axis coordinate mapping with known laterality.")
        print("Explicit usable named pairs:", len(trusted), "/", len(population))
        if path.exists():
            print("WARNING: pre-existing NPZ was NOT updated or validated; do not treat it as this run's output:", path)
    print("WORLD pixel -> real sensory group: UNKNOWN. Screen/eye orientation, projection and calibration are absent.")
    print("No missing coordinates fabricated; no spatial function inferred from cell type or laterality.")
    block("GENERATED FILES")
    for p in files:
        print(p)
    return population, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Default: script directory")
    parser.add_argument("--output", type=Path, help="Optional processed output directory for validation")
    args = parser.parse_args()
    try:
        main(args.root, args.output)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
