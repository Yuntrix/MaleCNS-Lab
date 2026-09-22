"""MaleCNS direct-target candidate hex discovery (unweighted, offline).

Run: python visual_retina_encoder_v3.py
Dependencies: numpy, pandas, pyarrow.
Each DISTINCT downstream neuron contributes one vote per annotation field.
No weights, multi-hop tracing, WORLD mapping, NPZ, simulation or injection.
All thresholds are descriptive sensitivity analyses, not acceptance criteria.
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
import pyarrow.parquet as pq


FIELDS = ("assignedOlHex1", "assignedOlHex2")
COLUMNAR_TYPES = ("L1", "L2", "L3", "T1")
DOMINANCES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MIN_TARGETS = (1, 2, 3, 5, 10)
CLASSES = ("R1-R6", "R7", "R8", "unknown")


def block(title):
    print("\n" + "=" * 96 + "\n" + title + "\n" + "=" * 96, flush=True)


def missing(value):
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return False
    return bool(pd.isna(value))


def label(value):
    return "unknown" if missing(value) or not str(value).strip() else str(value).strip()


def encode_hex(value, depth=0):
    """Preserve assignment values as labels; never infer a coordinate geometry.

    Numeric strings/scalars are normalized. Numeric pairs retain both elements
    as ONE label; they are not split into two votes. Unsupported structures are
    excluded and counted. Arbitrary string labels remain opaque string labels.
    """
    if missing(value):
        return None, "missing"
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return encode_hex(value.item(), depth)
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in ("", "unknown", "none", "nan", "null", "<na>"):
            return None, "missing"
        if len(value) > 4096 or depth > 1:
            return None, "unsupported"
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError, RecursionError):
                if value[:1] in "[{(":
                    return None, "unsupported"
                return "string:" + value, "opaque_string"
        if isinstance(parsed, str):
            return "string:" + parsed, "opaque_string"
        result, status = encode_hex(parsed, depth + 1)
        return result, "string_" + status if result is not None else status
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_)):
        if not math.isfinite(float(value)):
            return None, "unsupported"
        val = int(value) if float(value).is_integer() else float(value)
        return "number:" + str(val), "numeric"
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) == 2:
            vals = [encode_hex(v, depth + 1)[0] for v in value]
            if all(v is not None and v.startswith("number:") for v in vals):
                return "pair:" + json.dumps(vals, separators=(",", ":")), "numeric_pair"
        return None, "unsupported"
    return None, "unsupported"


def get_column(frame, wanted):
    found = [c for c in frame if c.casefold() == wanted.casefold()]
    if len(found) > 1:
        raise ValueError(f"Ambiguous field {wanted}: {found}")
    return found[0] if found else None


def series(frame, wanted):
    col = get_column(frame, wanted)
    return frame[col].map(label) if col else pd.Series("unknown", index=frame.index)


def check_identifier(frame, col):
    if col not in frame or frame[col].isna().any():
        raise ValueError(f"Missing identifier: {col}")
    if not frame[col].astype(str).str.fullmatch(r"\d+").all():
        raise ValueError(f"Non-integer identifier: {col}")
    frame[col] = frame[col].astype("int64")
    if frame[col].duplicated().any():
        raise ValueError(f"Duplicate identifier: {col}")


def cell_class(row):
    evidence = []
    for field in ("type", "flywireType", "hemibrainType"):
        v = row[field]
        if v in ("R1-R6", "R1-6"):
            evidence.append(("R1-R6", field + "=" + v))
        elif v in ("R7", "R8"):
            evidence.append((v, field + "=" + v))
    classes = {x[0] for x in evidence}
    return (next(iter(classes)) if len(classes) == 1 else "unknown", "; ".join(x[1] for x in evidence) or "unknown")


def load_population(root):
    ap = root / "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    npth = root / "data/processed/simulation-neurons.parquet"
    print(ap); print(npth)
    a, n = pd.read_feather(ap), pd.read_parquet(npth, columns=["bodyId", "neuron_idx"])
    check_identifier(a, "bodyId")
    check_identifier(n, "bodyId"); check_identifier(n, "neuron_idx")
    if not np.array_equal(np.sort(n.neuron_idx.to_numpy()), np.arange(len(n))):
        raise ValueError("Simulation indices are not a unique contiguous zero-based set")
    arrays = {}
    for side in ("all", "left", "right"):
        path = root / f"data/processed/retina_{side}_indices.npy"
        print(path)
        v = np.load(path, allow_pickle=False)
        if v.ndim != 1 or not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"Invalid index array: {path}")
        if len(np.unique(v)) != len(v) or ((v < 0) | (v >= len(n))).any():
            raise ValueError(f"Duplicate/out-of-range index: {path}")
        arrays[side] = set(map(int, v))
    if arrays["left"] & arrays["right"] or (arrays["left"] | arrays["right"]) - arrays["all"]:
        raise ValueError("LEFT/RIGHT arrays overlap or contain non-retina indices")
    meta = pd.DataFrame({"bodyId": a.bodyId})
    for field in ("type", "instance", "somaSide", "rootSide", "superclass", "flywireType", "hemibrainType"):
        meta[field] = series(a, field)
    for field in FIELDS:
        col = get_column(a, field)
        vals = a[col] if col else pd.Series(None, index=a.index, dtype=object)
        encoded = [encode_hex(v) for v in vals]
        meta[field] = [v[0] for v in encoded]
        meta[field + "_parse_status"] = [v[1] for v in encoded]
        print(field, "dtype:", str(vals.dtype), "parse formats:", dict(Counter(x[1] for x in encoded)))
    meta = n.merge(meta, on="bodyId", how="left", validate="one_to_one").set_index("neuron_idx").sort_index()
    expected = set(meta.index[meta.superclass.eq("ol_sensory")])
    if expected != arrays["all"]:
        raise ValueError("retina_all_indices does not match current annotation-defined simulation ol_sensory")
    pop = meta.loc[sorted(arrays["all"])].copy().reset_index()
    pop["laterality"] = ["LEFT" if i in arrays["left"] else "RIGHT" if i in arrays["right"] else "UNKNOWN" for i in pop.neuron_idx]
    # Reject stale laterality arrays when explicit annotation disagrees.
    for row in pop.itertuples():
        known = set()
        for v in (row.somaSide, row.rootSide):
            if v in ("L", "LEFT"): known.add("LEFT")
            if v in ("R", "RIGHT"): known.add("RIGHT")
        if str(row.instance).endswith("_L"): known.add("LEFT")
        if str(row.instance).endswith("_R"): known.add("RIGHT")
        if len(known) > 1 or (row.laterality != "UNKNOWN" and known and row.laterality not in known):
            raise ValueError(f"Laterality conflict for bodyId={row.bodyId}; rerun V1")
    identities = [cell_class(row) for row in pop.to_dict("records")]
    pop["cell_class"] = [x[0] for x in identities]
    pop["class_evidence"] = [x[1] for x in identities]
    return pop, meta


def direct_targets(path, sources, n):
    print(path, flush=True)
    file = pq.ParquetFile(path)
    if not {"pre_idx", "post_idx"}.issubset(file.schema_arrow.names):
        raise ValueError("Unknown edge schema: pre_idx/post_idx required; no direction guessed")
    print("Reading ONLY pre_idx/post_idx; edge weights are not used.", flush=True)
    member = np.zeros(n, dtype=bool)
    member[list(sources)] = True
    chunks, scanned, selected = [], 0, 0
    for batch in file.iter_batches(batch_size=500000, columns=["pre_idx", "post_idx"]):
        frame = batch.to_pandas()
        for col in ("pre_idx", "post_idx"):
            if frame[col].isna().any() or not pd.api.types.is_integer_dtype(frame[col]):
                raise ValueError(f"Non-integer edge endpoint: {col}")
            if ((frame[col] < 0) | (frame[col] >= n)).any():
                raise ValueError(f"Out-of-range edge endpoint: {col}")
        chosen = frame[member[frame.pre_idx.to_numpy()]]
        scanned += len(frame); selected += len(chosen)
        if len(chosen): chunks.append(chosen)
    pairs = pd.concat(chunks, ignore_index=True).drop_duplicates() if chunks else pd.DataFrame({"pre_idx": pd.Series(dtype="int64"), "post_idx": pd.Series(dtype="int64")})
    print("Scanned edge rows:", scanned, "retina outgoing rows:", selected, "unique source-target pairs:", len(pairs), "duplicate votes removed:", selected - len(pairs))
    return pairs


def consensus_for(pop, targets, field, subset):
    eligible = targets[targets[field].notna()].copy()
    if subset == "columnar_L1_L2_L3_T1":
        eligible = eligible[eligible.type.isin(COLUMNAR_TYPES)]
    votes = eligible.groupby(["pre_idx", field]).size()
    by_source = {int(idx): part.droplevel(0) for idx, part in votes.groupby(level=0)}
    rows = []
    for row in pop.itertuples():
        counts = by_source.get(row.neuron_idx, pd.Series(dtype="int64"))
        total = int(counts.sum())
        top_count = int(counts.max()) if total else 0
        ties = sorted(counts.index[counts.eq(top_count)].tolist()) if total else []
        ambiguity = "no_eligible_targets" if not total else "top_vote_tie" if len(ties) > 1 else "multiple_candidates" if len(counts) > 1 else "single_candidate"
        rows.append(dict(neuron_idx=row.neuron_idx, bodyId=row.bodyId, cell_class=row.cell_class, laterality=row.laterality, field=field, subset=subset, top_candidate_hex=ties[0] if len(ties) == 1 else "unknown", top_candidates_json=json.dumps(ties), candidate_vote_count=top_count, total_eligible_hex_targets=total, dominance_ratio=top_count / total if total else np.nan, unique_candidate_hexes=len(counts), top_tie_count=len(ties), ambiguity=ambiguity, strict_majority=bool(total and top_count * 2 > total), vote_distribution_json=json.dumps({str(k): int(v) for k, v in counts.sort_index().items()}, sort_keys=True)))
    return pd.DataFrame(rows), eligible


def main(root=None, output=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    output = Path(output) if output else root / "data/processed/retina_reports"
    block("INPUT")
    pop, meta = load_population(root)
    block("TOTAL RETINA POPULATION")
    print("Current data:", len(pop), "(4114 is a prior result, not a hard-coded count)")
    print(pop.groupby(["cell_class", "laterality"]).size().to_string())
    print("R1-R6 remains unresolved; unrecognized/ambiguous labels remain unknown.")
    pairs = direct_targets(root / "data/processed/simulation-edges.parquet", set(pop.neuron_idx), len(meta))
    target_meta = meta[["bodyId", "type", "superclass", *FIELDS, *(f + "_parse_status" for f in FIELDS)]]
    targets = pairs.merge(target_meta, left_on="post_idx", right_index=True, how="left", validate="many_to_one")
    all_counts = pairs.groupby("pre_idx").size()
    either = targets[list(FIELDS)].notna().any(axis=1)
    either_counts = targets[either].groupby("pre_idx").size()
    pop["unique_direct_targets"] = pop.neuron_idx.map(all_counts).fillna(0).astype(int)
    pop["targets_with_either_hex_field"] = pop.neuron_idx.map(either_counts).fillna(0).astype(int)
    block("HEX-ANNOTATED DIRECT TARGET COVERAGE")
    print("At least one eligible target (either field):", int(pop.targets_with_either_hex_field.gt(0).sum()))
    print("No eligible target (either field):", int(pop.targets_with_either_hex_field.eq(0).sum()))
    print("Target count distribution:", pop.targets_with_either_hex_field.value_counts().sort_index().to_dict())
    consensus_parts, type_rows = [], []
    for field in FIELDS:
        for subset in ("all_direct", "columnar_L1_L2_L3_T1"):
            result, eligible = consensus_for(pop, targets, field, subset)
            consensus_parts.append(result)
            for target_type, frame in eligible.groupby("type", dropna=False):
                type_rows.append(dict(record_kind="direct_target_type", field=field, subset=subset, target_type=label(target_type), sensory_neurons=frame.pre_idx.nunique(), unique_target_neurons=frame.post_idx.nunique(), target_votes=len(frame), candidate_hex="", R1_R6=np.nan, R7=np.nan, R8=np.nan, unknown=np.nan, LEFT=np.nan, RIGHT=np.nan, UNKNOWN=np.nan, all_three_classes=np.nan, overlap_sides=np.nan))
            present = set(eligible.type)
            for target_type in COLUMNAR_TYPES:
                if target_type not in present:
                    type_rows.append(dict(record_kind="direct_target_type", field=field, subset=subset, target_type=target_type, sensory_neurons=0, unique_target_neurons=0, target_votes=0))
    consensus = pd.concat(consensus_parts, ignore_index=True)
    coverage, sweep = [], []
    for (field, subset), frame in consensus.groupby(["field", "subset"], sort=False):
        scopes = [("all", "ALL", frame)] + [("cell_class", c, frame[frame.cell_class.eq(c)]) for c in CLASSES] + [("laterality", s, frame[frame.laterality.eq(s)]) for s in ("LEFT", "RIGHT", "UNKNOWN")]
        for scope, name, part in scopes:
            covered = part.total_eligible_hex_targets.gt(0)
            coverage.append(dict(record_kind="coverage", field=field, subset=subset, scope=scope, group=name, population_count=len(part), covered_count=int(covered.sum()), uncovered_count=int((~covered).sum()), coverage_ratio=float(covered.mean()) if len(part) else np.nan, unique_top_count=int(part.top_tie_count.eq(1).sum()), top_tie_count=int(part.top_tie_count.gt(1).sum()), median_dominance_covered=part.loc[covered, "dominance_ratio"].median(), median_eligible_targets=part.total_eligible_hex_targets.median()))
            for minimum in MIN_TARGETS:
                for threshold in DOMINANCES:
                    hits = part.total_eligible_hex_targets.ge(minimum) & part.dominance_ratio.ge(threshold)
                    sweep.append(dict(field=field, subset=subset, scope=scope, group=name, min_eligible_targets=minimum, min_dominance=threshold, population_count=len(part), covered_count=int(hits.sum()), unique_top_count=int((hits & part.top_tie_count.eq(1)).sum()), tied_top_count=int((hits & part.top_tie_count.gt(1)).sum()), covered_fraction=float(hits.mean()) if len(part) else np.nan))
        for count, freq in frame.total_eligible_hex_targets.value_counts().sort_index().items():
            coverage.append(dict(record_kind="eligible_target_count_distribution", field=field, subset=subset, scope="all", group=str(count), population_count=len(frame), covered_count=int(freq)))
        for ambiguity, freq in frame.ambiguity.value_counts().items():
            coverage.append(dict(record_kind="ambiguity_distribution", field=field, subset=subset, scope="all", group=ambiguity, population_count=len(frame), covered_count=int(freq)))
        # Group ONLY unique top votes. Ties are retained in consensus but never assigned.
        unique = frame[frame.top_tie_count.eq(1)]
        for candidate, group in unique.groupby("top_candidate_hex"):
            cc, ss = group.cell_class.value_counts(), group.laterality.value_counts()
            type_rows.append(dict(record_kind="unique_top_candidate_group_NO_quality_cutoff", field=field, subset=subset, target_type="", candidate_hex=candidate, sensory_neurons=len(group), R1_R6=int(cc.get("R1-R6", 0)), R7=int(cc.get("R7", 0)), R8=int(cc.get("R8", 0)), unknown=int(cc.get("unknown", 0)), LEFT=int(ss.get("LEFT", 0)), RIGHT=int(ss.get("RIGHT", 0)), UNKNOWN=int(ss.get("UNKNOWN", 0)), all_three_classes=bool(all(cc.get(c, 0) for c in ("R1-R6", "R7", "R8"))), overlap_sides=bool(ss.get("LEFT", 0) and ss.get("RIGHT", 0))))
        left = set(unique.loc[unique.laterality.eq("LEFT"), "top_candidate_hex"])
        right = set(unique.loc[unique.laterality.eq("RIGHT"), "top_candidate_hex"])
        coverage.append(dict(record_kind="candidate_label_overlap_NO_quality_cutoff", field=field, subset=subset, scope="unique_top_only", group="LEFT_vs_RIGHT", left_unique_hexes=len(left), right_unique_hexes=len(right), shared_hexes=len(left & right), shared_hexes_json=json.dumps(sorted(left & right))))
    cov = pd.DataFrame(coverage)
    sweep = pd.DataFrame(sweep)
    types = pd.DataFrame(type_rows)
    block("R1-R6 / R7 / R8 COVERAGE")
    show = ["field", "subset", "group", "population_count", "covered_count", "uncovered_count", "coverage_ratio"]
    print(cov[(cov.record_kind == "coverage") & (cov.scope == "cell_class")][show].to_string(index=False))
    block("LEFT / RIGHT COVERAGE")
    print(cov[(cov.record_kind == "coverage") & (cov.scope == "laterality")][show].to_string(index=False))
    print("\nUnique-top label overlap (not physical overlap):")
    print(cov[cov.record_kind.str.startswith("candidate_label_overlap")][["field", "subset", "left_unique_hexes", "right_unique_hexes", "shared_hexes"]].to_string(index=False))
    block("HEX CONSENSUS DISTRIBUTION")
    print(cov[(cov.record_kind == "coverage") & (cov.scope == "all")][["field", "subset", "covered_count", "unique_top_count", "top_tie_count", "median_dominance_covered", "median_eligible_targets"]].to_string(index=False))
    print(consensus.groupby(["field", "subset", "ambiguity"]).size().to_string())
    print("\nPer-neuron vote distributions are in vote_distribution_json; no-candidate dominance is undefined/blank.")
    block("DOMINANCE THRESHOLD SWEEP")
    print(sweep[sweep.scope.eq("all")][["field", "subset", "min_eligible_targets", "min_dominance", "covered_count", "unique_top_count", "tied_top_count"]].to_string(index=False))
    block("COLUMNAR TARGET SUBSET COMPARISON")
    comparisons = []
    for field in FIELDS:
        f = consensus[consensus.field.eq(field)]
        allpart = f[f.subset.eq("all_direct")].set_index("neuron_idx")
        colpart = f[f.subset.eq("columnar_L1_L2_L3_T1")].set_index("neuron_idx")
        paired = allpart.total_eligible_hex_targets.gt(0) & colpart.total_eligible_hex_targets.gt(0)
        delta = colpart.loc[paired, "dominance_ratio"] - allpart.loc[paired, "dominance_ratio"]
        tolerance = 1e-12
        item = dict(record_kind="paired_subset_comparison", field=field, subset="columnar_vs_all", scope="both_covered", group="paired", population_count=len(delta), dominance_increased=int((delta > tolerance).sum()), dominance_decreased=int((delta < -tolerance).sum()), dominance_equal=int((delta.abs() <= tolerance).sum()), mean_dominance_delta=delta.mean(), median_target_count_all=allpart.loc[paired, "total_eligible_hex_targets"].median(), median_target_count_subset=colpart.loc[paired, "total_eligible_hex_targets"].median())
        comparisons.append(item)
        print(item)
    cov = pd.concat([cov, pd.DataFrame(comparisons)], ignore_index=True)
    target_types = types[types.record_kind.eq("direct_target_type")]
    print("\nExplicit requested target types:")
    print(target_types[target_types.target_type.isin(COLUMNAR_TYPES)][["field", "subset", "target_type", "sensory_neurons", "unique_target_neurons", "target_votes"]].to_string(index=False))
    print("\nTop 20 target types per field (all eligible direct targets, not inferred visual classes):")
    for field in FIELDS:
        print(field)
        print(target_types[(target_types.field == field) & (target_types.subset == "all_direct")].sort_values("target_votes", ascending=False).head(20)[["target_type", "sensory_neurons", "unique_target_neurons", "target_votes"]].to_string(index=False))
    grouped = types[types.record_kind.str.startswith("unique_top_candidate")]
    if len(grouped):
        print("\nCandidate class grouping (unique top, no quality cutoff):")
        print(grouped.groupby(["field", "subset"])[["all_three_classes", "overlap_sides"]].sum().to_string())
    block("LIMITATIONS")
    print("Results are connectome-supported candidate hex LABELS, not true retinal coordinates.")
    print("Fields are separate. Their names do not establish axes or biological meanings.")
    print("No acceptance threshold selected. Report coverage/dominance/min-target sweeps together.")
    print("One target can yield dominance=1.0; this is weak evidence, not proof of reliability.")
    print("Subset agreement may rise simply because targets were removed; paired counts are reported.")
    print("Tied maxima are ambiguous and never assigned a single candidate.")
    print("LEFT/RIGHT shared labels do not establish screen orientation or shared physical locations.")
    print("Processed edges may omit connections; absence here is not proof of anatomical absence.")
    print("No weights, synapse-count claims, WORLD mapping, NPZ, OBS or spike injection.")
    if not pop.targets_with_either_hex_field.gt(0).any():
        print("FAILED CANDIDATE RECOVERY: zero coverage; no coordinate can be inferred.")
    else:
        print("Reliability is UNRESOLVED; the hypothesis is not declared successful by this script.")
        print("Low/zero coverage, ties and diffuse votes remain explicit failures/unresolved cases, not fabricated coordinates.")
    output.mkdir(parents=True, exist_ok=True)
    block("GENERATED FILES")
    for filename, frame in (("retina_downstream_hex_population.csv", pop), ("retina_downstream_hex_consensus.csv", consensus), ("retina_downstream_hex_coverage_summary.csv", cov), ("retina_downstream_hex_threshold_sweep.csv", sweep), ("retina_downstream_hex_type_summary.csv", types)):
        path = output / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(path)
    return pop, consensus, cov, sweep, types


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Default: script directory")
    parser.add_argument("--output", type=Path, help="Optional report directory for validation")
    args = parser.parse_args()
    try:
        main(args.root, args.output)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
