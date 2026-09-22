"""MaleCNS same-target HEX PAIR candidate analysis (unweighted, offline).

Run: python visual_retina_encoder_v4.py
Dependencies: numpy, pandas, pyarrow.
Each DISTINCT downstream neuron contributes one vote per annotation field.
No weights, multi-hop tracing, WORLD mapping, simulation or injection.
Optional NPZ is analysis-only, never a WORLD map.
Confidence tiers are engineering criteria, not biological validation.
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
COLUMNAR_TYPES = ("L1", "L2", "L3", "T1", "L5", "C2", "C3", "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20")
DOMINANCES = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
MIN_TARGETS = (1, 2, 3, 5)
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


def numeric_component(encoded):
    # load_population preserves numeric scalar labels as number:VALUE.
    # Positional pairs, opaque labels and missing fields cannot be components.
    if not isinstance(encoded, str) or not encoded.startswith("number:"):
        return None
    v = float(encoded.split(":", 1)[1])
    if not math.isfinite(v):
        return None
    return int(v) if v.is_integer() else v


def target_side(row):
    sides = set()
    evidence = []
    for field in ("somaSide", "rootSide"):
        v = row.get(field, "unknown")
        s = {"L": "LEFT", "R": "RIGHT", "LEFT": "LEFT", "RIGHT": "RIGHT"}.get(v)
        if s:
            sides.add(s); evidence.append(field + "=" + v)
    instance = str(row.get("instance", "unknown"))
    primary = {"L": "LEFT", "R": "RIGHT"}.get(instance[-1:]) if instance.endswith(("_L", "_R")) else None
    if primary:
        sides.add(primary); evidence.append("instance=" + instance)
    # rootSide alone is a root label, not a verified eye label.
    has_primary = primary is not None or row.get("somaSide") in ("L", "R", "LEFT", "RIGHT")
    return (next(iter(sides)) if len(sides) == 1 and has_primary else "UNKNOWN",
            len(sides) > 1, "; ".join(evidence) or "unknown")


def pair_consensus(pop, eligible, subset, mode):
    cols = ["target_side", "hex1", "hex2"] if mode == "side_pair" else ["hex1", "hex2"]
    distributions = {}
    agreement = {}
    for source, group in eligible.groupby("pre_idx"):
        distributions[int(source)] = Counter(tuple(row) for row in group[cols].itertuples(index=False, name=None))
        agreement[int(source)] = Counter(group.side_relation)
    rows = []
    for p in pop.itertuples():
        votes = distributions.get(p.neuron_idx, Counter())
        total, unique = sum(votes.values()), len(votes)
        count = max(votes.values(), default=0)
        tops = sorted([k for k, v in votes.items() if v == count])
        single = len(tops) == 1
        top = tops[0] if single else None
        dom = count / total if total else np.nan
        strong = bool(single and total >= 2 and dom >= 0.8)
        very = bool(single and total >= 2 and count == total)
        tside = top[0] if top and mode == "side_pair" else "UNKNOWN"
        hx = top[-2:] if top else (np.nan, np.nan)
        resolved = bool(single and (mode == "pair" or tside != "UNKNOWN"))
        relation = "unknown" if not top or tside == "UNKNOWN" or p.laterality == "UNKNOWN" else "same" if tside == p.laterality else "cross"
        ag = agreement.get(p.neuron_idx, Counter())
        rows.append(dict(neuron_idx=p.neuron_idx, body_id=p.bodyId, cell_class=p.cell_class, sensory_laterality=p.laterality,
            subset=subset, voting_mode=mode, eligible_target_count=total, total_votes=total,
            unique_coordinate_count=unique, top_hex1=hx[0], top_hex2=hx[1], top_target_side=tside,
            top_vote_count=count, dominance=dom, top_tie=len(tops)>1, top_tie_count=len(tops),
            top_candidates_json=json.dumps(tops), ambiguity="no_targets" if not total else "top_tie" if not single else "unknown_target_side" if not resolved else "multiple_candidates" if unique>1 else "single_candidate",
            candidate_resolved=resolved, strong=strong, very_strong=very,
            strong_resolved=bool(strong and resolved), confidence_code=2 if very else 1 if strong else 0,
            same_side_votes=ag.get("same",0), cross_side_votes=ag.get("cross",0), unknown_side_votes=ag.get("unknown",0),
            top_side_relation=relation,
            vote_distribution_json=json.dumps([{"coordinate": k, "votes": v} for k,v in sorted(votes.items())])))
    return pd.DataFrame(rows)


def main(root=None, output=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    processed = Path(output) if output else root / "data/processed"
    reports = processed / "retina_reports"
    block("INPUT")
    pop, meta = load_population(root)
    block("TOTAL RETINA")
    print(len(pop), "annotation-validated simulation ol_sensory neurons")
    print("User-supplied premise: assignedOlHex1/2 jointly define an optic-lobe column coordinate.")
    print("This script tests downstream support, not the premise or WORLD orientation.")
    pairs = direct_targets(root / "data/processed/simulation-edges.parquet", set(pop.neuron_idx), len(meta))
    target_meta = meta.copy()
    for f, out in zip(FIELDS, ("hex1", "hex2")):
        target_meta[out] = target_meta[f].map(numeric_component)
    sides = [target_side(row) for row in target_meta.to_dict("records")]
    target_meta["target_side"] = [s[0] for s in sides]
    target_meta["side_disagreement"] = [s[1] for s in sides]
    target_meta["side_evidence"] = [s[2] for s in sides]
    targets = pairs.merge(target_meta, left_on="post_idx", right_index=True, how="left", validate="many_to_one")
    eligible = targets[targets.hex1.notna() & targets.hex2.notna()].copy()
    for f in ("hex1", "hex2"):
        eligible[f] = eligible[f].map(lambda x: int(x) if float(x).is_integer() else float(x))
    source_sides = pop.set_index("neuron_idx").laterality
    eligible["sensory_laterality"] = eligible.pre_idx.map(source_sides)
    eligible["side_relation"] = np.where(eligible.target_side.eq("UNKNOWN") | eligible.sensory_laterality.eq("UNKNOWN"), "unknown", np.where(eligible.target_side.eq(eligible.sensory_laterality), "same", "cross"))
    counts = eligible.groupby("pre_idx").size()
    pop["unique_direct_targets"] = pop.neuron_idx.map(pairs.groupby("pre_idx").size()).fillna(0).astype(int)
    pop["eligible_pair_targets"] = pop.neuron_idx.map(counts).fillna(0).astype(int)
    pop["ineligible_or_missing_pair_targets"] = pop.unique_direct_targets-pop.eligible_pair_targets
    block("PAIR TARGET COVERAGE")
    print("Covered:", int(pop.eligible_pair_targets.gt(0).sum()), "No pair targets:", int(pop.eligible_pair_targets.eq(0).sum()))
    print("Eligible unique source-target votes:", len(eligible))
    print("Missing/non-numeric pair votes excluded:", len(targets)-len(eligible))
    print("Cross-side and UNKNOWN-side votes are INCLUDED.")
    block("COLUMNAR SUBSET VALIDATION")
    print("Explicit HARD-CODED user-requested type whitelist:", ", ".join(COLUMNAR_TYPES))
    print("Presence of numeric pair annotations is verified; biological columnarity is not inferred here.")
    for t in COLUMNAR_TYPES:
        annotated = target_meta[target_meta.type.eq(t)]
        valid = annotated.hex1.notna() & annotated.hex2.notna()
        actual = eligible[eligible.type.eq(t)]
        print(t, "simulation neurons:", len(annotated), "with pair:", int(valid.sum()), "retina direct unique targets:", actual.post_idx.nunique(), "votes:", len(actual))
    subset_frames = {"all_direct":eligible, "columnar_subset":eligible[eligible.type.isin(COLUMNAR_TYPES)]}
    consensus = pd.concat([pair_consensus(pop, frame, subset, mode) for subset, frame in subset_frames.items() for mode in ("pair", "side_pair")], ignore_index=True)
    block("PAIR CONSENSUS")
    print(consensus.groupby(["subset", "voting_mode", "ambiguity"]).size().to_string())
    block("STRONG / VERY STRONG")
    print("Engineering tiers, NOT biological truth. VERY_STRONG is a subset of STRONG.")
    print("STRONG: unique maximum, dominance >=0.8, >=2 eligible targets.")
    print("VERY_STRONG: unique maximum, dominance ==1, >=2 eligible targets.")
    print(consensus.groupby(["subset", "voting_mode"])[["strong", "very_strong", "strong_resolved"]].sum().to_string())
    sweep_rows = []
    for (subset, mode), frame in consensus.groupby(["subset", "voting_mode"]):
        scopes = [("all", "ALL", frame)] + [("cell_class", t, frame[frame.cell_class.eq(t)]) for t in CLASSES] + [("laterality", side, frame[frame.sensory_laterality.eq(side)]) for side in ("LEFT", "RIGHT", "UNKNOWN")]
        for scope, name, group in scopes:
            for minimum in MIN_TARGETS:
                for dominance in DOMINANCES:
                    hit = group.eligible_target_count.ge(minimum) & group.dominance.ge(dominance)
                    sweep_rows.append(dict(subset=subset, voting_mode=mode, scope=scope, group=name, population_count=len(group), min_targets=minimum, min_dominance=dominance, threshold_count=int(hit.sum()), unique_top_count=int((hit & group.top_tie_count.eq(1)).sum()), resolved_count=int((hit & group.candidate_resolved).sum()), top_tie_count=int((hit & group.top_tie).sum())))
    sweep = pd.DataFrame(sweep_rows)
    for title, col in (("R1-R6 / R7 / R8", "cell_class"), ("LEFT / RIGHT", "sensory_laterality")):
        block(title)
        groups = consensus[consensus.voting_mode.eq("side_pair")].groupby(["subset", col])
        print(groups.agg(population=("neuron_idx","size"),covered=("eligible_target_count",lambda s:int(s.gt(0).sum())),strong=("strong","sum"),very_strong=("very_strong","sum"),strong_resolved=("strong_resolved","sum")).to_string())
    block("SIDE AGREEMENT")
    for subset, frame in subset_frames.items():
        print(subset, "votes (deduplicated source-target pairs):")
        print(frame.groupby(["sensory_laterality", "target_side"]).size().to_string())
        print("Cross-side votes:", int(frame.side_relation.eq("cross").sum()), "unknown-side:", int(frame.side_relation.eq("unknown").sum()), "target-side disagreement votes:", int(frame.side_disagreement.sum()))
        print("Distinct sensory with cross-side vote:",frame.loc[frame.side_relation.eq("cross"),"pre_idx"].nunique())
    # Geometry and receptor grouping use inferred TARGET side, not sensory side.
    # Keep paired labels and side-aware labels distinct; no arbitrary tie breaks.
    coord_rows, occupancy_rows = [], []
    for subset, frame in consensus[consensus.voting_mode.eq("side_pair")].groupby("subset"):
        for tier, selected in (("UNIQUE_TOP",frame[frame.candidate_resolved]),("STRONG",frame[frame.strong_resolved]),("VERY_STRONG",frame[frame.very_strong & frame.candidate_resolved])):
            for (side,h1,h2), group in selected.groupby(["top_target_side","top_hex1","top_hex2"]):
                classes=group.cell_class.value_counts()
                coord_rows.append(dict(subset=subset,tier=tier,laterality=side,hex1=h1,hex2=h2,sensory_count=len(group),R1_R6=int(classes.get("R1-R6",0)),R7=int(classes.get("R7",0)),R8=int(classes.get("R8",0)),unknown=int(classes.get("unknown",0)),all_three_classes=bool(all(classes.get(t,0) for t in ("R1-R6","R7","R8"))),sensory_LEFT=int(group.sensory_laterality.eq("LEFT").sum()),sensory_RIGHT=int(group.sensory_laterality.eq("RIGHT").sum()),sensory_UNKNOWN=int(group.sensory_laterality.eq("UNKNOWN").sum()),cross_side_candidates=int(group.top_side_relation.eq("cross").sum()),neuron_indices_json=json.dumps(group.neuron_idx.tolist()),body_ids_json=json.dumps(group.body_id.tolist())))
    columns=["subset","tier","laterality","hex1","hex2","sensory_count","R1_R6","R7","R8","unknown","all_three_classes","sensory_LEFT","sensory_RIGHT","sensory_UNKNOWN","cross_side_candidates","neuron_indices_json","body_ids_json"]
    coordinates=pd.DataFrame(coord_rows,columns=columns)
    block("COORDINATE RANGE")
    if coordinates.empty: print("No resolved side-aware candidate coordinates; ranges UNKNOWN.")
    for (subset,tier),frame in coordinates.groupby(["subset","tier"]):
        for side in ("LEFT","RIGHT"):
            group=frame[frame.laterality.eq(side)]
            print(subset,tier,side,"unique pairs:",len(group),"hex1 min/max:",(group.hex1.min(),group.hex1.max()),"hex2 min/max:",(group.hex2.min(),group.hex2.max()))
            for number,count in group.sensory_count.value_counts().sort_index().items():
                occupancy_rows.append(dict(subset=subset,tier=tier,laterality=side,sensory_per_coordinate=int(number),coordinate_count=int(count)))
        left=set(frame.loc[frame.laterality.eq("LEFT"),["hex1","hex2"]].itertuples(index=False,name=None))
        right=set(frame.loc[frame.laterality.eq("RIGHT"),["hex1","hex2"]].itertuples(index=False,name=None))
        print("Numeric pair overlap across TARGET sides:",len(left & right),"(distinct side-aware identities)")
    occupancy=pd.DataFrame(occupancy_rows,columns=["subset","tier","laterality","sensory_per_coordinate","coordinate_count"])
    block("COORDINATE OCCUPANCY")
    print(occupancy.to_string(index=False))
    if len(coordinates):
        print("\nCoordinates containing all R1-R6/R7/R8 classes:")
        print(coordinates.groupby(["subset","tier","laterality"]).all_three_classes.sum().to_string())
    block("LIMITATIONS")
    print("Only connectome-supported column candidates; not true retinal coordinates.")
    print("Target side is annotation consensus (soma/instance, root corroboration), not measured eye orientation.")
    print("Cross-side votes retained; inspect side_relation before using any candidate.")
    print("UNKNOWN-side/tied candidates remain unresolved; numeric pair mode alone cannot resolve side.")
    print("Subset uses the explicitly printed whitelist. Higher agreement can reflect fewer targets.")
    print("NPZ is analysis-only, includes STRONG resolved side_pair rows for BOTH subsets; subset_code disambiguates duplicate neuron IDs.")
    print("No final spatial map, WORLD pixels, screen orientation, OBS, AI APIs, spikes, weights, motor or LIF changes.")
    print("CSV coordinate summary provides numeric 2D scatter-ready columns; axes are not interpreted as screen directions.")
    reports.mkdir(parents=True,exist_ok=True)
    generated=[]
    for name,frame in (("retina_hex_pair_population.csv",pop),("retina_hex_pair_consensus.csv",consensus),("retina_hex_pair_threshold_sweep.csv",sweep),("retina_hex_pair_coordinate_summary.csv",coordinates),("retina_hex_pair_occupancy.csv",occupancy)):
        path=reports/name;frame.to_csv(path,index=False,encoding="utf-8-sig");generated.append(path)
    chosen=consensus[consensus.voting_mode.eq("side_pair") & consensus.strong_resolved].copy()
    npz_path=processed/"retina_hex_pair_candidates.npz"
    # Empty arrays are a valid explicit 'no qualifying candidates' analysis result.
    np.savez_compressed(npz_path,
        neuron_idx=chosen.neuron_idx.to_numpy(dtype=np.int64),body_id=chosen.body_id.to_numpy(dtype=np.int64),
        laterality_code=chosen.top_target_side.map({"LEFT":-1,"RIGHT":1}).to_numpy(dtype=np.int8),
        sensory_laterality_code=chosen.sensory_laterality.map({"LEFT":-1,"RIGHT":1,"UNKNOWN":0}).to_numpy(dtype=np.int8),
        hex1=chosen.top_hex1.to_numpy(dtype=np.float64),hex2=chosen.top_hex2.to_numpy(dtype=np.float64),
        eligible_target_count=chosen.eligible_target_count.to_numpy(dtype=np.int64),dominance=chosen.dominance.to_numpy(dtype=np.float64),
        confidence_code=chosen.confidence_code.to_numpy(dtype=np.int8),subset_code=chosen.subset.map({"all_direct":0,"columnar_subset":1}).to_numpy(dtype=np.int8),
        is_cross_side=chosen.top_side_relation.eq("cross").to_numpy(dtype=bool),
        purpose=np.asarray("ANALYSIS_ONLY_not_WORLD_mapping_not_true_retinal_coordinates"),
        confidence_legend=np.asarray(["1=STRONG","2=VERY_STRONG_also_STRONG"]),subset_legend=np.asarray(["0=all_direct","1=columnar_subset"]),
        laterality_legend=np.asarray(["-1=LEFT","1=RIGHT","laterality_code=TARGET_side","sensory_laterality_code=SOURCE_side;0=UNKNOWN"]))
    generated.append(npz_path)
    block("GENERATED FILES")
    for path in generated: print(path)
    return pop,consensus,sweep,coordinates,occupancy


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,help="Default: script directory")
    parser.add_argument("--output",type=Path,help="Optional processed directory for validation")
    args=parser.parse_args()
    try:
        main(args.root,args.output)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}",file=sys.stderr)
        sys.exit(1)


