"""Read-only MaleCNS population discovery; no capture, simulation or injection.

Run: python visual_retina_encoder_v1.py
Dependencies: numpy, pandas, pyarrow.
R1-R6 remains a group; rootSide is corroboration, not a soma-side substitute.
Edge counts are processed edge ROW counts, not inferred synapse counts.
"""
from pathlib import Path
import argparse
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def block(title):
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)


def text(value):
    if value is None:
        return "unknown"
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value.tolist() if isinstance(value, np.ndarray) else value)
    if pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value).strip()


def column(frame, name):
    matches = [c for c in frame.columns if c.casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous column {name}: {matches}")
    return matches[0] if matches else None


def values(frame, name):
    c = column(frame, name)
    return frame[c].map(text) if c else pd.Series("unknown", index=frame.index)


def integer_column(frame, name):
    c = column(frame, name)
    if not c:
        raise ValueError(f"Required identifier column missing: {name}")
    raw = frame[c]
    # Refuse floats: IDs must not silently lose precision.
    if raw.isna().any() or not raw.map(lambda v: bool(re.fullmatch(r"\d+", str(v)))).all():
        raise ValueError(f"{name} contains missing/non-integer IDs")
    frame[c] = raw.astype("int64")
    if c != name:
        frame.rename(columns={c: name}, inplace=True)


def identity(row):
    """Only exact annotation labels. No substring-based R-number inference."""
    evidence = []
    categories = set()
    for field in ("type", "flywireType", "hemibrainType", "receptorType", "instance"):
        v = row.get(field, "unknown")
        label = re.sub(r"_[LR]$", "", v) if field == "instance" else v
        if re.fullmatch(r"R[1-8]", label, flags=re.I):
            category = label.upper()
        elif label.casefold() in ("r1-r6", "r1-6"):
            category = "R1-R6_group_unresolved"
        elif label.casefold() == "photoreceptor":
            category = "photoreceptor_unspecified"
        else:
            continue
        categories.add(category)
        evidence.append(f"{field}={v}")
    specific = categories - {"photoreceptor_unspecified"}
    if len(specific) > 1:
        return "unknown", "disagreement: " + "; ".join(evidence)
    return (next(iter(specific)) if specific else
            "photoreceptor_unspecified" if categories else "unknown"), "; ".join(evidence) or "unknown"


def side(v):
    return {"L": "LEFT", "LEFT": "LEFT", "R": "RIGHT", "RIGHT": "RIGHT"}.get(v.upper(), "UNKNOWN")


def laterality(row):
    soma = side(row.get("somaSide", "unknown"))
    match = re.search(r"_([LR])$", row.get("instance", "unknown"))
    instance = side(match.group(1)) if match else "UNKNOWN"
    root = side(row.get("rootSide", "unknown"))
    known = {v for v in (soma, instance, root) if v != "UNKNOWN"}
    conflict = len(known) > 1
    primary = {v for v in (soma, instance) if v != "UNKNOWN"}
    resolved = next(iter(primary)) if len(primary) == 1 and not conflict else "UNKNOWN"
    return soma, instance, root, resolved, conflict


def main(root=None, output=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    processed = Path(output) if output else root / "data/processed"
    reports = processed / "retina_reports"
    ap = root / "data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    npth = root / "data/processed/simulation-neurons.parquet"
    ep = root / "data/processed/simulation-edges.parquet"
    block("FILES / SCOPE")
    for p in (ap, npth, ep):
        print(p)
        if not p.is_file():
            raise FileNotFoundError(p)
    print("Annotation discovery only. No OBS, AI API, LIF, motor or spike injection.")
    print("Neurotransmitter file not required; transmitter is not used to infer cell identity.")
    a = pd.read_feather(ap)
    n = pd.read_parquet(npth)
    block("ANNOTATION COLUMNS")
    print("\n".join(a.columns))
    integer_column(a, "bodyId")
    integer_column(n, "bodyId")
    integer_column(n, "neuron_idx")
    if a.bodyId.duplicated().any() or n.bodyId.duplicated().any() or n.neuron_idx.duplicated().any():
        raise ValueError("Duplicate bodyId/neuron_idx: ambiguous mapping; no reports overwritten.")
    if set(n.neuron_idx) != set(range(len(n))):
        raise ValueError("neuron_idx is not a complete zero-based simulation index set")
    fields = ["instance", "type", "somaSide", "rootSide", "group", "class", "subclass",
              "system", "superclass", "hemibrainType", "flywireType", "receptorType", "supertype"]
    metadata = pd.DataFrame({"bodyId": a.bodyId})
    for f in fields:
        metadata[f] = values(a, f)
    discovery_fields = [f for f in ("superclass", "class", "subclass", "system") if column(a, f)]
    masks = {f: values(a, f).str.casefold().eq("ol_sensory") for f in discovery_fields}
    mask = pd.Series(False, index=a.index)
    for m in masks.values():
        mask |= m
    block("OL_SENSORY POPULATION")
    print("Exact ol_sensory annotation matches:", {f: int(m.sum()) for f, m in masks.items()})
    pop = metadata.loc[mask].copy()
    pop["population_evidence"] = ["; ".join(f"{f}=ol_sensory" for f, m in masks.items() if m.loc[i]) for i in pop.index]
    print("Unique annotated neurons:", len(pop))
    print("Missing optional fields:", [f for f in fields if column(a, f) is None])
    if pop.empty:
        print("No explicit ol_sensory population found. Identity is unknown; no population inferred.")
    ids = [identity(row) for row in pop.to_dict("records")]
    pop["retina_identity"] = [v[0] for v in ids]
    pop["identity_evidence"] = [v[1] for v in ids]
    sides = [laterality(row) for row in pop.to_dict("records")]
    for j, f in enumerate(("side_soma", "side_instance", "side_root", "side_consensus", "side_disagreement")):
        pop[f] = [v[j] for v in sides]
    pop = pop.merge(n[["bodyId", "neuron_idx"]], on="bodyId", how="left", validate="one_to_one")
    pop["neuron_idx"] = pop.neuron_idx.astype("Int64")
    pop["in_simulation"] = pop.neuron_idx.notna()
    mapped = pop[pop.in_simulation].copy()
    block("R1-R8 / LABEL EVIDENCE")
    print("Exact labels only; R1-R6 is NOT split into individual R1...R6.")
    print("R7/R8 subtype membership is used only when another field explicitly says R7/R8.")
    summaries = []
    for scope, frame in (("annotation_ol_sensory", pop), ("simulation_ol_sensory", mapped)):
        for label in [f"R{i}" for i in range(1, 9)] + ["R1-R6_group_unresolved", "photoreceptor_unspecified", "unknown"]:
            count = int(frame.retina_identity.eq(label).sum())
            summaries.append(dict(scope=scope, field="retina_identity", label=label, neuron_count=count))
            print(scope, label, count)
        for f in fields:
            for label, count in frame[f].value_counts(dropna=False).items():
                summaries.append(dict(scope=scope, field=f, label=label, neuron_count=int(count)))
    print("\nRaw type distribution:")
    print(pop.type.value_counts().to_string())
    # Independent explicit photoreceptor labels, including rows also assigned an R identity.
    for scope, frame in (("annotation", pop), ("simulation", mapped)):
        explicit = frame[["type", "class", "subclass", "receptorType", "flywireType", "hemibrainType"]].apply(lambda col: col.str.casefold().eq("photoreceptor")).any(axis=1)
        print(scope, "explicit photoreceptor label:", int(explicit.sum()))
        summaries.append(dict(scope=scope, field="explicit_photoreceptor_label", label="photoreceptor", neuron_count=int(explicit.sum())))
    block("LATERALITY")
    lateral_rows = []
    for scope, frame in (("annotation", pop), ("simulation", mapped)):
        for f in ("side_soma", "side_instance", "side_root", "side_consensus"):
            for label in ("LEFT", "RIGHT", "UNKNOWN"):
                count = int(frame[f].eq(label).sum())
                lateral_rows.append(dict(scope=scope, source=f, side=label, neuron_count=count))
        count = int(frame.side_disagreement.sum())
        lateral_rows.append(dict(scope=scope, source="disagreement_any_source", side="UNKNOWN", neuron_count=count))
        for left, right in (("side_soma", "side_instance"), ("side_soma", "side_root"), ("side_instance", "side_root")):
            conflict = frame[left].ne("UNKNOWN") & frame[right].ne("UNKNOWN") & frame[left].ne(frame[right])
            lateral_rows.append(dict(scope=scope, source=left + "_vs_" + right, side="DISAGREEMENT", neuron_count=int(conflict.sum())))
        print(scope, frame.side_consensus.value_counts().to_dict(), "disagreements:", count)
    print("rootSide is reported as root side; alone it does not assign soma/body laterality.")
    print("Any conflict -> UNKNOWN. Missing fields -> unknown, not inferred.")
    block("SIMULATION MATCH")
    print("Annotated:", len(pop), "Matched:", len(mapped), "Not in simulation:", len(pop) - len(mapped))
    print("Indices use explicit neuron_idx joined by bodyId; not annotation row numbers.")
    print("Simulation index sample:", mapped.neuron_idx.head(20).tolist())
    sim_meta = n[["bodyId", "neuron_idx"]].merge(metadata, on="bodyId", how="left", validate="one_to_one")
    for f in fields:
        sim_meta[f] = sim_meta[f].map(text)
    sim_meta = sim_meta.set_index("neuron_idx")
    edge_file = pq.ParquetFile(ep)
    if not {"pre_idx", "post_idx"}.issubset(edge_file.schema_arrow.names):
        raise ValueError("Unknown edge schema: explicit pre_idx/post_idx required; no orientation guessed")
    block("DOWNSTREAM / PROCESSED EDGES")
    print("Columns:", edge_file.schema_arrow.names)
    print("Counting stored outgoing edge rows, including population-internal edges.")
    print("Weights are not interpreted as synapse counts or biological effects.")
    member = np.zeros(len(n), dtype=bool)
    member[mapped.neuron_idx.to_numpy(dtype=np.int64)] = True
    counts = np.zeros(len(n), dtype=np.int64)
    total_rows = 0
    for batch in edge_file.iter_batches(batch_size=500000, columns=["pre_idx", "post_idx"]):
        frame = batch.to_pandas()
        for c in ("pre_idx", "post_idx"):
            if frame[c].isna().any() or not pd.api.types.is_integer_dtype(frame[c]):
                raise ValueError(f"Invalid edge indices: {c}")
            if ((frame[c] < 0) | (frame[c] >= len(n))).any():
                raise ValueError(f"Out-of-range edge indices: {c}")
        pre = frame.pre_idx.to_numpy(dtype=np.int64)
        post = frame.post_idx.to_numpy(dtype=np.int64)
        selected = post[member[pre]]
        np.add.at(counts, selected, 1)
        total_rows += len(frame)
    targets = sim_meta.loc[np.flatnonzero(counts)].copy()
    targets["outgoing_edge_rows"] = counts[targets.index.to_numpy()]
    print("Processed rows scanned:", total_rows)
    print("Total outgoing edges:", int(counts.sum()))
    print("Unique downstream neurons:", len(targets))
    downstream = []
    for field in ("type", "class", "subclass", "system", "superclass"):
        for label, group in targets.groupby(field, dropna=False):
            downstream.append(dict(field=field, label=label, unique_downstream_neurons=len(group), outgoing_edge_rows=int(group.outgoing_edge_rows.sum()), explicit_visual_label=bool(re.search(r"visual|photoreceptor|^ol_", label, re.I))))
    ds = pd.DataFrame(downstream, columns=["field", "label", "unique_downstream_neurons", "outgoing_edge_rows", "explicit_visual_label"])
    ds = ds.sort_values(["field", "outgoing_edge_rows", "label"], ascending=[True, False, True])
    top = ds[ds.field.eq("type")].head(20).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    print("\nTop 20 target types ranked by outgoing edge rows:\n", top.to_string(index=False))
    print("\nClass/system distributions:\n", ds[ds.field.isin(["class", "system", "superclass"])].to_string(index=False))
    print("\nExplicit visual-related downstream LABELS (not function predictions):\n", ds[ds.explicit_visual_label.astype(bool)].to_string(index=False))
    print("Unique-neuron counts across DIFFERENT fields must not be added together.")
    reports.mkdir(parents=True, exist_ok=True)
    generated = []
    for name, frame in (("visual_retina_population.csv", pop), ("visual_retina_type_summary.csv", pd.DataFrame(summaries)), ("visual_retina_laterality_summary.csv", pd.DataFrame(lateral_rows)), ("visual_retina_downstream_summary.csv", ds), ("visual_retina_downstream_top_targets.csv", top)):
        path = reports / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        generated.append(path)
    # These arrays represent annotation-defined ol_sensory, NOT proven retina input dynamics.
    reliable = len(mapped) > 0 and not mapped.side_disagreement.any() and mapped.side_consensus.ne("UNKNOWN").all()
    selections = {"all": mapped}
    if reliable:
        selections.update(left=mapped[mapped.side_consensus.eq("LEFT")], right=mapped[mapped.side_consensus.eq("RIGHT")])
    else:
        print("LEFT/RIGHT arrays NOT generated: empty population, unknown laterality or disagreement in simulation subset.")
        for label in ("left", "right"):
            existing = processed / f"retina_{label}_indices.npy"
            if existing.exists():
                print("WARNING: pre-existing array NOT validated by this run; do not use:", existing)
    if mapped.empty:
        selections = {}
        print("ALL array NOT generated: no matched explicit ol_sensory population.")
    for label, frame in selections.items():
        path = processed / f"retina_{label}_indices.npy"
        np.save(path, np.sort(frame.neuron_idx.to_numpy(dtype=np.int64)), allow_pickle=False)
        generated.append(path)
    block("GENERATED FILES")
    for path in generated:
        print(path)
    print("Done. Connectivity and annotation report only; biological function remains untested.")
    return pop, ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Default: script directory")
    parser.add_argument("--output", type=Path, default=None, help="Optional processed output directory for validation")
    args = parser.parse_args()
    try:
        main(args.root, args.output)
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
