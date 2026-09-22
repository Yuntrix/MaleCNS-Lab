from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# MALECNS EXACT VISUAL LATERALITY BUILDER
#
# Exact types only:
#   LPLC2
#   LC4
#
# Anatomical laterality:
#   instance suffix _L / _R
#   somaSide L / R
#
# Output:
#   data/processed/lplc2_left_indices.npy
#   data/processed/lplc2_right_indices.npy
#   data/processed/lc4_left_indices.npy
#   data/processed/lc4_right_indices.npy
#   data/processed/looming_left_indices.npy
#   data/processed/looming_right_indices.npy
#
# IMPORTANT:
# Bunlar anatomik LEFT / RIGHT havuzlarıdır.
# Ekranın solu -> anatomik L mapping'i henüz burada yapılmaz.
# ============================================================


ROOT = Path(__file__).resolve().parent


SIM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "simulation-neurons.parquet"
)


ANNOTATION_FILE = (
    ROOT
    / "data"
    / "raw"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 90)
print("MALECNS EXACT VISUAL LATERALITY BUILDER")
print("=" * 90)


# ============================================================
# LOAD
# ============================================================

if not SIM_FILE.exists():

    raise FileNotFoundError(
        SIM_FILE
    )


if not ANNOTATION_FILE.exists():

    raise FileNotFoundError(
        ANNOTATION_FILE
    )


print()
print("Loading simulation neurons...")


sim = pd.read_parquet(
    SIM_FILE,
    columns=[
        "neuron_idx",
        "bodyId",
        "type",
    ]
)


print(
    f"Simulation neurons: {len(sim):,}"
)


print()
print("Loading annotations...")


annotations = pd.read_feather(
    ANNOTATION_FILE,
    columns=[
        "bodyId",
        "type",
        "instance",
        "somaSide",
    ]
)


print(
    f"Annotation rows: {len(annotations):,}"
)


# ============================================================
# CHECK BODYID DUPLICATES
# ============================================================

duplicates = annotations[
    annotations.duplicated(
        subset=["bodyId"],
        keep=False
    )
]


if len(duplicates) > 0:

    print()
    print(
        "WARNING: duplicate annotation bodyIds:",
        duplicates["bodyId"].nunique()
    )


    annotations = (
        annotations
        .drop_duplicates(
            subset=["bodyId"],
            keep="first"
        )
        .copy()
    )


# ============================================================
# RENAME ANNOTATION TYPE
# ============================================================

annotations = annotations.rename(
    columns={
        "type":
            "annotation_type"
    }
)


# ============================================================
# MERGE
# ============================================================

merged = sim.merge(
    annotations,
    on="bodyId",
    how="left",
    validate="one_to_one"
)


print()
print(
    f"Merged neurons: {len(merged):,}"
)


# ============================================================
# EXACT TYPE FILTER
#
# CRITICAL:
# str.contains("LC4") kullanmıyoruz.
# ============================================================

visual = merged[
    merged["type"].isin(
        [
            "LPLC2",
            "LC4",
        ]
    )
].copy()


print()
print("=" * 90)
print("EXACT VISUAL POPULATION")
print("=" * 90)


print(
    "Exact LPLC2:",
    int(
        (
            visual["type"]
            ==
            "LPLC2"
        ).sum()
    )
)


print(
    "Exact LC4:",
    int(
        (
            visual["type"]
            ==
            "LC4"
        ).sum()
    )
)


print(
    "Exact total:",
    len(visual)
)


# ============================================================
# INSTANCE SIDE
# ============================================================

def side_from_instance(
    value
):

    if pd.isna(
        value
    ):

        return None


    text = str(
        value
    ).strip()


    if text.endswith(
        "_L"
    ):

        return "L"


    if text.endswith(
        "_R"
    ):

        return "R"


    return None


visual[
    "instanceSide"
] = visual[
    "instance"
].apply(
    side_from_instance
)


# ============================================================
# NORMALIZE somaSide
# ============================================================

def normalize_side(
    value
):

    if pd.isna(
        value
    ):

        return None


    text = str(
        value
    ).strip().upper()


    if text in (
        "L",
        "R"
    ):

        return text


    return None


visual[
    "somaSideNormalized"
] = visual[
    "somaSide"
].apply(
    normalize_side
)


# ============================================================
# CHECK INSTANCE VS SOMA SIDE
# ============================================================

both_known = visual[
    visual["instanceSide"].notna()
    &
    visual["somaSideNormalized"].notna()
]


disagreements = both_known[
    both_known["instanceSide"]
    !=
    both_known["somaSideNormalized"]
]


print()
print("=" * 90)
print("LATERALITY CONSISTENCY")
print("=" * 90)


print(
    "Rows with both instance + soma side:",
    len(
        both_known
    )
)


print(
    "Disagreements:",
    len(
        disagreements
    )
)


if len(
    disagreements
) > 0:

    print()
    print(
        disagreements[
            [
                "bodyId",
                "neuron_idx",
                "type",
                "instance",
                "somaSide",
                "instanceSide",
                "somaSideNormalized",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL ANATOMICAL SIDE
#
# Prefer instance suffix.
# Fall back to somaSide.
# ============================================================

visual[
    "anatomicalSide"
] = visual[
    "instanceSide"
]


missing_instance_side = visual[
    "anatomicalSide"
].isna()


visual.loc[
    missing_instance_side,
    "anatomicalSide"
] = visual.loc[
    missing_instance_side,
    "somaSideNormalized"
]


# ============================================================
# UNKNOWN SIDE
# ============================================================

unknown = visual[
    ~visual[
        "anatomicalSide"
    ].isin(
        [
            "L",
            "R",
        ]
    )
]


print()
print(
    "Unknown-side visual neurons:",
    len(
        unknown
    )
)


if len(
    unknown
) > 0:

    print(
        unknown[
            [
                "bodyId",
                "neuron_idx",
                "type",
                "instance",
                "somaSide",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# EXACT POPULATIONS
# ============================================================

lplc2_left = visual[
    (
        visual["type"]
        ==
        "LPLC2"
    )
    &
    (
        visual["anatomicalSide"]
        ==
        "L"
    )
]


lplc2_right = visual[
    (
        visual["type"]
        ==
        "LPLC2"
    )
    &
    (
        visual["anatomicalSide"]
        ==
        "R"
    )
]


lc4_left = visual[
    (
        visual["type"]
        ==
        "LC4"
    )
    &
    (
        visual["anatomicalSide"]
        ==
        "L"
    )
]


lc4_right = visual[
    (
        visual["type"]
        ==
        "LC4"
    )
    &
    (
        visual["anatomicalSide"]
        ==
        "R"
    )
]


# ============================================================
# PRINT COUNTS
# ============================================================

print()
print("=" * 90)
print("EXACT ANATOMICAL POPULATIONS")
print("=" * 90)


print(
    f"LPLC2 LEFT : {len(lplc2_left)}"
)


print(
    f"LPLC2 RIGHT: {len(lplc2_right)}"
)


print(
    f"LC4 LEFT   : {len(lc4_left)}"
)


print(
    f"LC4 RIGHT  : {len(lc4_right)}"
)


print()


left_total = (
    len(
        lplc2_left
    )
    +
    len(
        lc4_left
    )
)


right_total = (
    len(
        lplc2_right
    )
    +
    len(
        lc4_right
    )
)


print(
    f"Combined anatomical LEFT : {left_total}"
)


print(
    f"Combined anatomical RIGHT: {right_total}"
)


print(
    f"Combined total           : "
    f"{left_total + right_total}"
)


# ============================================================
# ARRAY HELPER
# ============================================================

def neuron_indices(
    frame
):

    return (
        frame[
            "neuron_idx"
        ]
        .astype(
            np.int32
        )
        .to_numpy()
    )


lplc2_left_idx = neuron_indices(
    lplc2_left
)


lplc2_right_idx = neuron_indices(
    lplc2_right
)


lc4_left_idx = neuron_indices(
    lc4_left
)


lc4_right_idx = neuron_indices(
    lc4_right
)


looming_left_idx = np.concatenate(
    [
        lplc2_left_idx,
        lc4_left_idx,
    ]
).astype(
    np.int32
)


looming_right_idx = np.concatenate(
    [
        lplc2_right_idx,
        lc4_right_idx,
    ]
).astype(
    np.int32
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

if (
    len(
        np.unique(
            looming_left_idx
        )
    )
    !=
    len(
        looming_left_idx
    )
):

    raise RuntimeError(
        "Duplicate neuron_idx in LEFT pool."
    )


if (
    len(
        np.unique(
            looming_right_idx
        )
    )
    !=
    len(
        looming_right_idx
    )
):

    raise RuntimeError(
        "Duplicate neuron_idx in RIGHT pool."
    )


overlap = np.intersect1d(
    looming_left_idx,
    looming_right_idx
)


print()
print(
    "LEFT/RIGHT neuron overlap:",
    len(
        overlap
    )
)


if len(
    overlap
) > 0:

    raise RuntimeError(
        "LEFT and RIGHT populations overlap."
    )


# ============================================================
# SAVE NPY FILES
# ============================================================

files = {
    "lplc2_left_indices.npy":
        lplc2_left_idx,

    "lplc2_right_indices.npy":
        lplc2_right_idx,

    "lc4_left_indices.npy":
        lc4_left_idx,

    "lc4_right_indices.npy":
        lc4_right_idx,

    "looming_left_indices.npy":
        looming_left_idx,

    "looming_right_indices.npy":
        looming_right_idx,
}


print()
print("=" * 90)
print("SAVING")
print("=" * 90)


for filename, array in files.items():

    path = (
        OUTPUT_DIR
        /
        filename
    )


    np.save(
        path,
        array
    )


    print(
        f"{filename:<32} "
        f"{len(array):>4} neurons"
    )


# ============================================================
# SAVE HUMAN-READABLE CSV
# ============================================================

csv_file = (
    OUTPUT_DIR
    /
    "visual_laterality_exact.csv"
)


visual[
    [
        "neuron_idx",
        "bodyId",
        "type",
        "instance",
        "somaSide",
        "instanceSide",
        "anatomicalSide",
    ]
].sort_values(
    [
        "type",
        "anatomicalSide",
        "neuron_idx",
    ]
).to_csv(
    csv_file,
    index=False
)


print()
print(
    "CSV:",
    csv_file
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 90)
print("DONE")
print("=" * 90)


print(
    "Anatomical LEFT/RIGHT visual pools created."
)


print(
    "Screen LEFT/RIGHT mapping has NOT been assumed yet."
)


print(
    "Next step:"
)


print(
    "Stimulate anatomical LEFT vs RIGHT separately "
    "and measure MaleCNS descending/motor asymmetry."
)


print("=" * 90)