from pathlib import Path

import numpy as np
import pandas as pd

from brain_session import ContinuousBrainSession
from lif_engine_realtime import RealtimeMaleCNSLIF


# ============================================================
# MALECNS VISUAL -> DESCENDING STEERING CANDIDATE DISCOVERY
#
# Goal:
#   Find descending neuron TYPES that actually carry a robust
#   LEFT-vs-RIGHT signal in THIS MaleCNS + LIF model.
#
# Desired pattern:
#   visual LEFT  -> more activity on anatomical LEFT copy
#   visual RIGHT -> more activity on anatomical RIGHT copy
#
# IMPORTANT:
#   This is a model diagnostic, not proof of behavioral turning.
#   Do not connect any candidate to body steering until the
#   response is strong, repeatable, and inspected.
# ============================================================


ROOT = Path(__file__).resolve().parent

ANNOTATION_FILE = (
    ROOT
    / "data"
    / "raw"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

LEFT_SOURCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "looming_left_indices.npy"
)

RIGHT_SOURCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "looming_right_indices.npy"
)

OUTPUT_DIR = (
    ROOT
    / "laterality_reports"
)


# ============================================================
# MODEL / STIMULUS
# ============================================================

W_SYN = 0.110

STIM_RATE_HZ = 40.0

BASELINE_MS = 20.0
STIM_MS = 60.0
POST_MS = 120.0

SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

TOP_N = 30


# ============================================================
# HELPERS
# ============================================================

def side_from_instance(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith("_L"):
        return "L"

    if text.endswith("_R"):
        return "R"

    return None


def side_from_soma(value):

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {"L", "R"}:
        return text

    return None


def asymmetry(left_spikes, right_spikes):

    total = (
        left_spikes
        +
        right_spikes
    )

    if total <= 0:
        return 0.0

    # Negative = anatomical LEFT-biased
    # Positive = anatomical RIGHT-biased
    return (
        right_spikes
        -
        left_spikes
    ) / total


def configure_sources(
    session,
    source_indices
):

    source_indices = np.asarray(
        source_indices,
        dtype=np.int32
    )

    source_mask = np.zeros(
        session.brain.N,
        dtype=bool
    )

    source_mask[
        source_indices
    ] = True

    session.source_indices = np.ascontiguousarray(
        source_indices
    )

    session.source_mask = np.ascontiguousarray(
        source_mask
    )


def run_condition(
    brain,
    source_indices,
    seed
):

    session = ContinuousBrainSession(
        brain,
        source_types=[],
        seed=seed
    )

    configure_sources(
        session,
        source_indices
    )

    session.step(
        0.0,
        BASELINE_MS
    )

    stimulus = session.step(
        STIM_RATE_HZ,
        STIM_MS
    )

    post = session.step(
        0.0,
        POST_MS
    )

    spikes = (
        stimulus["spike_counts"].astype(
            np.int64
        )
        +
        post["spike_counts"].astype(
            np.int64
        )
    )

    return {
        "spikes": spikes,
        "stimulus_events": int(
            stimulus["stimulus_events"]
        ),
        "descending_total": int(
            spikes[
                brain.descending_mask
            ].sum()
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 110)
print("MALECNS VISUAL -> DESCENDING STEERING CANDIDATE DISCOVERY")
print("=" * 110)


for required_file in [
    ANNOTATION_FILE,
    LEFT_SOURCE_FILE,
    RIGHT_SOURCE_FILE,
]:

    if not required_file.exists():
        raise FileNotFoundError(
            required_file
        )


left_sources = np.load(
    LEFT_SOURCE_FILE
).astype(
    np.int32,
    copy=False
)

right_sources = np.load(
    RIGHT_SOURCE_FILE
).astype(
    np.int32,
    copy=False
)


print()
print(
    "Visual LEFT sources :",
    len(left_sources)
)

print(
    "Visual RIGHT sources:",
    len(right_sources)
)


# ============================================================
# BRAIN
# ============================================================

print()
print("Loading MaleCNS...")


brain = RealtimeMaleCNSLIF()

brain.w_syn = np.float32(
    W_SYN
)


print()
print(
    "W_SYN:",
    float(
        brain.w_syn
    )
)


# ============================================================
# METADATA
# ============================================================

print()
print("Loading laterality metadata...")


annotations = pd.read_feather(
    ANNOTATION_FILE,
    columns=[
        "bodyId",
        "instance",
        "somaSide",
    ]
)

annotations = annotations.drop_duplicates(
    subset=["bodyId"],
    keep="first"
)


meta = brain.neurons[
    [
        "neuron_idx",
        "bodyId",
        "type",
        "superclass",
    ]
].copy()


meta = meta.merge(
    annotations,
    on="bodyId",
    how="left",
    validate="one_to_one"
)


meta["instance_side"] = (
    meta["instance"]
    .apply(
        side_from_instance
    )
)

meta["soma_side"] = (
    meta["somaSide"]
    .apply(
        side_from_soma
    )
)

meta["side"] = (
    meta["instance_side"]
)

missing_side = (
    meta["side"]
    .isna()
)

meta.loc[
    missing_side,
    "side"
] = meta.loc[
    missing_side,
    "soma_side"
]


meta["type_clean"] = (
    meta["type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


descending_meta = meta[
    meta["superclass"]
    ==
    "descending_neuron"
].copy()


# ============================================================
# BUILD BILATERAL DESCENDING TYPE GROUPS
# ============================================================

groups = {}


for dn_type, group in descending_meta.groupby(
    "type_clean",
    sort=True
):

    if not dn_type:
        continue

    left_idx = (
        group.loc[
            group["side"] == "L",
            "neuron_idx"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    right_idx = (
        group.loc[
            group["side"] == "R",
            "neuron_idx"
        ]
        .to_numpy(
            dtype=np.int32
        )
    )

    if (
        len(left_idx) == 0
        or
        len(right_idx) == 0
    ):
        continue

    groups[
        dn_type
    ] = {
        "L": left_idx,
        "R": right_idx,
    }


print()
print(
    "Bilateral descending types:",
    len(groups)
)


# ============================================================
# RUN MATCHED LEFT / RIGHT TRIALS
# ============================================================

trial_rows = []


print()
print("=" * 110)
print("RUNNING MATCHED VISUAL LEFT / RIGHT TRIALS")
print("=" * 110)


for seed in SEEDS:

    left_result = run_condition(
        brain,
        left_sources,
        seed
    )

    right_result = run_condition(
        brain,
        right_sources,
        seed
    )

    print()
    print(
        f"Seed {seed}"
        f" | LEFT stim={left_result['stimulus_events']}"
        f" DN={left_result['descending_total']}"
        f" | RIGHT stim={right_result['stimulus_events']}"
        f" DN={right_result['descending_total']}"
    )

    left_spikes_all = (
        left_result["spikes"]
    )

    right_spikes_all = (
        right_result["spikes"]
    )

    for dn_type, sides in groups.items():

        indices_left = (
            sides["L"]
        )

        indices_right = (
            sides["R"]
        )

        visual_left_L = int(
            left_spikes_all[
                indices_left
            ].sum()
        )

        visual_left_R = int(
            left_spikes_all[
                indices_right
            ].sum()
        )

        visual_right_L = int(
            right_spikes_all[
                indices_left
            ].sum()
        )

        visual_right_R = int(
            right_spikes_all[
                indices_right
            ].sum()
        )

        visual_left_asym = asymmetry(
            visual_left_L,
            visual_left_R
        )

        visual_right_asym = asymmetry(
            visual_right_L,
            visual_right_R
        )

        shift = (
            visual_right_asym
            -
            visual_left_asym
        )

        desired_direction = int(
            visual_left_asym < 0
            and
            visual_right_asym > 0
        )

        total_spikes = (
            visual_left_L
            +
            visual_left_R
            +
            visual_right_L
            +
            visual_right_R
        )

        trial_rows.append(
            {
                "seed": seed,
                "type": dn_type,
                "left_neurons": int(
                    len(indices_left)
                ),
                "right_neurons": int(
                    len(indices_right)
                ),
                "visual_left_L": visual_left_L,
                "visual_left_R": visual_left_R,
                "visual_left_asym": visual_left_asym,
                "visual_right_L": visual_right_L,
                "visual_right_R": visual_right_R,
                "visual_right_asym": visual_right_asym,
                "shift": shift,
                "desired_direction": desired_direction,
                "total_spikes": total_spikes,
            }
        )


trials = pd.DataFrame(
    trial_rows
)


# ============================================================
# SUMMARY PER TYPE
# ============================================================

summary_rows = []


for dn_type, group in trials.groupby(
    "type",
    sort=True
):

    left_neurons = int(
        group["left_neurons"].iloc[0]
    )

    right_neurons = int(
        group["right_neurons"].iloc[0]
    )

    mean_visual_left_L = float(
        group["visual_left_L"].mean()
    )

    mean_visual_left_R = float(
        group["visual_left_R"].mean()
    )

    mean_visual_right_L = float(
        group["visual_right_L"].mean()
    )

    mean_visual_right_R = float(
        group["visual_right_R"].mean()
    )

    mean_left_asym = float(
        group["visual_left_asym"].mean()
    )

    mean_right_asym = float(
        group["visual_right_asym"].mean()
    )

    mean_shift = float(
        group["shift"].mean()
    )

    consistency = float(
        group["desired_direction"].mean()
    )

    total_spikes = int(
        group["total_spikes"].sum()
    )

    mean_spikes_per_seed = float(
        group["total_spikes"].mean()
    )

    directional_score = (
        max(
            mean_shift,
            0.0
        )
        *
        consistency
        *
        np.log1p(
            mean_spikes_per_seed
        )
    )

    summary_rows.append(
        {
            "type": dn_type,
            "left_neurons": left_neurons,
            "right_neurons": right_neurons,
            "visual_left_mean_L": mean_visual_left_L,
            "visual_left_mean_R": mean_visual_left_R,
            "visual_left_asym": mean_left_asym,
            "visual_right_mean_L": mean_visual_right_L,
            "visual_right_mean_R": mean_visual_right_R,
            "visual_right_asym": mean_right_asym,
            "mean_shift": mean_shift,
            "consistency": consistency,
            "mean_spikes_per_seed": mean_spikes_per_seed,
            "total_spikes_all_trials": total_spikes,
            "directional_score": directional_score,
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(
    by=[
        "directional_score",
        "consistency",
        "mean_shift",
        "mean_spikes_per_seed",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ]
).reset_index(
    drop=True
)


# ============================================================
# ACTIVE TYPES, REGARDLESS OF DIRECTION
# ============================================================

active = summary.sort_values(
    by=[
        "mean_spikes_per_seed",
        "mean_shift",
    ],
    ascending=[
        False,
        False,
    ]
).reset_index(
    drop=True
)


# ============================================================
# PRINT
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

pd.set_option(
    "display.max_colwidth",
    40
)


print()
print("=" * 110)
print("TOP DIRECTIONAL CANDIDATES")
print("=" * 110)

display_columns = [
    "type",
    "left_neurons",
    "right_neurons",
    "visual_left_mean_L",
    "visual_left_mean_R",
    "visual_left_asym",
    "visual_right_mean_L",
    "visual_right_mean_R",
    "visual_right_asym",
    "mean_shift",
    "consistency",
    "mean_spikes_per_seed",
    "directional_score",
]


print(
    summary[
        display_columns
    ]
    .head(
        TOP_N
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


print()
print("=" * 110)
print("MOST ACTIVE BILATERAL DESCENDING TYPES")
print("=" * 110)


print(
    active[
        display_columns
    ]
    .head(
        TOP_N
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# EXPLICIT DNa01 / DNa02 CHECK
# ============================================================

print()
print("=" * 110)
print("DNa01 / DNa02 CHECK")
print("=" * 110)


for dn_type in [
    "DNa01",
    "DNa02",
]:

    row = summary[
        summary["type"]
        ==
        dn_type
    ]

    if row.empty:

        print(
            dn_type,
            ": bilateral type not found"
        )

    else:

        print()
        print(
            row[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.3f}"
            )
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


trials_file = (
    OUTPUT_DIR
    /
    "visual_dn_candidate_trials.csv"
)

summary_file = (
    OUTPUT_DIR
    /
    "visual_dn_candidate_summary.csv"
)


trials.to_csv(
    trials_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)


print()
print("=" * 110)
print("FILES SAVED")
print("=" * 110)

print(
    trials_file
)

print(
    summary_file
)


print()
print("=" * 110)
print("DONE")
print("=" * 110)

print(
    "Do not connect a candidate to body steering yet."
)

print(
    "First inspect whether one or more descending TYPES show "
    "strong activity, opposite LEFT/RIGHT asymmetry, and "
    "high seed-to-seed consistency."
)

print("=" * 110)