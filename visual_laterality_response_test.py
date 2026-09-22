from pathlib import Path

import numpy as np
import pandas as pd

from brain_session import ContinuousBrainSession
from lif_engine_realtime import RealtimeMaleCNSLIF
from motor_decoder import MotorDecoder


# ============================================================
# MALECNS VISUAL LATERALITY RESPONSE TEST
#
# PURPOSE
# -------
# Stimulate the anatomically annotated LPLC2 + LC4 LEFT pool
# and RIGHT pool separately, then measure:
#
#   - descending LEFT / RIGHT spikes
#   - motor LEFT / RIGHT spikes
#   - flight motor LEFT / RIGHT spikes
#   - non-flight motor LEFT / RIGHT spikes
#
# IMPORTANT
# ---------
# This does NOT assume:
#   screen-left == anatomical-left
#   anatomical-left visual input == left turn/right turn
#
# We measure the connectome response first.
#
# MaleCNS connectivity is real connectome structure.
# LIF dynamics, W_SYN, stimulus rate, and encoding are modeled.
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


# ============================================================
# MODEL / STIMULUS SETTINGS
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


# ============================================================
# HELPERS
# ============================================================

def instance_side(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith("_L"):
        return "L"

    if text.endswith("_R"):
        return "R"

    if text.endswith("_M"):
        return "M"

    return None


def normalize_side(value):

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {"L", "R", "M"}:
        return text

    return None


def asymmetry_index(left_spikes, right_spikes):

    total = left_spikes + right_spikes

    if total <= 0:
        return 0.0

    # Positive = more RIGHT-side spikes.
    # Negative = more LEFT-side spikes.
    return (
        right_spikes - left_spikes
    ) / total


def configure_session_sources(session, source_indices):

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


def add_result_spikes(total, result):

    total += result["spike_counts"]


def make_side_masks(brain, annotations):

    meta = brain.neurons[
        [
            "neuron_idx",
            "bodyId",
            "type",
            "superclass",
        ]
    ].copy()

    raw = annotations[
        [
            "bodyId",
            "instance",
            "somaSide",
        ]
    ].copy()

    raw = raw.drop_duplicates(
        subset=["bodyId"],
        keep="first"
    )

    meta = meta.merge(
        raw,
        on="bodyId",
        how="left",
        validate="one_to_one"
    )

    meta["instanceSide"] = (
        meta["instance"]
        .apply(instance_side)
    )

    meta["somaSideNormalized"] = (
        meta["somaSide"]
        .apply(normalize_side)
    )

    meta["anatomicalSide"] = (
        meta["instanceSide"]
    )

    missing = (
        meta["anatomicalSide"]
        .isna()
    )

    meta.loc[
        missing,
        "anatomicalSide"
    ] = meta.loc[
        missing,
        "somaSideNormalized"
    ]

    descending = (
        meta["superclass"]
        ==
        "descending_neuron"
    )

    motor = (
        meta["superclass"]
        .isin(
            [
                "vnc_motor",
                "cb_motor",
            ]
        )
    )

    left = (
        meta["anatomicalSide"]
        ==
        "L"
    )

    right = (
        meta["anatomicalSide"]
        ==
        "R"
    )

    middle = (
        meta["anatomicalSide"]
        ==
        "M"
    )

    known_side = (
        left
        |
        right
        |
        middle
    )

    unknown = (
        ~known_side
    )

    decoder = MotorDecoder()

    type_clean = (
        meta["type"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    flight = (
        type_clean.isin(
            decoder.flight_types
        )
    )

    masks = {
        "descending_left":
            (
                descending
                &
                left
            ).to_numpy(),

        "descending_right":
            (
                descending
                &
                right
            ).to_numpy(),

        "descending_middle":
            (
                descending
                &
                middle
            ).to_numpy(),

        "descending_unknown":
            (
                descending
                &
                unknown
            ).to_numpy(),

        "motor_left":
            (
                motor
                &
                left
            ).to_numpy(),

        "motor_right":
            (
                motor
                &
                right
            ).to_numpy(),

        "motor_middle":
            (
                motor
                &
                middle
            ).to_numpy(),

        "motor_unknown":
            (
                motor
                &
                unknown
            ).to_numpy(),

        "flight_left":
            (
                motor
                &
                flight
                &
                left
            ).to_numpy(),

        "flight_right":
            (
                motor
                &
                flight
                &
                right
            ).to_numpy(),

        "other_motor_left":
            (
                motor
                &
                ~flight
                &
                left
            ).to_numpy(),

        "other_motor_right":
            (
                motor
                &
                ~flight
                &
                right
            ).to_numpy(),
    }

    return meta, masks


def count_mask(mask):

    return int(
        np.count_nonzero(
            mask
        )
    )


def spikes_in(spike_counts, mask):

    return int(
        spike_counts[
            mask
        ].sum()
    )


def run_condition(
    brain,
    source_indices,
    side_name,
    seed,
    masks,
    decoder
):

    # Create an empty-source session first.
    # Then install the exact anatomical neuron_idx pool.
    session = ContinuousBrainSession(
        brain,
        source_types=[],
        seed=seed
    )

    configure_session_sources(
        session,
        source_indices
    )

    # --------------------------------------------------------
    # 20 ms baseline
    # --------------------------------------------------------

    baseline = session.step(
        0.0,
        BASELINE_MS
    )

    # --------------------------------------------------------
    # 60 ms modeled visual drive
    # --------------------------------------------------------

    stimulus = session.step(
        STIM_RATE_HZ,
        STIM_MS
    )

    # --------------------------------------------------------
    # 120 ms post-stimulus propagation
    # --------------------------------------------------------

    post = session.step(
        0.0,
        POST_MS
    )

    # Response window excludes the pre-stimulus baseline.
    response_spikes = np.zeros(
        brain.N,
        dtype=np.int64
    )

    add_result_spikes(
        response_spikes,
        stimulus
    )

    add_result_spikes(
        response_spikes,
        post
    )

    decoded = decoder.decode(
        brain.neurons,
        response_spikes,
        brain.motor_mask
    )

    dn_left = spikes_in(
        response_spikes,
        masks["descending_left"]
    )

    dn_right = spikes_in(
        response_spikes,
        masks["descending_right"]
    )

    motor_left = spikes_in(
        response_spikes,
        masks["motor_left"]
    )

    motor_right = spikes_in(
        response_spikes,
        masks["motor_right"]
    )

    flight_left = spikes_in(
        response_spikes,
        masks["flight_left"]
    )

    flight_right = spikes_in(
        response_spikes,
        masks["flight_right"]
    )

    other_left = spikes_in(
        response_spikes,
        masks["other_motor_left"]
    )

    other_right = spikes_in(
        response_spikes,
        masks["other_motor_right"]
    )

    return {
        "condition":
            side_name,

        "seed":
            seed,

        "source_neurons":
            int(
                len(
                    source_indices
                )
            ),

        "stimulus_events":
            int(
                stimulus["stimulus_events"]
            ),

        "response_total_spikes":
            int(
                response_spikes.sum()
            ),

        "descending_total":
            int(
                response_spikes[
                    brain.descending_mask
                ].sum()
            ),

        "dn_left":
            dn_left,

        "dn_right":
            dn_right,

        "dn_asym":
            asymmetry_index(
                dn_left,
                dn_right
            ),

        "motor_total":
            int(
                response_spikes[
                    brain.motor_mask
                ].sum()
            ),

        "motor_left":
            motor_left,

        "motor_right":
            motor_right,

        "motor_asym":
            asymmetry_index(
                motor_left,
                motor_right
            ),

        "flight_left":
            flight_left,

        "flight_right":
            flight_right,

        "flight_asym":
            asymmetry_index(
                flight_left,
                flight_right
            ),

        "other_motor_left":
            other_left,

        "other_motor_right":
            other_right,

        "other_motor_asym":
            asymmetry_index(
                other_left,
                other_right
            ),

        "decoded_action":
            decoded["action"],

        "decoded_flight_spikes":
            decoded["flight_spikes"],

        "decoded_jump_spikes":
            decoded["jump_spikes"],

        "decoded_other_spikes":
            decoded["other_spikes"],
    }


# ============================================================
# LOAD INPUTS
# ============================================================

print("=" * 100)
print("MALECNS VISUAL LATERALITY RESPONSE TEST")
print("=" * 100)


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
    "Anatomical visual LEFT sources :",
    len(
        left_sources
    )
)

print(
    "Anatomical visual RIGHT sources:",
    len(
        right_sources
    )
)


# ============================================================
# LOAD BRAIN
# ============================================================

print()
print("Loading MaleCNS realtime engine...")


brain = RealtimeMaleCNSLIF()


# CRITICAL:
# The class default is 0.275, but this project calibrated
# the current long-running MaleCNS model to W_SYN = 0.110.
brain.w_syn = np.float32(
    W_SYN
)


print()
print(
    "Using calibrated modeled W_SYN:",
    float(
        brain.w_syn
    )
)


# ============================================================
# LOAD ANNOTATIONS / MASKS
# ============================================================

print()
print("Loading anatomical annotations...")


annotations = pd.read_feather(
    ANNOTATION_FILE,
    columns=[
        "bodyId",
        "instance",
        "somaSide",
    ]
)


meta, masks = make_side_masks(
    brain,
    annotations
)


print()
print("=" * 100)
print("ANATOMICAL OUTPUT MASK COUNTS")
print("=" * 100)


for name in [
    "descending_left",
    "descending_right",
    "descending_middle",
    "descending_unknown",
    "motor_left",
    "motor_right",
    "motor_middle",
    "motor_unknown",
    "flight_left",
    "flight_right",
    "other_motor_left",
    "other_motor_right",
]:

    print(
        f"{name:<24}: "
        f"{count_mask(masks[name]):,}"
    )


# ============================================================
# RUN MATCHED PAIRS
# ============================================================

decoder = MotorDecoder()

rows = []


print()
print("=" * 100)
print("RUNNING MATCHED LEFT / RIGHT STIMULUS PAIRS")
print("=" * 100)


for seed in SEEDS:

    print()
    print(
        f"Seed {seed}"
    )

    left_result = run_condition(
        brain,
        left_sources,
        "ANATOMICAL_LEFT",
        seed,
        masks,
        decoder
    )

    right_result = run_condition(
        brain,
        right_sources,
        "ANATOMICAL_RIGHT",
        seed,
        masks,
        decoder
    )

    rows.append(
        left_result
    )

    rows.append(
        right_result
    )

    print(
        " LEFT "
        f"| stim={left_result['stimulus_events']:4d} "
        f"| DN L/R={left_result['dn_left']:4d}/"
        f"{left_result['dn_right']:4d} "
        f"| MOTOR L/R={left_result['motor_left']:4d}/"
        f"{left_result['motor_right']:4d} "
        f"| FLIGHT L/R={left_result['flight_left']:4d}/"
        f"{left_result['flight_right']:4d} "
        f"| action={left_result['decoded_action']}"
    )

    print(
        " RIGHT"
        f" | stim={right_result['stimulus_events']:4d} "
        f"| DN L/R={right_result['dn_left']:4d}/"
        f"{right_result['dn_right']:4d} "
        f"| MOTOR L/R={right_result['motor_left']:4d}/"
        f"{right_result['motor_right']:4d} "
        f"| FLIGHT L/R={right_result['flight_left']:4d}/"
        f"{right_result['flight_right']:4d} "
        f"| action={right_result['decoded_action']}"
    )


# ============================================================
# DATAFRAME
# ============================================================

results = pd.DataFrame(
    rows
)


# ============================================================
# NORMALIZED OUTPUTS
# ============================================================

dn_left_count = count_mask(
    masks["descending_left"]
)

dn_right_count = count_mask(
    masks["descending_right"]
)

motor_left_count = count_mask(
    masks["motor_left"]
)

motor_right_count = count_mask(
    masks["motor_right"]
)


results["dn_left_per_neuron"] = (
    results["dn_left"]
    /
    max(
        dn_left_count,
        1
    )
)

results["dn_right_per_neuron"] = (
    results["dn_right"]
    /
    max(
        dn_right_count,
        1
    )
)

results["motor_left_per_neuron"] = (
    results["motor_left"]
    /
    max(
        motor_left_count,
        1
    )
)

results["motor_right_per_neuron"] = (
    results["motor_right"]
    /
    max(
        motor_right_count,
        1
    )
)


# Normalize gross response by actual stimulus events,
# because LEFT has 165 sources and RIGHT has 146.
stim_denominator = (
    results["stimulus_events"]
    .clip(
        lower=1
    )
)

results["dn_spikes_per_stim_event"] = (
    results["descending_total"]
    /
    stim_denominator
)

results["motor_spikes_per_stim_event"] = (
    results["motor_total"]
    /
    stim_denominator
)


# ============================================================
# SUMMARY
# ============================================================

summary_columns = [
    "stimulus_events",
    "response_total_spikes",
    "descending_total",
    "dn_left",
    "dn_right",
    "dn_asym",
    "motor_total",
    "motor_left",
    "motor_right",
    "motor_asym",
    "flight_left",
    "flight_right",
    "flight_asym",
    "other_motor_left",
    "other_motor_right",
    "other_motor_asym",
    "dn_spikes_per_stim_event",
    "motor_spikes_per_stim_event",
]


summary = (
    results
    .groupby(
        "condition"
    )[summary_columns]
    .agg(
        [
            "mean",
            "std",
        ]
    )
)


print()
print("=" * 100)
print("AVERAGE RESPONSE ACROSS SEEDS")
print("=" * 100)


for condition in [
    "ANATOMICAL_LEFT",
    "ANATOMICAL_RIGHT",
]:

    subset = results[
        results["condition"]
        ==
        condition
    ]

    print()
    print(
        condition
    )

    print(
        "-" * 100
    )

    print(
        "Mean stimulus events:",
        f"{subset['stimulus_events'].mean():.2f}"
    )

    print(
        "Mean DN L/R:",
        f"{subset['dn_left'].mean():.2f} / "
        f"{subset['dn_right'].mean():.2f}"
    )

    print(
        "Mean DN asymmetry "
        "(+R, -L):",
        f"{subset['dn_asym'].mean():+.4f}"
    )

    print(
        "Mean MOTOR L/R:",
        f"{subset['motor_left'].mean():.2f} / "
        f"{subset['motor_right'].mean():.2f}"
    )

    print(
        "Mean MOTOR asymmetry "
        "(+R, -L):",
        f"{subset['motor_asym'].mean():+.4f}"
    )

    print(
        "Mean FLIGHT L/R:",
        f"{subset['flight_left'].mean():.2f} / "
        f"{subset['flight_right'].mean():.2f}"
    )

    print(
        "Mean FLIGHT asymmetry "
        "(+R, -L):",
        f"{subset['flight_asym'].mean():+.4f}"
    )

    print(
        "DN spikes / stimulus event:",
        f"{subset['dn_spikes_per_stim_event'].mean():.4f}"
    )

    print(
        "Motor spikes / stimulus event:",
        f"{subset['motor_spikes_per_stim_event'].mean():.4f}"
    )

    print(
        "Actions:",
        subset[
            "decoded_action"
        ].value_counts().to_dict()
    )


# ============================================================
# PAIRED ASYMMETRY DIFFERENCES
# ============================================================

left_rows = (
    results[
        results["condition"]
        ==
        "ANATOMICAL_LEFT"
    ]
    .set_index(
        "seed"
    )
)

right_rows = (
    results[
        results["condition"]
        ==
        "ANATOMICAL_RIGHT"
    ]
    .set_index(
        "seed"
    )
)


paired = pd.DataFrame(
    index=SEEDS
)


paired["dn_asym_left_stim"] = (
    left_rows["dn_asym"]
)

paired["dn_asym_right_stim"] = (
    right_rows["dn_asym"]
)

paired["dn_asym_shift"] = (
    paired["dn_asym_right_stim"]
    -
    paired["dn_asym_left_stim"]
)


paired["motor_asym_left_stim"] = (
    left_rows["motor_asym"]
)

paired["motor_asym_right_stim"] = (
    right_rows["motor_asym"]
)

paired["motor_asym_shift"] = (
    paired["motor_asym_right_stim"]
    -
    paired["motor_asym_left_stim"]
)


print()
print("=" * 100)
print("PAIRED LATERALITY SHIFT")
print("=" * 100)


print(
    paired.to_string()
)


print()
print(
    "Mean DN asymmetry shift "
    "(RIGHT-stim minus LEFT-stim):",
    f"{paired['dn_asym_shift'].mean():+.4f}"
)


print(
    "Mean MOTOR asymmetry shift "
    "(RIGHT-stim minus LEFT-stim):",
    f"{paired['motor_asym_shift'].mean():+.4f}"
)


# ============================================================
# SAVE
# ============================================================

output_dir = (
    ROOT
    /
    "laterality_reports"
)

output_dir.mkdir(
    exist_ok=True
)


results_file = (
    output_dir
    /
    "visual_laterality_response_trials.csv"
)


paired_file = (
    output_dir
    /
    "visual_laterality_response_paired.csv"
)


summary_file = (
    output_dir
    /
    "visual_laterality_response_summary.csv"
)


results.to_csv(
    results_file,
    index=False
)


paired.to_csv(
    paired_file,
    index=True,
    index_label="seed"
)


summary.to_csv(
    summary_file
)


print()
print("=" * 100)
print("FILES SAVED")
print("=" * 100)


print(
    results_file
)

print(
    paired_file
)

print(
    summary_file
)


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Do NOT map screen LEFT/RIGHT to a turn direction yet."
)

print(
    "First inspect whether anatomical LEFT vs RIGHT stimulation "
    "causes a repeatable descending/motor asymmetry shift."
)

print("=" * 100)