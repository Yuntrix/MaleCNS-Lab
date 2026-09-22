from pathlib import Path

import numpy as np
import pandas as pd

from brain_session import ContinuousBrainSession
from lif_engine_realtime import RealtimeMaleCNSLIF
from motor_decoder import MotorDecoder


# ============================================================
# MALECNS DESCENDING NEURON -> MOTOR CAUSALITY TEST
#
# Purpose:
#   Directly stimulate LEFT and RIGHT copies of candidate
#   descending neuron types and inspect downstream motor output.
#
# This is a MODEL INTERVENTION.
#
# We are NOT claiming that direct Poisson stimulation represents
# natural biological firing.
#
# We want to learn:
#
#   DN LEFT stimulation  -> motor LEFT or RIGHT bias?
#   DN RIGHT stimulation -> mirrored motor bias?
#
# Only after that should a DN population be used as the live
# steering decoder.
# ============================================================


ROOT = Path(__file__).resolve().parent


ANNOTATION_FILE = (
    ROOT
    / "data"
    / "raw"
    / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


OUTPUT_DIR = (
    ROOT
    / "laterality_reports"
)


# ============================================================
# MODEL
# ============================================================

W_SYN = 0.110


# Direct DN stimulation is a diagnostic perturbation.
# One DN per anatomical side is common, therefore we use a
# longer moderate-rate pulse than the visual-source experiment.

DN_STIM_RATE_HZ = 60.0

BASELINE_MS = 20.0
STIM_MS = 120.0
POST_MS = 200.0


SEEDS = [
    101,
    202,
    303,
    404,
    505,
]


# ============================================================
# CANDIDATES
#
# Includes:
# - strong ipsilateral visual-response candidates
# - highly active candidates
# - several contralateral visual-response candidates
# ============================================================

CANDIDATE_TYPES = [
    "DNp04",
    "DNp02",
    "DNp11",
    "DNp03",
    "DNp01",
    "DNp103",
    "DNg40",
    "DNpe056",

    # Opposite-side visual response candidates
    "DNp34",
    "DNae004",
    "DNb09",
    "DNa04",
]


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

    if text in {
        "L",
        "R",
    }:
        return text

    return None


def asymmetry(
    left_spikes,
    right_spikes
):

    total = (
        left_spikes
        +
        right_spikes
    )

    if total <= 0:
        return 0.0

    # Negative:
    # more anatomical LEFT output
    #
    # Positive:
    # more anatomical RIGHT output

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


def count_spikes(
    spikes,
    mask
):

    return int(
        spikes[
            mask
        ].sum()
    )


# ============================================================
# START
# ============================================================

print("=" * 120)
print("MALECNS DESCENDING -> MOTOR CAUSALITY TEST")
print("=" * 120)


if not ANNOTATION_FILE.exists():

    raise FileNotFoundError(
        ANNOTATION_FILE
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


decoder = MotorDecoder()


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
print(
    "Loading anatomical metadata..."
)


annotations = pd.read_feather(

    ANNOTATION_FILE,

    columns=[
        "bodyId",
        "instance",
        "somaSide",
    ]

)


annotations = annotations.drop_duplicates(

    subset=[
        "bodyId"
    ],

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


meta[
    "instance_side"
] = meta[
    "instance"
].apply(
    side_from_instance
)


meta[
    "soma_side"
] = meta[
    "somaSide"
].apply(
    side_from_soma
)


meta[
    "side"
] = meta[
    "instance_side"
]


missing = (
    meta[
        "side"
    ].isna()
)


meta.loc[
    missing,
    "side"
] = meta.loc[
    missing,
    "soma_side"
]


meta[
    "type_clean"
] = (
    meta[
        "type"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# MOTOR MASKS
# ============================================================

motor_meta_mask = (

    meta[
        "superclass"
    ].isin(
        [
            "vnc_motor",
            "cb_motor",
        ]
    )

)


motor_left_mask = (

    motor_meta_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "L"
    )

).to_numpy()


motor_right_mask = (

    motor_meta_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "R"
    )

).to_numpy()


motor_type_upper = (

    meta[
        "type_clean"
    ]
    .str.upper()

)


flight_type_mask = (

    motor_type_upper.isin(
        decoder.flight_types
    )

)


flight_left_mask = (

    motor_meta_mask

    &

    flight_type_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "L"
    )

).to_numpy()


flight_right_mask = (

    motor_meta_mask

    &

    flight_type_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "R"
    )

).to_numpy()


other_left_mask = (

    motor_meta_mask

    &

    ~flight_type_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "L"
    )

).to_numpy()


other_right_mask = (

    motor_meta_mask

    &

    ~flight_type_mask

    &

    (
        meta[
            "side"
        ]
        ==
        "R"
    )

).to_numpy()


print()
print("=" * 120)
print("MOTOR OUTPUT POPULATIONS")
print("=" * 120)

print(
    "Motor LEFT :",
    int(
        np.count_nonzero(
            motor_left_mask
        )
    )
)

print(
    "Motor RIGHT:",
    int(
        np.count_nonzero(
            motor_right_mask
        )
    )
)

print(
    "Flight LEFT :",
    int(
        np.count_nonzero(
            flight_left_mask
        )
    )
)

print(
    "Flight RIGHT:",
    int(
        np.count_nonzero(
            flight_right_mask
        )
    )
)


# ============================================================
# BUILD DN SOURCE POOLS
# ============================================================

candidate_sources = {}


print()
print("=" * 120)
print("CANDIDATE DN SOURCE POOLS")
print("=" * 120)


for dn_type in CANDIDATE_TYPES:

    left_indices = (

        meta.loc[
            (
                meta[
                    "type_clean"
                ]
                ==
                dn_type
            )
            &
            (
                meta[
                    "side"
                ]
                ==
                "L"
            ),
            "neuron_idx"
        ]

        .to_numpy(
            dtype=np.int32
        )

    )


    right_indices = (

        meta.loc[
            (
                meta[
                    "type_clean"
                ]
                ==
                dn_type
            )
            &
            (
                meta[
                    "side"
                ]
                ==
                "R"
            ),
            "neuron_idx"
        ]

        .to_numpy(
            dtype=np.int32
        )

    )


    candidate_sources[
        dn_type
    ] = {

        "L":
            left_indices,

        "R":
            right_indices,

    }


    print(

        f"{dn_type:<10}"

        f" | LEFT={len(left_indices)}"

        f" | RIGHT={len(right_indices)}"

    )


# ============================================================
# ONE INTERVENTION
# ============================================================

def run_intervention(
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


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    session.step(

        0.0,

        BASELINE_MS

    )


    # --------------------------------------------------------
    # DIRECT DN PERTURBATION
    # --------------------------------------------------------

    stimulus = session.step(

        DN_STIM_RATE_HZ,

        STIM_MS

    )


    # --------------------------------------------------------
    # DOWNSTREAM PROPAGATION
    # --------------------------------------------------------

    post = session.step(

        0.0,

        POST_MS

    )


    spikes = (

        stimulus[
            "spike_counts"
        ].astype(
            np.int64
        )

        +

        post[
            "spike_counts"
        ].astype(
            np.int64
        )

    )


    # --------------------------------------------------------
    # MOTOR OUTPUT
    # --------------------------------------------------------

    motor_left = count_spikes(

        spikes,

        motor_left_mask

    )


    motor_right = count_spikes(

        spikes,

        motor_right_mask

    )


    flight_left = count_spikes(

        spikes,

        flight_left_mask

    )


    flight_right = count_spikes(

        spikes,

        flight_right_mask

    )


    other_left = count_spikes(

        spikes,

        other_left_mask

    )


    other_right = count_spikes(

        spikes,

        other_right_mask

    )


    movement = decoder.decode(

        neurons=
            brain.neurons,

        spike_counts=
            spikes,

        motor_mask=
            brain.motor_mask

    )


    return {

        "stimulus_events":
            int(
                stimulus[
                    "stimulus_events"
                ]
            ),

        "total_spikes":
            int(
                spikes.sum()
            ),

        "motor_left":
            motor_left,

        "motor_right":
            motor_right,

        "motor_total":
            motor_left
            +
            motor_right,

        "motor_asym":
            asymmetry(
                motor_left,
                motor_right
            ),

        "flight_left":
            flight_left,

        "flight_right":
            flight_right,

        "flight_total":
            flight_left
            +
            flight_right,

        "flight_asym":
            asymmetry(
                flight_left,
                flight_right
            ),

        "other_left":
            other_left,

        "other_right":
            other_right,

        "other_asym":
            asymmetry(
                other_left,
                other_right
            ),

        "action":
            movement[
                "action"
            ],

        "decoder_flight":
            movement[
                "flight_spikes"
            ],

        "decoder_jump":
            movement[
                "jump_spikes"
            ],

        "decoder_other":
            movement[
                "other_spikes"
            ],

    }


# ============================================================
# RUN ALL CANDIDATES
# ============================================================

rows = []


print()
print("=" * 120)
print("RUNNING LEFT / RIGHT DN INTERVENTIONS")
print("=" * 120)


for dn_type in CANDIDATE_TYPES:

    sources = candidate_sources[
        dn_type
    ]


    if (

        len(
            sources[
                "L"
            ]
        )
        ==
        0

        or

        len(
            sources[
                "R"
            ]
        )
        ==
        0

    ):

        print()
        print(
            dn_type,
            "SKIPPED - bilateral source pair missing"
        )

        continue


    print()
    print(
        "-" * 120
    )

    print(
        dn_type
    )

    print(
        "-" * 120
    )


    for seed in SEEDS:

        left_result = run_intervention(

            sources[
                "L"
            ],

            seed

        )


        right_result = run_intervention(

            sources[
                "R"
            ],

            seed

        )


        rows.append(
            {
                "type":
                    dn_type,

                "seed":
                    seed,

                "stim_side":
                    "L",

                **left_result,
            }
        )


        rows.append(
            {
                "type":
                    dn_type,

                "seed":
                    seed,

                "stim_side":
                    "R",

                **right_result,
            }
        )


        print(

            f"Seed {seed} "

            f"| L-DN: "

            f"motor "
            f"{left_result['motor_left']}/"
            f"{left_result['motor_right']} "

            f"asym="
            f"{left_result['motor_asym']:+.3f} "

            f"| R-DN: "

            f"motor "
            f"{right_result['motor_left']}/"
            f"{right_result['motor_right']} "

            f"asym="
            f"{right_result['motor_asym']:+.3f}"

        )


results = pd.DataFrame(
    rows
)


# ============================================================
# SUMMARIZE MIRRORING
# ============================================================

summary_rows = []


for dn_type in CANDIDATE_TYPES:

    candidate = results[
        results[
            "type"
        ]
        ==
        dn_type
    ]


    if candidate.empty:
        continue


    left = (

        candidate[
            candidate[
                "stim_side"
            ]
            ==
            "L"
        ]

        .set_index(
            "seed"
        )

    )


    right = (

        candidate[
            candidate[
                "stim_side"
            ]
            ==
            "R"
        ]

        .set_index(
            "seed"
        )

    )


    common_seeds = (

        left.index

        .intersection(
            right.index
        )

    )


    left = left.loc[
        common_seeds
    ]


    right = right.loc[
        common_seeds
    ]


    left_motor_asym = float(
        left[
            "motor_asym"
        ].mean()
    )


    right_motor_asym = float(
        right[
            "motor_asym"
        ].mean()
    )


    left_flight_asym = float(
        left[
            "flight_asym"
        ].mean()
    )


    right_flight_asym = float(
        right[
            "flight_asym"
        ].mean()
    )


    motor_shift = (

        right_motor_asym

        -

        left_motor_asym

    )


    flight_shift = (

        right_flight_asym

        -

        left_flight_asym

    )


    mirror_hits = []


    for seed in common_seeds:

        left_value = float(
            left.loc[
                seed,
                "motor_asym"
            ]
        )


        right_value = float(
            right.loc[
                seed,
                "motor_asym"
            ]
        )


        mirror = (

            (
                left_value < 0
                and
                right_value > 0
            )

            or

            (
                left_value > 0
                and
                right_value < 0
            )

        )


        mirror_hits.append(
            int(
                mirror
            )
        )


    mirror_consistency = (

        float(
            np.mean(
                mirror_hits
            )
        )

        if mirror_hits

        else 0.0

    )


    mean_motor_spikes = float(

        pd.concat(
            [
                left[
                    "motor_total"
                ],
                right[
                    "motor_total"
                ],
            ]
        ).mean()

    )


    mean_flight_spikes = float(

        pd.concat(
            [
                left[
                    "flight_total"
                ],
                right[
                    "flight_total"
                ],
            ]
        ).mean()

    )


    # --------------------------------------------------------
    # INTERPRETATION OF MEAN MOTOR LATERALITY
    # --------------------------------------------------------

    if (

        left_motor_asym < 0

        and

        right_motor_asym > 0

    ):

        mapping = (
            "SAME_SIDE"
        )


    elif (

        left_motor_asym > 0

        and

        right_motor_asym < 0

    ):

        mapping = (
            "CROSS_SIDE"
        )


    else:

        mapping = (
            "NOT_MIRRORED"
        )


    # Strong score requires:
    # - large left-vs-right motor shift
    # - repeatability across seeds
    # - actual motor activity

    motor_score = (

        abs(
            motor_shift
        )

        *

        mirror_consistency

        *

        np.log1p(
            mean_motor_spikes
        )

    )


    summary_rows.append(
        {
            "type":
                dn_type,

            "left_motor_asym":
                left_motor_asym,

            "right_motor_asym":
                right_motor_asym,

            "motor_shift":
                motor_shift,

            "mirror_consistency":
                mirror_consistency,

            "mapping":
                mapping,

            "mean_motor_spikes":
                mean_motor_spikes,

            "left_flight_asym":
                left_flight_asym,

            "right_flight_asym":
                right_flight_asym,

            "flight_shift":
                flight_shift,

            "mean_flight_spikes":
                mean_flight_spikes,

            "motor_score":
                motor_score,

        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(

    by=[
        "motor_score",
        "mirror_consistency",
        "mean_motor_spikes",
    ],

    ascending=[
        False,
        False,
        False,
    ]

).reset_index(
    drop=True
)


# ============================================================
# DISPLAY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


print()
print("=" * 120)
print("DN -> MOTOR LATERALITY SUMMARY")
print("=" * 120)


display_columns = [
    "type",
    "left_motor_asym",
    "right_motor_asym",
    "motor_shift",
    "mirror_consistency",
    "mapping",
    "mean_motor_spikes",
    "left_flight_asym",
    "right_flight_asym",
    "flight_shift",
    "mean_flight_spikes",
    "motor_score",
]


print()

print(

    summary[
        display_columns
    ].to_string(

        index=False,

        float_format=lambda x: f"{x:.3f}"

    )

)


# ============================================================
# ACTION COUNTS
# ============================================================

print()
print("=" * 120)
print("ACTIONS PER CANDIDATE / SIDE")
print("=" * 120)


action_counts = (

    results

    .groupby(
        [
            "type",
            "stim_side",
            "action",
        ]
    )

    .size()

    .rename(
        "count"
    )

    .reset_index()

)


print()

print(
    action_counts.to_string(
        index=False
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

    "dn_motor_causality_trials.csv"

)


summary_file = (

    OUTPUT_DIR

    /

    "dn_motor_causality_summary.csv"

)


results.to_csv(

    trials_file,

    index=False

)


summary.to_csv(

    summary_file,

    index=False

)


print()
print("=" * 120)
print("FILES SAVED")
print("=" * 120)

print(
    trials_file
)

print(
    summary_file
)


print()
print("=" * 120)
print("DONE")
print("=" * 120)

print(
    "Interpretation:"
)

print(
    "SAME_SIDE = left DN biases left motor and right DN biases right motor."
)

print(
    "CROSS_SIDE = left DN biases right motor and right DN biases left motor."
)

print(
    "NOT_MIRRORED = no clean bilateral motor steering pattern."
)

print()
print(
    "This is a modeled causal perturbation test."
)

print(
    "Do not treat a candidate as a biological steering command "
    "solely from this result."
)

print("=" * 120)