from pathlib import Path

import numpy as np
import pandas as pd

from brain_session import ContinuousBrainSession
from lif_engine_realtime import RealtimeMaleCNSLIF


# ============================================================
# MALECNS STEERING DN RESPONSE TEST
#
# Test edilen steering-related descending neurons:
#   DNa01
#   DNa02
#
# Visual input:
#   anatomical LEFT LPLC2 + LC4
#   anatomical RIGHT LPLC2 + LC4
#
# IMPORTANT:
# - MaleCNS connectivity gerçek connectome yapısından.
# - LIF dynamics / W_SYN / stimulus encoding modellenmiş.
# - Henüz body steering yapılmaz.
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
# MODEL
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


STEERING_TYPES = [
    "DNa01",
    "DNa02",
]


# ============================================================
# HELPERS
# ============================================================

def get_side_from_instance(value):

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith("_L"):
        return "L"

    if text.endswith("_R"):
        return "R"

    return None


def get_side_from_soma(value):

    if pd.isna(value):
        return None

    text = str(value).strip().upper()

    if text in {
        "L",
        "R",
    }:
        return text

    return None


def configure_sources(
    session,
    indices
):

    indices = np.asarray(
        indices,
        dtype=np.int32
    )


    mask = np.zeros(
        session.brain.N,
        dtype=bool
    )


    mask[
        indices
    ] = True


    session.source_indices = (
        np.ascontiguousarray(
            indices
        )
    )


    session.source_mask = (
        np.ascontiguousarray(
            mask
        )
    )


def asymmetry(
    left,
    right
):

    total = (
        left
        +
        right
    )


    if total == 0:
        return 0.0


    return (
        right
        -
        left
    ) / total


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 100)

print(
    "MALECNS STEERING DN RESPONSE TEST"
)

print("=" * 100)


for file in [
    ANNOTATION_FILE,
    LEFT_SOURCE_FILE,
    RIGHT_SOURCE_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            file
        )


left_sources = np.load(
    LEFT_SOURCE_FILE
).astype(
    np.int32
)


right_sources = np.load(
    RIGHT_SOURCE_FILE
).astype(
    np.int32
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
print(
    "Loading MaleCNS..."
)


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
# ANNOTATIONS
# ============================================================

annotations = pd.read_feather(

    ANNOTATION_FILE,

    columns=[
        "bodyId",
        "instance",
        "somaSide",
    ]
)


annotations = (
    annotations
    .drop_duplicates(
        subset=[
            "bodyId"
        ],
        keep="first"
    )
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
    get_side_from_instance
)


meta[
    "soma_side"
] = meta[
    "somaSide"
].apply(
    get_side_from_soma
)


meta[
    "side"
] = meta[
    "instance_side"
]


missing = (
    meta["side"]
    .isna()
)


meta.loc[
    missing,
    "side"
] = meta.loc[
    missing,
    "soma_side"
]


# ============================================================
# CREATE STEERING MASKS
# ============================================================

masks = {}


print()
print("=" * 100)
print("STEERING DN POPULATIONS")
print("=" * 100)


for dn_type in STEERING_TYPES:

    left_mask = (

        (
            meta["type"]
            ==
            dn_type
        )

        &

        (
            meta["side"]
            ==
            "L"
        )

    ).to_numpy()


    right_mask = (

        (
            meta["type"]
            ==
            dn_type
        )

        &

        (
            meta["side"]
            ==
            "R"
        )

    ).to_numpy()


    masks[
        dn_type
    ] = {

        "L":
            left_mask,

        "R":
            right_mask,

    }


    print()

    print(
        dn_type
    )

    print(
        " LEFT neurons :",
        int(
            np.count_nonzero(
                left_mask
            )
        )
    )

    print(
        " RIGHT neurons:",
        int(
            np.count_nonzero(
                right_mask
            )
        )
    )


# ============================================================
# RUN ONE CONDITION
# ============================================================

def run_condition(
    source_indices,
    condition,
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
    # VISUAL STIMULUS
    # --------------------------------------------------------

    stimulus = session.step(
        STIM_RATE_HZ,
        STIM_MS
    )


    # --------------------------------------------------------
    # POST-STIMULUS PROPAGATION
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


    row = {

        "condition":
            condition,

        "seed":
            seed,

        "stimulus_events":
            stimulus[
                "stimulus_events"
            ],

    }


    # --------------------------------------------------------
    # DNa01 / DNa02
    # --------------------------------------------------------

    for dn_type in STEERING_TYPES:

        left_spikes = int(

            spikes[
                masks[
                    dn_type
                ][
                    "L"
                ]
            ].sum()

        )


        right_spikes = int(

            spikes[
                masks[
                    dn_type
                ][
                    "R"
                ]
            ].sum()

        )


        row[
            f"{dn_type}_L"
        ] = left_spikes


        row[
            f"{dn_type}_R"
        ] = right_spikes


        row[
            f"{dn_type}_asym"
        ] = asymmetry(

            left_spikes,
            right_spikes

        )


    # --------------------------------------------------------
    # COMBINED DNa01 + DNa02
    # --------------------------------------------------------

    combined_left = sum(

        row[
            f"{dn_type}_L"
        ]

        for dn_type in STEERING_TYPES

    )


    combined_right = sum(

        row[
            f"{dn_type}_R"
        ]

        for dn_type in STEERING_TYPES

    )


    row[
        "combined_L"
    ] = combined_left


    row[
        "combined_R"
    ] = combined_right


    row[
        "combined_asym"
    ] = asymmetry(

        combined_left,
        combined_right

    )


    return row


# ============================================================
# RUN TEST
# ============================================================

rows = []


print()
print("=" * 100)
print("MATCHED LEFT / RIGHT VISUAL TEST")
print("=" * 100)


for seed in SEEDS:

    left = run_condition(

        left_sources,

        "VISUAL_LEFT",

        seed

    )


    right = run_condition(

        right_sources,

        "VISUAL_RIGHT",

        seed

    )


    rows.append(
        left
    )

    rows.append(
        right
    )


    print()
    print(
        "Seed",
        seed
    )


    print(

        " LEFT "
        f"| DNa01 "
        f"{left['DNa01_L']}/{left['DNa01_R']} "

        f"| DNa02 "
        f"{left['DNa02_L']}/{left['DNa02_R']} "

        f"| combined "
        f"{left['combined_L']}/{left['combined_R']} "

        f"| asym="
        f"{left['combined_asym']:+.3f}"

    )


    print(

        " RIGHT"
        f" | DNa01 "
        f"{right['DNa01_L']}/{right['DNa01_R']} "

        f"| DNa02 "
        f"{right['DNa02_L']}/{right['DNa02_R']} "

        f"| combined "
        f"{right['combined_L']}/{right['combined_R']} "

        f"| asym="
        f"{right['combined_asym']:+.3f}"

    )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)


print()
print("=" * 100)
print("AVERAGE STEERING DN RESPONSE")
print("=" * 100)


for condition in [
    "VISUAL_LEFT",
    "VISUAL_RIGHT",
]:

    subset = results[
        results[
            "condition"
        ]
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


    for dn_type in STEERING_TYPES:

        print(

            f"{dn_type} mean L/R: "

            f"{subset[f'{dn_type}_L'].mean():.2f}"

            " / "

            f"{subset[f'{dn_type}_R'].mean():.2f}"

        )


        print(

            f"{dn_type} mean asym (+R,-L): "

            f"{subset[f'{dn_type}_asym'].mean():+.4f}"

        )


    print()

    print(

        "COMBINED mean L/R: "

        f"{subset['combined_L'].mean():.2f}"

        " / "

        f"{subset['combined_R'].mean():.2f}"

    )


    print(

        "COMBINED asym (+R,-L): "

        f"{subset['combined_asym'].mean():+.4f}"

    )


# ============================================================
# PAIRED SHIFT
# ============================================================

left = (

    results[
        results["condition"]
        ==
        "VISUAL_LEFT"
    ]

    .set_index(
        "seed"
    )

)


right = (

    results[
        results["condition"]
        ==
        "VISUAL_RIGHT"
    ]

    .set_index(
        "seed"
    )

)


paired = pd.DataFrame(
    index=SEEDS
)


for dn_type in STEERING_TYPES:

    paired[
        f"{dn_type}_left_visual"
    ] = left[
        f"{dn_type}_asym"
    ]


    paired[
        f"{dn_type}_right_visual"
    ] = right[
        f"{dn_type}_asym"
    ]


    paired[
        f"{dn_type}_shift"
    ] = (

        paired[
            f"{dn_type}_right_visual"
        ]

        -

        paired[
            f"{dn_type}_left_visual"
        ]

    )


paired[
    "combined_left_visual"
] = left[
    "combined_asym"
]


paired[
    "combined_right_visual"
] = right[
    "combined_asym"
]


paired[
    "combined_shift"
] = (

    paired[
        "combined_right_visual"
    ]

    -

    paired[
        "combined_left_visual"
    ]

)


print()
print("=" * 100)
print("PAIRED STEERING SHIFT")
print("=" * 100)

print()
print(
    paired.to_string()
)


print()

print(
    "Mean combined steering shift:",
    f"{paired['combined_shift'].mean():+.4f}"
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

    "steering_dn_response_trials.csv"

)


paired_file = (

    output_dir

    /

    "steering_dn_response_paired.csv"

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


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "If LEFT and RIGHT visual stimulation produce "
    "opposite DNa01/DNa02 asymmetry, "
    "we have a biologically supported steering channel."
)

print("=" * 100)