import time

import numpy as np

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session_multi import MultiInputBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# MODEL
# ============================================================

W_SYN = 0.110

CHUNK_MS = 20.0

TEST_BRAIN_MS = 10000.0

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# CONNECTOME-GUIDED ENDOGENOUS DRIVE
# ============================================================

DRIVE_NEURON_COUNT = 128


DRIVE_RATES_HZ = [

    0.10,
    0.25,
    0.50,
    1.00,
    2.00,

]


# ============================================================
# HIGH ACTIVITY SAFETY
# ============================================================

HIGH_STATE_SPIKES = 1500

HIGH_STATE_CONSECUTIVE_CHUNKS = 25


# ============================================================
# BRAIN
# ============================================================

print("=" * 110)
print("MALECNS CONNECTOME-GUIDED PREMOTOR DRIVE SWEEP")
print("=" * 110)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


neurons = brain.neurons


print()

print(
    "Nöron:",
    brain.N
)

print(
    "Bağlantı:",
    len(
        brain.indices
    )
)

print(
    "W_SYN:",
    brain.w_syn
)


# ============================================================
# CONNECTIVITY
# ============================================================

out_degree = np.diff(
    brain.indptr
).astype(
    np.int32
)


in_degree = np.bincount(

    brain.indices,

    minlength=brain.N

).astype(
    np.int32
)


# ============================================================
# BASE POPULATION
#
# Sadece:
#
# cb_intrinsic
# fast_excitatory
#
# Motor değil.
# Descending değil.
# Sensory değil.
# Visual projection değil.
# ============================================================

superclass = (

    neurons["superclass"]

    .fillna("")

    .astype(str)

    .to_numpy()

)


nt_role = (

    neurons["nt_role"]

    .fillna("")

    .astype(str)

    .to_numpy()

)


base_mask = (

    (
        superclass
        ==
        "cb_intrinsic"
    )

    &

    (
        nt_role
        ==
        "fast_excitatory"
    )

    &

    (
        ~brain.motor_mask
    )

    &

    (
        ~brain.descending_mask
    )

)


base_indices = np.flatnonzero(
    base_mask
)


print("\n" + "=" * 110)
print("BASE CENTRAL POPULATION")
print("=" * 110)


print(
    "cb_intrinsic + fast_excitatory:",
    len(
        base_indices
    )
)


# ============================================================
# HUB FİLTRESİ
#
# Dev hub'ları kullanmıyoruz.
# Ama önceki testteki kadar dar p50-p75 de kullanmıyoruz.
#
# p25 - p90
# ============================================================

base_out = out_degree[
    base_indices
]


out_low = float(

    np.percentile(
        base_out,
        25
    )

)


out_high = float(

    np.percentile(
        base_out,
        90
    )

)


print(
    "Allowed out-degree:",
    round(
        out_low,
        2
    ),
    "-",
    round(
        out_high,
        2
    )
)


# ============================================================
# HER CENTRAL NÖRONUN DESCENDING BAĞLANTISINI ÖLÇ
#
# Mevcut matrix:
#
# indptr[presynaptic]
# indices = postsynaptic targets
#
# Burada behavior seçmiyoruz.
#
# Connectome'da gerçekten descending neuronlara outgoing
# bağlantısı bulunan central neurons aranıyor.
# ============================================================

direct_dn_strength = np.zeros(

    brain.N,

    dtype=np.float64

)


direct_dn_targets = np.zeros(

    brain.N,

    dtype=np.int32

)


print("\nDescending bağlantıları taranıyor...")


for neuron_idx in base_indices:

    # --------------------------------------------------------
    # HUB FILTER
    # --------------------------------------------------------

    degree = out_degree[
        neuron_idx
    ]


    if (
        degree < out_low
        or
        degree > out_high
    ):

        continue


    start = brain.indptr[
        neuron_idx
    ]


    end = brain.indptr[
        neuron_idx + 1
    ]


    if end <= start:

        continue


    targets = brain.indices[
        start:end
    ]


    weights = brain.weights[
        start:end
    ]


    # --------------------------------------------------------
    # DESCENDING POSTSYNAPTIC TARGETS
    # --------------------------------------------------------

    dn_mask = brain.descending_mask[
        targets
    ]


    # --------------------------------------------------------
    # Sadece pozitif modeled synaptic influence
    # --------------------------------------------------------

    positive_mask = (

        dn_mask

        &

        (
            weights > 0
        )

    )


    if not np.any(
        positive_mask
    ):

        continue


    direct_dn_targets[
        neuron_idx
    ] = int(

        np.count_nonzero(
            positive_mask
        )

    )


    direct_dn_strength[
        neuron_idx
    ] = float(

        np.sum(

            weights[
                positive_mask
            ]

        )

    )


# ============================================================
# DIRECT DN CONNECTED CENTRAL NEURONS
# ============================================================

dn_connected_mask = (

    base_mask

    &

    (
        direct_dn_targets
        >
        0
    )

    &

    (
        out_degree
        >=
        out_low
    )

    &

    (
        out_degree
        <=
        out_high
    )

)


dn_connected_indices = np.flatnonzero(
    dn_connected_mask
)


print(
    "Direct DN-connected central neuron:",
    len(
        dn_connected_indices
    )
)


if len(
    dn_connected_indices
) == 0:

    raise RuntimeError(
        "Direct DN bağlantılı candidate bulunamadı."
    )


# ============================================================
# SCORE
#
# DN strength yüksek olsun.
#
# Ama sadece en büyük out-degree hub kazanmasın.
#
# score =
# direct DN strength / sqrt(out-degree)
# ============================================================

scores = np.zeros(

    brain.N,

    dtype=np.float64

)


scores[
    dn_connected_indices
] = (

    direct_dn_strength[
        dn_connected_indices
    ]

    /

    np.sqrt(

        np.maximum(

            out_degree[
                dn_connected_indices
            ],

            1

        )

    )

)


# ============================================================
# EN İYİ CONNECTOME-GUIDED CENTRAL POPULATION
# ============================================================

rank_order = np.argsort(

    scores[
        dn_connected_indices
    ]

)[::-1]


ranked_indices = dn_connected_indices[
    rank_order
]


take_count = min(

    DRIVE_NEURON_COUNT,

    len(
        ranked_indices
    )

)


drive_indices = ranked_indices[
    :take_count
].astype(
    np.int32
)


print(
    "Selected drive neuron:",
    len(
        drive_indices
    )
)


print(
    "Mean out-degree:",
    round(
        float(
            out_degree[
                drive_indices
            ].mean()
        ),
        2
    )
)


print(
    "Mean direct DN targets:",
    round(
        float(
            direct_dn_targets[
                drive_indices
            ].mean()
        ),
        2
    )
)


print(
    "Mean direct DN strength:",
    round(
        float(
            direct_dn_strength[
                drive_indices
            ].mean()
        ),
        2
    )
)


# ============================================================
# TOP SELECTED TYPES
# ============================================================

selected_types = (

    neurons.iloc[
        drive_indices
    ]["type"]

    .fillna("")

    .astype(str)

    .replace(
        "",
        "<EMPTY>"
    )

    .value_counts()

)


print("\nTop selected types:")

print(
    selected_types.head(
        30
    ).to_string()
)


# ============================================================
# TOP 20 DETAIL
# ============================================================

print("\n" + "=" * 110)
print("TOP 20 CONNECTOME-GUIDED CENTRAL NEURON")
print("=" * 110)


for rank, neuron_idx in enumerate(
    drive_indices[:20],
    start=1
):

    neuron = neurons.iloc[
        neuron_idx
    ]


    print(

        f"{rank:2d}"

        f" | idx {neuron_idx:6d}"

        f" | body {str(neuron['bodyId']):>10}"

        f" | type {str(neuron['type']):20s}"

        f" | out {out_degree[neuron_idx]:4d}"

        f" | DNtargets {direct_dn_targets[neuron_idx]:3d}"

        f" | DNstrength {direct_dn_strength[neuron_idx]:8.2f}"

        f" | score {scores[neuron_idx]:8.3f}"

    )


# ============================================================
# WARMUP
# ============================================================

warmup = MultiInputBrainSession(

    brain,

    input_groups={

        "endogenous":
            drive_indices

    },

    seed=999

)


warmup.step(

    group_rates={

        "endogenous":
            0.0

    },

    chunk_ms=1.0

)


print(
    "\nNumba warmup tamam."
)


# ============================================================
# TEK RATE TEST
# ============================================================

def run_rate(
    rate_hz
):

    session = MultiInputBrainSession(

        brain,

        input_groups={

            "endogenous":
                drive_indices

        },

        seed=SEED

    )


    # ========================================================
    # ACTION
    # ========================================================

    idle_chunks = 0

    move_chunks = 0

    fly_chunks = 0

    jump_chunks = 0


    active_dn_chunks = 0

    active_motor_chunks = 0


    # ========================================================
    # TOTALS
    # ========================================================

    total_events = 0

    total_spikes = 0

    total_dn = 0

    total_motor = 0


    first_dn_ms = None

    first_motor_ms = None

    first_action_ms = None


    peak_spikes = 0


    # ========================================================
    # PERFORMANCE
    # ========================================================

    compute_sum_ms = 0.0

    compute_max_ms = 0.0

    over_20ms = 0


    # ========================================================
    # SAFETY
    # ========================================================

    high_state_counter = 0

    high_state = False

    runaway = False


    # ========================================================
    # LATE WINDOW
    # ========================================================

    late_start_ms = (

        TEST_BRAIN_MS

        -

        2000.0

    )


    late_chunks = 0

    late_spikes = 0

    late_dn = 0

    late_motor = 0


    chunk_count = int(

        TEST_BRAIN_MS

        /

        CHUNK_MS

    )


    # ========================================================
    # LOOP
    # ========================================================

    for chunk in range(
        chunk_count
    ):

        brain_time_ms = (

            chunk

            *

            CHUNK_MS

        )


        compute_start = (
            time.perf_counter()
        )


        result = session.step(

            group_rates={

                "endogenous":
                    rate_hz

            },

            chunk_ms=CHUNK_MS

        )


        compute_ms = (

            time.perf_counter()

            -

            compute_start

        ) * 1000.0


        compute_sum_ms += (
            compute_ms
        )


        compute_max_ms = max(

            compute_max_ms,

            compute_ms

        )


        if compute_ms > CHUNK_MS:

            over_20ms += 1


        # ====================================================
        # NETWORK VALUES
        # ====================================================

        spikes = result[
            "total_spikes"
        ]


        dn = result[
            "descending_spikes"
        ]


        motor = result[
            "motor_spikes"
        ]


        events = (

            result[
                "stimulus_by_group"
            ].get(
                "endogenous",
                0
            )

        )


        total_events += events

        total_spikes += spikes

        total_dn += dn

        total_motor += motor


        peak_spikes = max(

            peak_spikes,

            spikes

        )


        if dn > 0:

            active_dn_chunks += 1


        if motor > 0:

            active_motor_chunks += 1


        # ====================================================
        # FIRST DN
        # ====================================================

        if (

            dn > 0

            and

            first_dn_ms is None

        ):

            first_dn_ms = (
                brain_time_ms
            )


        # ====================================================
        # FIRST MOTOR
        # ====================================================

        if (

            motor > 0

            and

            first_motor_ms is None

        ):

            first_motor_ms = (
                brain_time_ms
            )


        # ====================================================
        # DECODER
        # ====================================================

        movement = decoder.decode(

            neurons=
                brain.neurons,

            spike_counts=
                result[
                    "spike_counts"
                ],

            motor_mask=
                brain.motor_mask

        )


        action = movement[
            "action"
        ]


        if action == "IDLE":

            idle_chunks += 1


        elif action == "MOVE":

            move_chunks += 1


        elif action == "FLY":

            fly_chunks += 1


        elif action == "JUMP":

            jump_chunks += 1


        if (

            action != "IDLE"

            and

            first_action_ms is None

        ):

            first_action_ms = (
                brain_time_ms
            )


        # ====================================================
        # LATE WINDOW
        # ====================================================

        if brain_time_ms >= late_start_ms:

            late_chunks += 1

            late_spikes += spikes

            late_dn += dn

            late_motor += motor


        # ====================================================
        # HIGH STATE
        # ====================================================

        if spikes >= HIGH_STATE_SPIKES:

            high_state_counter += 1

        else:

            high_state_counter = 0


        if (

            high_state_counter

            >=

            HIGH_STATE_CONSECUTIVE_CHUNKS

        ):

            high_state = True

            break


        # ====================================================
        # RUNAWAY
        # ====================================================

        if spikes > RUNAWAY_LIMIT:

            runaway = True

            break


    # ========================================================
    # SUMMARY
    # ========================================================

    completed_chunks = (

        idle_chunks

        +

        move_chunks

        +

        fly_chunks

        +

        jump_chunks

    )


    completed_ms = (

        completed_chunks

        *

        CHUNK_MS

    )


    if completed_chunks > 0:

        mean_compute_ms = (

            compute_sum_ms

            /

            completed_chunks

        )

    else:

        mean_compute_ms = 0.0


    if late_chunks > 0:

        late_mean_spikes = (

            late_spikes

            /

            late_chunks

        )


        late_mean_dn = (

            late_dn

            /

            late_chunks

        )


        late_mean_motor = (

            late_motor

            /

            late_chunks

        )

    else:

        late_mean_spikes = 0.0

        late_mean_dn = 0.0

        late_mean_motor = 0.0


    return {

        "rate":
            rate_hz,

        "events":
            total_events,

        "spikes":
            total_spikes,

        "dn":
            total_dn,

        "dn_chunks":
            active_dn_chunks,

        "motor":
            total_motor,

        "motor_chunks":
            active_motor_chunks,

        "idle":
            idle_chunks,

        "move":
            move_chunks,

        "fly":
            fly_chunks,

        "jump":
            jump_chunks,

        "first_dn":
            first_dn_ms,

        "first_motor":
            first_motor_ms,

        "first_action":
            first_action_ms,

        "peak":
            peak_spikes,

        "late_spikes":
            late_mean_spikes,

        "late_dn":
            late_mean_dn,

        "late_motor":
            late_mean_motor,

        "mean_compute":
            mean_compute_ms,

        "max_compute":
            compute_max_ms,

        "over20":
            over_20ms,

        "high_state":
            high_state,

        "runaway":
            runaway,

        "completed_ms":
            completed_ms,

    }


# ============================================================
# SWEEP
# ============================================================

results = []


print("\n" + "=" * 110)
print("CONNECTOME-GUIDED RATE SWEEP")
print("=" * 110)


for rate in DRIVE_RATES_HZ:

    print()

    print(
        f"Drive {rate:.2f} Hz çalışıyor..."
    )


    result = run_rate(
        rate
    )


    results.append(
        result
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 205)
print("PREMOTOR DRIVE SONUCU")
print("=" * 205)


print(

    f"{'Rate':>6}"

    f" {'Events':>7}"

    f" {'DN':>7}"

    f" {'DNChk':>6}"

    f" {'Motor':>7}"

    f" {'MotChk':>6}"

    f" {'IDLE':>6}"

    f" {'MOVE':>6}"

    f" {'FLY':>5}"

    f" {'JUMP':>6}"

    f" {'FirstDN':>8}"

    f" {'FirstMot':>9}"

    f" {'FirstAct':>9}"

    f" {'Peak':>7}"

    f" {'LateSpk':>9}"

    f" {'LateDN':>8}"

    f" {'LateMot':>8}"

    f" {'MeanMS':>8}"

    f" {'MaxMS':>8}"

    f" {'>20ms':>7}"

    f" {'HighState':>10}"

    f" {'Runaway':>9}"

    f" {'DoneMS':>8}"

)


for r in results:

    def fmt_time(
        value
    ):

        if value is None:

            return "-"

        return f"{value:.0f}"


    print(

        f"{r['rate']:>6.2f}"

        f" {r['events']:>7}"

        f" {r['dn']:>7}"

        f" {r['dn_chunks']:>6}"

        f" {r['motor']:>7}"

        f" {r['motor_chunks']:>6}"

        f" {r['idle']:>6}"

        f" {r['move']:>6}"

        f" {r['fly']:>5}"

        f" {r['jump']:>6}"

        f" {fmt_time(r['first_dn']):>8}"

        f" {fmt_time(r['first_motor']):>9}"

        f" {fmt_time(r['first_action']):>9}"

        f" {r['peak']:>7}"

        f" {r['late_spikes']:>9.1f}"

        f" {r['late_dn']:>8.2f}"

        f" {r['late_motor']:>8.2f}"

        f" {r['mean_compute']:>8.2f}"

        f" {r['max_compute']:>8.2f}"

        f" {r['over20']:>7}"

        f" {str(r['high_state']):>10}"

        f" {str(r['runaway']):>9}"

        f" {r['completed_ms']:>8.0f}"

    )


print("\n" + "=" * 205)
print("TEST BİTTİ")
print("=" * 205)