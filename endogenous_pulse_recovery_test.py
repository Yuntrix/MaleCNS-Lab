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

TOTAL_BRAIN_MS = 10000.0

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# ENDOGENOUS BURST
#
# 2 saniye tamamen sessiz
# sonra kısa internal burst
# sonra hiçbir input yok
#
# Böylece brain kendi kendine recovery yapıyor mu göreceğiz.
# ============================================================

BURST_START_MS = 2000.0

BURST_DURATION_MS = 200.0

BURST_END_MS = (
    BURST_START_MS
    +
    BURST_DURATION_MS
)


# ============================================================
# AYNI PREMOTOR POPULATION
# ============================================================

DRIVE_NEURON_COUNT = 128


# ============================================================
# BURST RATE ADAYLARI
#
# Sürekli drive değil.
# Sadece 200 ms boyunca.
# ============================================================

BURST_RATES_HZ = [
    1.0,
    2.0,
    5.0,
    10.0,
]


# ============================================================
# HIGH STATE SAFETY
# ============================================================

HIGH_STATE_SPIKES = 1500

HIGH_STATE_CONSECUTIVE_CHUNKS = 25


# ============================================================
# BRAIN
# ============================================================

print("=" * 110)
print("MALECNS ENDOGENOUS BURST + RECOVERY TEST")
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


# ============================================================
# BASE POPULATION
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


# ============================================================
# HUB FILTER
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


# ============================================================
# DIRECT DN CONNECTIVITY
# ============================================================

direct_dn_strength = np.zeros(

    brain.N,

    dtype=np.float64

)


direct_dn_targets = np.zeros(

    brain.N,

    dtype=np.int32

)


print("\nConnectome-guided drive population hazırlanıyor...")


for neuron_idx in base_indices:

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


    dn_target_mask = (

        brain.descending_mask[
            targets
        ]

        &

        (
            weights > 0
        )

    )


    if not np.any(
        dn_target_mask
    ):

        continue


    direct_dn_targets[
        neuron_idx
    ] = int(

        np.count_nonzero(
            dn_target_mask
        )

    )


    direct_dn_strength[
        neuron_idx
    ] = float(

        np.sum(

            weights[
                dn_target_mask
            ]

        )

    )


# ============================================================
# CANDIDATES
# ============================================================

candidate_mask = (

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


candidate_indices = np.flatnonzero(
    candidate_mask
)


# ============================================================
# SCORE
# ============================================================

scores = np.zeros(

    brain.N,

    dtype=np.float64

)


scores[
    candidate_indices
] = (

    direct_dn_strength[
        candidate_indices
    ]

    /

    np.sqrt(

        np.maximum(

            out_degree[
                candidate_indices
            ],

            1

        )

    )

)


rank_order = np.argsort(

    scores[
        candidate_indices
    ]

)[::-1]


ranked_indices = candidate_indices[
    rank_order
]


drive_indices = ranked_indices[
    :DRIVE_NEURON_COUNT
].astype(
    np.int32
)


print(
    "Drive neuron:",
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
    "Mean DN targets:",
    round(
        float(
            direct_dn_targets[
                drive_indices
            ].mean()
        ),
        2
    )
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

        "endogenous": 0.0

    },

    chunk_ms=1.0

)


print(
    "Numba warmup tamam."
)


# ============================================================
# TEK BURST TESTİ
# ============================================================

def run_test(
    burst_rate_hz
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
    # COUNTERS
    # ========================================================

    total_events = 0

    total_spikes = 0

    total_dn = 0

    total_motor = 0


    idle_chunks = 0

    move_chunks = 0

    fly_chunks = 0

    jump_chunks = 0


    # ========================================================
    # RESPONSE WINDOW
    #
    # Burst + ilk 2 saniye sonrası
    # ========================================================

    response_move = 0

    response_motor = 0

    response_dn = 0


    # ========================================================
    # LATE RECOVERY
    #
    # Son 2 saniye: 8-10s
    # ========================================================

    late_spikes = 0

    late_dn = 0

    late_motor = 0

    late_move = 0

    late_chunks = 0


    first_motor_ms = None

    last_motor_ms = None


    peak_spikes = 0


    # ========================================================
    # PERFORMANCE
    # ========================================================

    compute_sum_ms = 0.0

    compute_max_ms = 0.0

    over20 = 0


    # ========================================================
    # SAFETY
    # ========================================================

    high_counter = 0

    high_state = False

    runaway = False


    chunk_count = int(

        TOTAL_BRAIN_MS

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


        # ====================================================
        # BURST RATE
        # ====================================================

        if (

            BURST_START_MS
            <=
            brain_time_ms
            <
            BURST_END_MS

        ):

            current_rate = (
                burst_rate_hz
            )

        else:

            current_rate = 0.0


        # ====================================================
        # SIMULATION
        # ====================================================

        start_time = (
            time.perf_counter()
        )


        result = session.step(

            group_rates={

                "endogenous":
                    current_rate

            },

            chunk_ms=CHUNK_MS

        )


        compute_ms = (

            time.perf_counter()

            -

            start_time

        ) * 1000.0


        compute_sum_ms += (
            compute_ms
        )


        compute_max_ms = max(

            compute_max_ms,

            compute_ms

        )


        if compute_ms > CHUNK_MS:

            over20 += 1


        # ====================================================
        # RAW VALUES
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


        # ====================================================
        # MOTOR TIMING
        # ====================================================

        if motor > 0:

            if first_motor_ms is None:

                first_motor_ms = (
                    brain_time_ms
                )


            last_motor_ms = (
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


        # ====================================================
        # RESPONSE WINDOW
        #
        # 2.0s - 4.0s
        # ====================================================

        if (

            BURST_START_MS
            <=
            brain_time_ms
            <
            4000.0

        ):

            response_dn += dn

            response_motor += motor


            if action == "MOVE":

                response_move += 1


        # ====================================================
        # LATE RECOVERY
        #
        # 8.0s - 10.0s
        # ====================================================

        if brain_time_ms >= 8000.0:

            late_chunks += 1

            late_spikes += spikes

            late_dn += dn

            late_motor += motor


            if action == "MOVE":

                late_move += 1


        # ====================================================
        # HIGH STATE
        # ====================================================

        if spikes >= HIGH_STATE_SPIKES:

            high_counter += 1

        else:

            high_counter = 0


        if (

            high_counter

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


    if completed_chunks > 0:

        mean_compute = (

            compute_sum_ms

            /

            completed_chunks

        )

    else:

        mean_compute = 0.0


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
            burst_rate_hz,

        "events":
            total_events,

        "idle":
            idle_chunks,

        "move":
            move_chunks,

        "fly":
            fly_chunks,

        "jump":
            jump_chunks,

        "response_move":
            response_move,

        "response_dn":
            response_dn,

        "response_motor":
            response_motor,

        "first_motor":
            first_motor_ms,

        "last_motor":
            last_motor_ms,

        "late_spikes":
            late_mean_spikes,

        "late_dn":
            late_mean_dn,

        "late_motor":
            late_mean_motor,

        "late_move":
            late_move,

        "peak":
            peak_spikes,

        "mean_ms":
            mean_compute,

        "max_ms":
            compute_max_ms,

        "over20":
            over20,

        "high_state":
            high_state,

        "runaway":
            runaway,

        "done_ms":
            completed_chunks
            *
            CHUNK_MS,

    }


# ============================================================
# TESTLER
# ============================================================

results = []


print("\n" + "=" * 110)
print("BURST TESTLERİ")
print("=" * 110)


for rate in BURST_RATES_HZ:

    print()

    print(
        f"{rate:.1f} Hz x "
        f"{BURST_DURATION_MS:.0f} ms çalışıyor..."
    )


    result = run_test(
        rate
    )


    results.append(
        result
    )


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 205)
print("BURST + RECOVERY SONUCU")
print("=" * 205)


print(

    f"{'Rate':>6}"

    f" {'Events':>7}"

    f" {'RespDN':>8}"

    f" {'RespMot':>8}"

    f" {'RespMove':>9}"

    f" {'FirstMot':>9}"

    f" {'LastMot':>9}"

    f" {'LateSpk':>9}"

    f" {'LateDN':>8}"

    f" {'LateMot':>8}"

    f" {'LateMove':>9}"

    f" {'IDLE':>6}"

    f" {'MOVE':>6}"

    f" {'FLY':>5}"

    f" {'Peak':>7}"

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

        f"{r['rate']:>6.1f}"

        f" {r['events']:>7}"

        f" {r['response_dn']:>8}"

        f" {r['response_motor']:>8}"

        f" {r['response_move']:>9}"

        f" {fmt_time(r['first_motor']):>9}"

        f" {fmt_time(r['last_motor']):>9}"

        f" {r['late_spikes']:>9.1f}"

        f" {r['late_dn']:>8.2f}"

        f" {r['late_motor']:>8.2f}"

        f" {r['late_move']:>9}"

        f" {r['idle']:>6}"

        f" {r['move']:>6}"

        f" {r['fly']:>5}"

        f" {r['peak']:>7}"

        f" {r['mean_ms']:>8.2f}"

        f" {r['max_ms']:>8.2f}"

        f" {r['over20']:>7}"

        f" {str(r['high_state']):>10}"

        f" {str(r['runaway']):>9}"

        f" {r['done_ms']:>8.0f}"

    )


print("\n" + "=" * 205)
print("TEST BİTTİ")
print("=" * 205)