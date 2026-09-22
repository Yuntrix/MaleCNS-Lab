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

TEST_BRAIN_MS = 8000.0

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# BACKGROUND ADAYLARI
# ============================================================

BACKGROUND_RATES = [
    0.05,
    0.10,
    0.15,
]


# ============================================================
# LOOMING
# ============================================================

LOOMING_RATE_HZ = 40.0

LOOMING_START_MS = 2000.0

LOOMING_END_MS = 2060.0


# ============================================================
# BACKGROUND POPULATION
# ============================================================

PER_GROUP = 200


BACKGROUND_SUPERCLASSES = [

    "ol_sensory",

    "cb_sensory",

    "vnc_sensory",

]


# ============================================================
# BRAIN
# ============================================================

print("=" * 90)
print("MALECNS LOW-LOAD AUTONOMOUS CALIBRATION")
print("=" * 90)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


# ============================================================
# OUT DEGREE
# ============================================================

out_degree = np.diff(
    brain.indptr
)


# ============================================================
# BACKGROUND NÖRONLARI
# ============================================================

background_indices = []


for superclass in BACKGROUND_SUPERCLASSES:

    candidates = np.flatnonzero(

        (
            brain.superclasses
            ==
            superclass
        ).to_numpy()

    )


    degree = out_degree[
        candidates
    ]


    order = np.argsort(
        degree
    )[::-1]


    chosen = candidates[
        order[:PER_GROUP]
    ]


    background_indices.extend(
        chosen.tolist()
    )


background_indices = np.asarray(

    background_indices,

    dtype=np.int32

)


# ============================================================
# LOOMING NÖRONLARI
# ============================================================

looming_indices = np.flatnonzero(

    brain.types.isin(

        [
            "LPLC2",
            "LC4"
        ]

    ).to_numpy()

).astype(
    np.int32
)


print(
    "Background neuron:",
    len(
        background_indices
    )
)


print(
    "Looming neuron:",
    len(
        looming_indices
    )
)


# ============================================================
# WARMUP
# ============================================================

warmup = MultiInputBrainSession(

    brain,

    input_groups={

        "background":
            background_indices,

        "looming":
            looming_indices,

    },

    seed=999

)


warmup.step(

    group_rates={

        "background": 0.0,

        "looming": 0.0,

    },

    chunk_ms=1.0

)


print(
    "Warmup tamam."
)


# ============================================================
# TEK RATE TESTİ
# ============================================================

def run_rate(
    background_rate
):

    session = MultiInputBrainSession(

        brain,

        input_groups={

            "background":
                background_indices,

            "looming":
                looming_indices,

        },

        seed=SEED

    )


    action_counts = {

        "IDLE": 0,

        "MOVE": 0,

        "FLY": 0,

        "JUMP": 0,

    }


    # ========================================================
    # PRE-LOOMING AUTONOMY
    # 0 - 2000 ms
    # ========================================================

    pre_move = 0

    pre_fly = 0

    pre_motor = 0


    # ========================================================
    # POST-LOOMING AUTONOMY
    # 2500 - 8000 ms
    # ========================================================

    post_move = 0

    post_fly = 0

    post_motor = 0


    background_events = 0

    looming_events = 0


    total_spikes = 0


    peak_spikes = 0

    max_compute_ms = 0.0

    sum_compute_ms = 0.0


    over_20ms_chunks = 0

    runaway = False


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


        # ====================================================
        # LOOMING RATE
        # ====================================================

        if (

            LOOMING_START_MS
            <=
            brain_time_ms
            <
            LOOMING_END_MS

        ):

            looming_rate = (
                LOOMING_RATE_HZ
            )

        else:

            looming_rate = 0.0


        # ====================================================
        # BRAIN
        # ====================================================

        compute_start = (
            time.perf_counter()
        )


        result = session.step(

            group_rates={

                "background":
                    background_rate,

                "looming":
                    looming_rate,

            },

            chunk_ms=CHUNK_MS

        )


        compute_ms = (

            time.perf_counter()

            -

            compute_start

        ) * 1000.0


        # ====================================================
        # PERFORMANCE
        # ====================================================

        sum_compute_ms += (
            compute_ms
        )


        max_compute_ms = max(

            max_compute_ms,

            compute_ms

        )


        if compute_ms > CHUNK_MS:

            over_20ms_chunks += 1


        # ====================================================
        # EVENTS
        # ====================================================

        background_events += (

            result[
                "stimulus_by_group"
            ].get(
                "background",
                0
            )

        )


        looming_events += (

            result[
                "stimulus_by_group"
            ].get(
                "looming",
                0
            )

        )


        total_spikes += result[
            "total_spikes"
        ]


        peak_spikes = max(

            peak_spikes,

            result[
                "total_spikes"
            ]

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


        action_counts[
            action
        ] = (

            action_counts.get(
                action,
                0
            )

            +

            1

        )


        # ====================================================
        # PRE-LOOMING
        # ====================================================

        if brain_time_ms < LOOMING_START_MS:

            if action == "MOVE":

                pre_move += 1


            if action == "FLY":

                pre_fly += 1


            pre_motor += result[
                "motor_spikes"
            ]


        # ====================================================
        # POST-LOOMING
        # 440 ms recovery payı bırakıyoruz.
        # ====================================================

        if brain_time_ms >= 2500.0:

            if action == "MOVE":

                post_move += 1


            if action == "FLY":

                post_fly += 1


            post_motor += result[
                "motor_spikes"
            ]


        # ====================================================
        # RUNAWAY
        # ====================================================

        if (

            result[
                "total_spikes"
            ]

            >

            RUNAWAY_LIMIT

        ):

            runaway = True

            break


    actual_chunks = sum(
        action_counts.values()
    )


    if actual_chunks > 0:

        mean_compute_ms = (

            sum_compute_ms

            /

            actual_chunks

        )

    else:

        mean_compute_ms = 0.0


    return {

        "rate":
            background_rate,

        "bg_events":
            background_events,

        "loom_events":
            looming_events,

        "total_spikes":
            total_spikes,

        "idle":
            action_counts.get(
                "IDLE",
                0
            ),

        "move":
            action_counts.get(
                "MOVE",
                0
            ),

        "fly":
            action_counts.get(
                "FLY",
                0
            ),

        "jump":
            action_counts.get(
                "JUMP",
                0
            ),

        "pre_move":
            pre_move,

        "pre_fly":
            pre_fly,

        "pre_motor":
            pre_motor,

        "post_move":
            post_move,

        "post_fly":
            post_fly,

        "post_motor":
            post_motor,

        "peak":
            peak_spikes,

        "mean_compute":
            mean_compute_ms,

        "max_compute":
            max_compute_ms,

        "over20":
            over_20ms_chunks,

        "runaway":
            runaway,

    }


# ============================================================
# TESTLER
# ============================================================

results = []


print("\n" + "=" * 90)
print("CALIBRATION BAŞLIYOR")
print("=" * 90)


for rate in BACKGROUND_RATES:

    print()

    print(
        f"Background {rate:.2f} Hz çalışıyor..."
    )


    result = run_rate(
        rate
    )


    results.append(
        result
    )


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 155)
print("LOW-LOAD SONUÇLARI")
print("=" * 155)


print(

    f"{'Rate':>6}"

    f" {'BgEvt':>7}"

    f" {'Loom':>6}"

    f" {'IDLE':>6}"

    f" {'MOVE':>6}"

    f" {'FLY':>5}"

    f" {'PreMove':>8}"

    f" {'PreMotor':>9}"

    f" {'PostMove':>9}"

    f" {'PostMotor':>10}"

    f" {'Peak':>7}"

    f" {'MeanMS':>8}"

    f" {'MaxMS':>8}"

    f" {'>20ms':>7}"

    f" {'Runaway':>9}"

)


for r in results:

    print(

        f"{r['rate']:>6.2f}"

        f" {r['bg_events']:>7}"

        f" {r['loom_events']:>6}"

        f" {r['idle']:>6}"

        f" {r['move']:>6}"

        f" {r['fly']:>5}"

        f" {r['pre_move']:>8}"

        f" {r['pre_motor']:>9}"

        f" {r['post_move']:>9}"

        f" {r['post_motor']:>10}"

        f" {r['peak']:>7}"

        f" {r['mean_compute']:>8.2f}"

        f" {r['max_compute']:>8.2f}"

        f" {r['over20']:>7}"

        f" {str(r['runaway']):>9}"

    )


print("\n" + "=" * 155)
print("TEST BİTTİ")
print("=" * 155)