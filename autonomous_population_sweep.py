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

BACKGROUND_RATE_HZ = 0.20

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# KAÇ SENSORY NÖRON?
#
# Her superclass'tan ayrı ayrı bu kadar seçilecek.
#
# Örnek:
# 170 -> toplam 510 sensory neuron
# ============================================================

PER_GROUP_VALUES = [
    170,
    180,
    190,
    200,
]


BACKGROUND_SUPERCLASSES = [
    "ol_sensory",
    "cb_sensory",
    "vnc_sensory",
]


# ============================================================
# YÜKSEK AKTİVİTE TEST GÜVENLİĞİ
#
# 1500+ spike/chunk 500 ms boyunca devam ederse
# bu adayı "HIGH STATE" kabul edip testi erken kesiyoruz.
#
# Bu sadece calibration script güvenliği.
# Brain modeline reset/suppression eklemiyoruz.
# ============================================================

HIGH_STATE_SPIKES = 1500

HIGH_STATE_CONSECUTIVE_CHUNKS = 25


# ============================================================
# BRAIN
# ============================================================

print("=" * 95)
print("MALECNS AUTONOMOUS POPULATION SIZE SWEEP")
print("=" * 95)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


print(
    "W_SYN:",
    brain.w_syn
)

print(
    "Background rate:",
    BACKGROUND_RATE_HZ,
    "Hz"
)


# ============================================================
# CONNECTOME OUT-DEGREE
# ============================================================

out_degree = np.diff(
    brain.indptr
)


# ============================================================
# HER SUPERCLASS İÇİN TOP 200'Ü ÖNCEDEN HAZIRLA
# ============================================================

MAX_PER_GROUP = max(
    PER_GROUP_VALUES
)


ranked_groups = {}


print("\n" + "=" * 95)
print("SENSORY POPULATION HAZIRLANIYOR")
print("=" * 95)


for superclass in BACKGROUND_SUPERCLASSES:

    candidates = np.flatnonzero(

        (
            brain.superclasses
            ==
            superclass
        ).to_numpy()

    )


    candidate_degree = out_degree[
        candidates
    ]


    order = np.argsort(
        candidate_degree
    )[::-1]


    ranked = candidates[
        order[:MAX_PER_GROUP]
    ]


    ranked_groups[
        superclass
    ] = ranked


    print(
        superclass,
        "| candidate:",
        len(candidates),
        "| hazır:",
        len(ranked)
    )


# ============================================================
# WARMUP
# ============================================================

warmup_indices = []


for superclass in BACKGROUND_SUPERCLASSES:

    warmup_indices.extend(

        ranked_groups[
            superclass
        ][:170].tolist()

    )


warmup_indices = np.asarray(

    warmup_indices,

    dtype=np.int32

)


warmup = MultiInputBrainSession(

    brain,

    input_groups={

        "background":
            warmup_indices

    },

    seed=999

)


warmup.step(

    group_rates={

        "background": 0.0

    },

    chunk_ms=1.0

)


print(
    "\nWarmup tamam."
)


# ============================================================
# TEK POPULATION SIZE TESTİ
# ============================================================

def run_population_test(
    per_group
):

    # ========================================================
    # POPULATION
    # ========================================================

    selected = []


    for superclass in BACKGROUND_SUPERCLASSES:

        selected.extend(

            ranked_groups[
                superclass
            ][:per_group].tolist()

        )


    selected = np.asarray(

        selected,

        dtype=np.int32

    )


    # ========================================================
    # CLEAN SESSION
    # ========================================================

    session = MultiInputBrainSession(

        brain,

        input_groups={

            "background":
                selected

        },

        seed=SEED

    )


    # ========================================================
    # STATS
    # ========================================================

    idle_chunks = 0

    move_chunks = 0

    fly_chunks = 0

    jump_chunks = 0


    total_motor = 0

    total_dn = 0

    total_spikes = 0

    background_events = 0


    first_motor_ms = None


    peak_spikes = 0

    sum_compute_ms = 0.0

    max_compute_ms = 0.0

    over_20ms = 0


    high_state_counter = 0

    high_state = False

    runaway = False


    # ========================================================
    # SON 2 SANİYE İÇİN
    # ========================================================

    late_spikes = 0

    late_motor = 0

    late_chunks = 0


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

                "background":
                    BACKGROUND_RATE_HZ

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

            over_20ms += 1


        # ====================================================
        # NETWORK
        # ====================================================

        spikes = result[
            "total_spikes"
        ]


        motor = result[
            "motor_spikes"
        ]


        dn = result[
            "descending_spikes"
        ]


        total_spikes += spikes

        total_motor += motor

        total_dn += dn


        background_events += (

            result[
                "stimulus_by_group"
            ].get(
                "background",
                0
            )

        )


        peak_spikes = max(

            peak_spikes,

            spikes

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
        # İLK MOTOR DAVRANIŞI
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
        # LATE WINDOW
        # 8-10 saniye
        # ====================================================

        if brain_time_ms >= 8000.0:

            late_spikes += spikes

            late_motor += motor

            late_chunks += 1


        # ====================================================
        # HIGH ACTIVITY STATE
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
    # CALCULATIONS
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

        mean_compute_ms = (

            sum_compute_ms

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


        late_mean_motor = (

            late_motor

            /

            late_chunks

        )

    else:

        late_mean_spikes = 0.0

        late_mean_motor = 0.0


    return {

        "per_group":
            per_group,

        "total_sources":
            len(selected),

        "events":
            background_events,

        "first_motor_ms":
            first_motor_ms,

        "idle":
            idle_chunks,

        "move":
            move_chunks,

        "fly":
            fly_chunks,

        "jump":
            jump_chunks,

        "motor":
            total_motor,

        "dn":
            total_dn,

        "total_spikes":
            total_spikes,

        "peak":
            peak_spikes,

        "late_spikes":
            late_mean_spikes,

        "late_motor":
            late_mean_motor,

        "mean_compute":
            mean_compute_ms,

        "max_compute":
            max_compute_ms,

        "over20":
            over_20ms,

        "high_state":
            high_state,

        "runaway":
            runaway,

        "completed_ms":
            completed_chunks
            *
            CHUNK_MS,

    }


# ============================================================
# SWEEP
# ============================================================

results = []


print("\n" + "=" * 95)
print("POPULATION SWEEP BAŞLIYOR")
print("=" * 95)


for per_group in PER_GROUP_VALUES:

    total_sources = (

        per_group

        *

        len(
            BACKGROUND_SUPERCLASSES
        )

    )


    print()

    print(
        f"{per_group} / grup "
        f"({total_sources} toplam) çalışıyor..."
    )


    result = run_population_test(
        per_group
    )


    results.append(
        result
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 175)
print("POPULATION SWEEP SONUCU")
print("=" * 175)


print(

    f"{'PerGrp':>7}"

    f" {'Sources':>8}"

    f" {'Events':>8}"

    f" {'FirstMotor':>11}"

    f" {'IDLE':>6}"

    f" {'MOVE':>6}"

    f" {'FLY':>5}"

    f" {'Motor':>8}"

    f" {'Peak':>7}"

    f" {'LateSpike':>10}"

    f" {'LateMotor':>10}"

    f" {'MeanMS':>8}"

    f" {'MaxMS':>8}"

    f" {'>20ms':>7}"

    f" {'HighState':>10}"

    f" {'Runaway':>9}"

    f" {'DoneMS':>8}"

)


for r in results:

    first_motor = (

        "-"

        if r[
            "first_motor_ms"
        ] is None

        else

        f"{r['first_motor_ms']:.0f}"

    )


    print(

        f"{r['per_group']:>7}"

        f" {r['total_sources']:>8}"

        f" {r['events']:>8}"

        f" {first_motor:>11}"

        f" {r['idle']:>6}"

        f" {r['move']:>6}"

        f" {r['fly']:>5}"

        f" {r['motor']:>8}"

        f" {r['peak']:>7}"

        f" {r['late_spikes']:>10.1f}"

        f" {r['late_motor']:>10.2f}"

        f" {r['mean_compute']:>8.2f}"

        f" {r['max_compute']:>8.2f}"

        f" {r['over20']:>7}"

        f" {str(r['high_state']):>10}"

        f" {str(r['runaway']):>9}"

        f" {r['completed_ms']:>8.0f}"

    )


print("\n" + "=" * 175)
print("TEST BİTTİ")
print("=" * 175)