import numpy as np
from time import perf_counter

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session import ContinuousBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# FINAL PARAMETRE
# ============================================================

W_SYN = 0.110


# ============================================================
# TEST AYARLARI
# ============================================================

SEEDS = [
    101,
    202,
    303
]

CHUNK_MS = 20.0

TOTAL_BRAIN_MS = 10000.0

STIM_RATE_HZ = 40.0

RUNAWAY_LIMIT = 10000


# ============================================================
# İKİ AYRI LOOMING EVENT
#
# 1. event: 500-560 ms
# 2. event: 5000-5060 ms
# ============================================================

FIRST_STIM_START = 500.0
FIRST_STIM_END = 560.0

SECOND_STIM_START = 5000.0
SECOND_STIM_END = 5060.0


# Response pencereleri

FIRST_RESPONSE_START = 500.0
FIRST_RESPONSE_END = 1000.0

SECOND_RESPONSE_START = 5000.0
SECOND_RESPONSE_END = 5500.0


# En son 500 ms

LATE_START = 9500.0
LATE_END = 10000.0


# ============================================================
# BRAIN
# ============================================================

brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


# ============================================================
# WARMUP
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION WARMUP")
print("=" * 70)


warmup = ContinuousBrainSession(

    brain,

    source_types=[
        "LPLC2",
        "LC4"
    ],

    seed=999
)


warmup.step(
    rate_hz=0,
    chunk_ms=1.0
)


print("Compile tamam.")


# ============================================================
# TEK SEED
# ============================================================

def run_seed(seed):

    session = ContinuousBrainSession(

        brain,

        source_types=[
            "LPLC2",
            "LC4"
        ],

        seed=seed
    )


    # --------------------------------------------------------
    # Spike accumulator
    # --------------------------------------------------------

    first_response = np.zeros(

        brain.N,

        dtype=np.int32

    )


    second_response = np.zeros(

        brain.N,

        dtype=np.int32

    )


    late_response = np.zeros(

        brain.N,

        dtype=np.int32

    )


    stimulus_events = 0

    peak_chunk = 0

    runaway = False


    chunks = int(

        TOTAL_BRAIN_MS

        /

        CHUNK_MS

    )


    start_time = perf_counter()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    for chunk in range(chunks):

        brain_time = (

            chunk

            *

            CHUNK_MS

        )


        # ====================================================
        # INPUT RATE
        # ====================================================

        first_event = (

            FIRST_STIM_START
            <=
            brain_time
            <
            FIRST_STIM_END

        )


        second_event = (

            SECOND_STIM_START
            <=
            brain_time
            <
            SECOND_STIM_END

        )


        if first_event or second_event:

            rate = STIM_RATE_HZ

        else:

            rate = 0.0


        # ====================================================
        # BRAIN STEP
        # ====================================================

        result = session.step(

            rate_hz=rate,

            chunk_ms=CHUNK_MS

        )


        stimulus_events += result[
            "stimulus_events"
        ]


        peak_chunk = max(

            peak_chunk,

            result[
                "total_spikes"
            ]

        )


        # ====================================================
        # FIRST RESPONSE
        # ====================================================

        if (

            FIRST_RESPONSE_START
            <=
            brain_time
            <
            FIRST_RESPONSE_END

        ):

            first_response += result[
                "spike_counts"
            ]


        # ====================================================
        # SECOND RESPONSE
        # ====================================================

        if (

            SECOND_RESPONSE_START
            <=
            brain_time
            <
            SECOND_RESPONSE_END

        ):

            second_response += result[
                "spike_counts"
            ]


        # ====================================================
        # FINAL LATE WINDOW
        # ====================================================

        if (

            LATE_START
            <=
            brain_time
            <
            LATE_END

        ):

            late_response += result[
                "spike_counts"
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


    elapsed = (

        perf_counter()

        -

        start_time

    )


    # ========================================================
    # MOTOR DECODER
    # ========================================================

    first_movement = decoder.decode(

        neurons=
            brain.neurons,

        spike_counts=
            first_response,

        motor_mask=
            brain.motor_mask

    )


    second_movement = decoder.decode(

        neurons=
            brain.neurons,

        spike_counts=
            second_response,

        motor_mask=
            brain.motor_mask

    )


    late_movement = decoder.decode(

        neurons=
            brain.neurons,

        spike_counts=
            late_response,

        motor_mask=
            brain.motor_mask

    )


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "seed":
            seed,

        "stimulus":
            stimulus_events,

        "first_total":
            int(
                first_response.sum()
            ),

        "first_dn":
            int(
                first_response[
                    brain.descending_mask
                ].sum()
            ),

        "first_motor":
            int(
                first_response[
                    brain.motor_mask
                ].sum()
            ),

        "first_action":
            first_movement[
                "action"
            ],

        "second_total":
            int(
                second_response.sum()
            ),

        "second_dn":
            int(
                second_response[
                    brain.descending_mask
                ].sum()
            ),

        "second_motor":
            int(
                second_response[
                    brain.motor_mask
                ].sum()
            ),

        "second_action":
            second_movement[
                "action"
            ],

        "late_total":
            int(
                late_response.sum()
            ),

        "late_dn":
            int(
                late_response[
                    brain.descending_mask
                ].sum()
            ),

        "late_motor":
            int(
                late_response[
                    brain.motor_mask
                ].sum()
            ),

        "late_action":
            late_movement[
                "action"
            ],

        "peak":
            peak_chunk,

        "runaway":
            runaway,

        "seconds":
            elapsed
    }


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 100)
print("MALECNS FINAL 3-SEED VALIDATION")
print("=" * 100)

print(
    "W_SYN:",
    W_SYN
)

print(
    "Her seed:",
    TOTAL_BRAIN_MS,
    "ms beyin zamanı"
)

print(
    "Event 1: 500-560 ms"
)

print(
    "Event 2: 5000-5060 ms"
)


results = []


for seed in SEEDS:

    print(
        f"\nSeed {seed} çalışıyor..."
    )

    result = run_seed(
        seed
    )

    results.append(
        result
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 150)
print("FINAL SONUÇ")
print("=" * 150)


print(

    f"{'Seed':>5}"

    f" {'Stim':>6}"

    f" {'R1Motor':>8}"

    f" {'R1Action':>9}"

    f" {'R2Motor':>8}"

    f" {'R2Action':>9}"

    f" {'LateSpike':>10}"

    f" {'LateDN':>8}"

    f" {'LateMotor':>10}"

    f" {'LateAction':>11}"

    f" {'Peak':>8}"

    f" {'Runaway':>10}"

    f" {'Sec':>7}"

)


for r in results:

    print(

        f"{r['seed']:>5}"

        f" {r['stimulus']:>6}"

        f" {r['first_motor']:>8}"

        f" {r['first_action']:>9}"

        f" {r['second_motor']:>8}"

        f" {r['second_action']:>9}"

        f" {r['late_total']:>10}"

        f" {r['late_dn']:>8}"

        f" {r['late_motor']:>10}"

        f" {r['late_action']:>11}"

        f" {r['peak']:>8}"

        f" {str(r['runaway']):>10}"

        f" {r['seconds']:>7.2f}"

    )


# ============================================================
# PASS / FAIL
# ============================================================

all_pass = True


for r in results:

    if r["runaway"]:

        all_pass = False


    if r["first_motor"] <= 0:

        all_pass = False


    if r["second_motor"] <= 0:

        all_pass = False


    if r["late_motor"] != 0:

        all_pass = False


    if r["late_action"] != "IDLE":

        all_pass = False


print("\n" + "=" * 100)


if all_pass:

    print(
        "FINAL VALIDATION: PASS"
    )

else:

    print(
        "FINAL VALIDATION: FAIL"
    )


print("=" * 100)