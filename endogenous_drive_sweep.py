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

TEST_BRAIN_MS = 12000.0

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# ENDOGENOUS DRIVE
#
# Aynı merkezi population kullanılacak.
# Sadece rate değişecek.
#
# Aynı seed + aynı source sayısı sayesinde rate sweep
# önceki population sweep'ten daha kontrollü.
# ============================================================

DRIVE_RATES_HZ = [

    0.05,
    0.10,
    0.20,
    0.40,
    0.80,

]


# ============================================================
# KAÇ CENTRAL NÖRONU DRIVE EDECEĞİZ
# ============================================================

DRIVE_NEURON_COUNT = 256


# ============================================================
# HIGH-ACTIVITY SAFETY
#
# Modeli değiştirmez.
# Sadece calibration testini erken durdurur.
# ============================================================

HIGH_STATE_SPIKES = 1500

HIGH_STATE_CONSECUTIVE_CHUNKS = 25


# ============================================================
# BRAIN
# ============================================================

print("=" * 105)
print("MALECNS ENDOGENOUS CENTRAL DRIVE SWEEP")
print("=" * 105)


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
# CB_INTRINSIC + FAST_EXCITATORY
#
# Visual projection değil.
# Sensory değil.
# Motor değil.
# Descending değil.
#
# İlk endogenous-drive adayımız.
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


print("\n" + "=" * 105)
print("CENTRAL DRIVE POPULATION")
print("=" * 105)


print(
    "cb_intrinsic + fast_excitatory:",
    len(
        base_indices
    )
)


# ============================================================
# ORTA CONNECTIVITY BANDI
#
# Aşırı hub kullanmıyoruz.
#
# Out-degree:
# p50 - p75
#
# In-degree:
# >= p25
# ============================================================

base_out = out_degree[
    base_indices
]


base_in = in_degree[
    base_indices
]


out_low = float(

    np.percentile(
        base_out,
        50
    )

)


out_high = float(

    np.percentile(
        base_out,
        75
    )

)


in_low = float(

    np.percentile(
        base_in,
        25
    )

)


balanced_mask = (

    base_mask

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

    &

    (
        in_degree
        >=
        in_low
    )

)


balanced_indices = np.flatnonzero(
    balanced_mask
)


print(
    "Out-degree band:",
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


print(
    "Minimum in-degree:",
    round(
        in_low,
        2
    )
)


print(
    "Balanced candidate:",
    len(
        balanced_indices
    )
)


if (
    len(
        balanced_indices
    )
    <
    DRIVE_NEURON_COUNT
):

    raise RuntimeError(

        "Yeterli balanced central neuron yok."

    )


# ============================================================
# DETERMINISTIC POPULATION SEÇ
#
# Davranış seçmiyoruz.
#
# Sadece hangi central neuronların düşük seviyeli modeled
# endogenous drive alacağını sabitliyoruz.
# ============================================================

selection_rng = np.random.default_rng(
    SEED
)


drive_indices = selection_rng.choice(

    balanced_indices,

    size=DRIVE_NEURON_COUNT,

    replace=False

).astype(
    np.int32
)


drive_indices.sort()


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
    "Mean in-degree:",
    round(
        float(
            in_degree[
                drive_indices
            ].mean()
        ),
        2
    )
)


# ============================================================
# SEÇİLEN TYPE'LARIN KISA ÖZETİ
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
        20
    ).to_string()
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
# TEK RATE TESTİ
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


    idle_chunks = 0

    move_chunks = 0

    fly_chunks = 0

    jump_chunks = 0


    active_motor_chunks = 0


    total_events = 0

    total_spikes = 0

    total_dn = 0

    total_motor = 0


    peak_spikes = 0


    first_motor_ms = None

    first_move_ms = None


    # ========================================================
    # PERFORMANCE
    # ========================================================

    compute_sum_ms = 0.0

    compute_max_ms = 0.0

    over_20ms = 0


    # ========================================================
    # HIGH STATE
    # ========================================================

    high_state_counter = 0

    high_state = False

    runaway = False


    # ========================================================
    # SON 2 SANİYE
    # ========================================================

    late_spikes = 0

    late_motor = 0

    late_dn = 0

    late_chunks = 0


    late_start_ms = (

        TEST_BRAIN_MS
        -
        2000.0

    )


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
        # RAW NETWORK
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


        if motor > 0:

            active_motor_chunks += 1


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
        # FIRST EVENTS
        # ====================================================

        if (

            motor > 0

            and

            first_motor_ms is None

        ):

            first_motor_ms = (
                brain_time_ms
            )


        if (

            action == "MOVE"

            and

            first_move_ms is None

        ):

            first_move_ms = (
                brain_time_ms
            )


        # ====================================================
        # LATE WINDOW
        # ====================================================

        if brain_time_ms >= late_start_ms:

            late_spikes += spikes

            late_motor += motor

            late_dn += dn

            late_chunks += 1


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


        late_mean_motor = (

            late_motor

            /

            late_chunks

        )


        late_mean_dn = (

            late_dn

            /

            late_chunks

        )

    else:

        late_mean_spikes = 0.0

        late_mean_motor = 0.0

        late_mean_dn = 0.0


    return {

        "rate":
            rate_hz,

        "events":
            total_events,

        "spikes":
            total_spikes,

        "dn":
            total_dn,

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

        "first_motor":
            first_motor_ms,

        "first_move":
            first_move_ms,

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


print("\n" + "=" * 105)
print("ENDOGENOUS RATE SWEEP")
print("=" * 105)


for rate in DRIVE_RATES_HZ:

    print()

    print(
        f"Endogenous drive {rate:.2f} Hz çalışıyor..."
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

print("\n" + "=" * 190)
print("ENDOGENOUS DRIVE SONUCU")
print("=" * 190)


print(

    f"{'Rate':>6}"

    f" {'Events':>8}"

    f" {'DN':>8}"

    f" {'Motor':>8}"

    f" {'MotChk':>7}"

    f" {'IDLE':>6}"

    f" {'MOVE':>6}"

    f" {'FLY':>5}"

    f" {'JUMP':>6}"

    f" {'FirstMot':>9}"

    f" {'FirstMove':>10}"

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

    if r["first_motor"] is None:

        first_motor = "-"

    else:

        first_motor = (
            f"{r['first_motor']:.0f}"
        )


    if r["first_move"] is None:

        first_move = "-"

    else:

        first_move = (
            f"{r['first_move']:.0f}"
        )


    print(

        f"{r['rate']:>6.2f}"

        f" {r['events']:>8}"

        f" {r['dn']:>8}"

        f" {r['motor']:>8}"

        f" {r['motor_chunks']:>7}"

        f" {r['idle']:>6}"

        f" {r['move']:>6}"

        f" {r['fly']:>5}"

        f" {r['jump']:>6}"

        f" {first_motor:>9}"

        f" {first_move:>10}"

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


print("\n" + "=" * 190)
print("TEST BİTTİ")
print("=" * 190)