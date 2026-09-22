import numpy as np

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session import ContinuousBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# AYARLAR
# ============================================================

W_SYN = 0.110

CHUNK_MS = 20.0

TEST_DURATION_MS = 6000.0

SEED = 404

RUNAWAY_LIMIT = 10000


# ============================================================
# DAHA GÜÇLÜ BACKGROUND RATE SWEEP
# ============================================================

BACKGROUND_RATES = [
    0.20,
    0.50,
    1.00,
    2.00,
    5.00,
]


# ============================================================
# HER SENSORY GRUPTAN SEÇİLECEK NÖRON
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

print("=" * 80)
print("MALECNS AUTONOMOUS BACKGROUND TEST V2")
print("=" * 80)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


print(
    "W_SYN:",
    brain.w_syn
)


# ============================================================
# CONNECTOME OUT-DEGREE
#
# Matrix CSC:
# column = presynaptic neuron
#
# Bir nöron kaç farklı downstream bağlantıya sahip?
# ============================================================

out_degree = np.diff(
    brain.indptr
)


# ============================================================
# RANDOM DEĞİL:
# HER SENSORY SUPERCLASS İÇİN EN ÇOK OUTGOING EDGE'E
# SAHİP NÖRONLARI SEÇ
# ============================================================

selected_indices = []


print("\n" + "=" * 80)
print("BACKGROUND NÖRON SEÇİMİ")
print("=" * 80)


for superclass in BACKGROUND_SUPERCLASSES:

    candidates = np.flatnonzero(

        (
            brain.superclasses
            ==
            superclass
        ).to_numpy()

    )


    candidate_degrees = out_degree[
        candidates
    ]


    order = np.argsort(
        candidate_degrees
    )[::-1]


    take_count = min(
        PER_GROUP,
        len(candidates)
    )


    chosen = candidates[
        order[:take_count]
    ]


    selected_indices.extend(
        chosen.tolist()
    )


    chosen_degrees = out_degree[
        chosen
    ]


    print()

    print(
        superclass
    )

    print(
        "Candidate:",
        len(candidates)
    )

    print(
        "Selected:",
        len(chosen)
    )

    print(
        "Selected mean out-degree:",
        round(
            float(
                chosen_degrees.mean()
            ),
            2
        )
    )

    print(
        "Selected max out-degree:",
        int(
            chosen_degrees.max()
        )
    )


selected_indices = np.array(

    selected_indices,

    dtype=np.int32

)


print(
    "\nToplam background sensory neuron:",
    len(selected_indices)
)


print(
    "Toplam seçili outgoing edge:",
    int(
        out_degree[
            selected_indices
        ].sum()
    )
)


# ============================================================
# WARMUP
# ============================================================

warmup = ContinuousBrainSession(

    brain,

    source_types=[],

    seed=999

)


warmup.step(

    rate_hz=0,

    chunk_ms=1.0

)


print(
    "\nNumba warmup tamam."
)


# ============================================================
# TEK BACKGROUND TEST
# ============================================================

def run_background_test(rate_hz):

    session = ContinuousBrainSession(

        brain,

        source_types=[],

        seed=SEED

    )


    # ========================================================
    # SEÇTİĞİMİZ SENSORY POPULATION'I SOURCE YAP
    # ========================================================

    session.source_indices = (
        selected_indices.copy()
    )


    session.source_mask = np.zeros(

        brain.N,

        dtype=np.bool_

    )


    session.source_mask[
        selected_indices
    ] = True


    # ========================================================
    # COUNTERS
    # ========================================================

    action_counts = {

        "IDLE": 0,

        "MOVE": 0,

        "FLY": 0,

        "JUMP": 0,

    }


    stimulus_events = 0

    total_spikes = 0

    total_dn = 0

    total_motor = 0

    peak_chunk = 0

    runaway = False


    last_action = "IDLE"

    transitions = []


    chunk_count = int(

        TEST_DURATION_MS

        /

        CHUNK_MS

    )


    # ========================================================
    # LOOP
    # ========================================================

    for chunk in range(
        chunk_count
    ):

        result = session.step(

            rate_hz=rate_hz,

            chunk_ms=CHUNK_MS

        )


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


        stimulus_events += result[
            "stimulus_events"
        ]


        total_spikes += result[
            "total_spikes"
        ]


        total_dn += result[
            "descending_spikes"
        ]


        total_motor += result[
            "motor_spikes"
        ]


        peak_chunk = max(

            peak_chunk,

            result[
                "total_spikes"
            ]

        )


        # ====================================================
        # ACTION TRANSITION
        # ====================================================

        if action != last_action:

            brain_time_ms = (

                chunk

                *

                CHUNK_MS

            )


            transitions.append(

                (
                    brain_time_ms,
                    last_action,
                    action
                )

            )


            last_action = action


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


    # ========================================================
    # SOURCE SPIKE'LARINI ÇIKAR
    #
    # Böylece connectome içinde kaç ekstra spike üretildiğini
    # görüyoruz.
    # ========================================================

    network_extra_spikes = max(

        0,

        total_spikes
        -
        stimulus_events

    )


    return {

        "rate":
            rate_hz,

        "events":
            stimulus_events,

        "spikes":
            total_spikes,

        "extra":
            network_extra_spikes,

        "dn":
            total_dn,

        "motor":
            total_motor,

        "peak":
            peak_chunk,

        "runaway":
            runaway,

        "actions":
            action_counts,

        "transitions":
            transitions,

    }


# ============================================================
# SWEEP
# ============================================================

results = []


print("\n" + "=" * 90)
print("HIGH-CONNECTIVITY SENSORY BACKGROUND SWEEP")
print("=" * 90)


for rate in BACKGROUND_RATES:

    print(
        f"\nBackground rate {rate:.2f} Hz çalışıyor..."
    )


    result = run_background_test(
        rate
    )


    results.append(
        result
    )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 125)
print("SONUÇ")
print("=" * 125)


print(

    f"{'Rate':>7}"

    f" {'Events':>8}"

    f" {'Spikes':>10}"

    f" {'Extra':>10}"

    f" {'DN':>8}"

    f" {'Motor':>8}"

    f" {'IDLE':>7}"

    f" {'MOVE':>7}"

    f" {'FLY':>7}"

    f" {'JUMP':>7}"

    f" {'Peak':>8}"

    f" {'Runaway':>10}"

)


for result in results:

    actions = result[
        "actions"
    ]


    print(

        f"{result['rate']:>7.2f}"

        f" {result['events']:>8}"

        f" {result['spikes']:>10}"

        f" {result['extra']:>10}"

        f" {result['dn']:>8}"

        f" {result['motor']:>8}"

        f" {actions.get('IDLE', 0):>7}"

        f" {actions.get('MOVE', 0):>7}"

        f" {actions.get('FLY', 0):>7}"

        f" {actions.get('JUMP', 0):>7}"

        f" {result['peak']:>8}"

        f" {str(result['runaway']):>10}"

    )


# ============================================================
# TRANSITIONS
# ============================================================

print("\n" + "=" * 100)
print("ACTION TRANSITION ÖRNEKLERİ")
print("=" * 100)


for result in results:

    print(
        f"\nRATE = {result['rate']:.2f} Hz"
    )


    transitions = result[
        "transitions"
    ]


    if len(transitions) == 0:

        print(
            "Action değişimi yok."
        )

        continue


    for transition in transitions[:25]:

        brain_time_ms, old, new = transition


        print(

            f"{brain_time_ms:7.0f} ms"

            f" | {old}"

            f" -> {new}"

        )


    if len(transitions) > 25:

        print(

            "... toplam",

            len(transitions),

            "transition"

        )


print("\n" + "=" * 80)
print("TEST BİTTİ")
print("=" * 80)