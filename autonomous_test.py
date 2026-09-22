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
# BACKGROUND RATE TESTLERİ
#
# Bunlar Hz / sensory neuron.
#
# Action rastgele seçilmiyor.
# Sadece düşük seviyeli sensory background geliyor.
# ============================================================

BACKGROUND_RATES = [
    0.02,
    0.05,
    0.10,
    0.20,
]


# ============================================================
# HER DUYU GRUBUNDAN KAÇ NÖRON
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
print("MALECNS AUTONOMOUS BACKGROUND TEST")
print("=" * 80)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


print(
    "W_SYN:",
    brain.w_syn
)


# ============================================================
# BACKGROUND SENSORY POPULATION SEÇ
#
# Reproducible seçim:
# seed aynıysa aynı nöronlar seçilir.
# ============================================================

rng = np.random.default_rng(
    SEED
)


selected_indices = []


for superclass in BACKGROUND_SUPERCLASSES:

    candidates = np.flatnonzero(

        (
            brain.superclasses
            ==
            superclass
        ).to_numpy()

    )


    if len(candidates) < PER_GROUP:

        chosen = candidates

    else:

        chosen = rng.choice(

            candidates,

            size=PER_GROUP,

            replace=False

        )


    selected_indices.extend(
        chosen.tolist()
    )


    print(

        superclass,

        "| candidate:",

        len(candidates),

        "| selected:",

        len(chosen)

    )


selected_indices = np.array(

    selected_indices,

    dtype=np.int32

)


print(
    "\nToplam background sensory neuron:",
    len(selected_indices)
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
    "Numba warmup tamam."
)


# ============================================================
# TEK RATE TESTİ
# ============================================================

def run_background_test(rate_hz):

    # --------------------------------------------------------
    # Yeni temiz brain session
    # --------------------------------------------------------

    session = ContinuousBrainSession(

        brain,

        source_types=[],

        seed=SEED

    )


    # --------------------------------------------------------
    # Normalde ContinuousBrainSession source_types kullanıyor.
    #
    # Bu testte seçtiğimiz sensory neuronları manuel olarak
    # external background source yapıyoruz.
    # --------------------------------------------------------

    session.source_indices = (
        selected_indices.copy()
    )


    session.source_mask = np.zeros(

        brain.N,

        dtype=np.bool_

    )


    session.source_mask[
        session.source_indices
    ] = True


    # --------------------------------------------------------
    # ACTION COUNTERS
    # --------------------------------------------------------

    action_counts = {

        "IDLE": 0,

        "MOVE": 0,

        "FLY": 0,

        "JUMP": 0,

    }


    total_motor = 0

    total_dn = 0

    total_spikes = 0

    total_stimulus = 0


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


        if action not in action_counts:

            action_counts[
                action
            ] = 0


        action_counts[
            action
        ] += 1


        total_motor += result[
            "motor_spikes"
        ]


        total_dn += result[
            "descending_spikes"
        ]


        total_spikes += result[
            "total_spikes"
        ]


        total_stimulus += result[
            "stimulus_events"
        ]


        peak_chunk = max(

            peak_chunk,

            result[
                "total_spikes"
            ]

        )


        # ====================================================
        # ACTION DEĞİŞİMİ
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


    return {

        "rate":
            rate_hz,

        "stimulus":
            total_stimulus,

        "spikes":
            total_spikes,

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
# TÜM RATE TESTLERİ
# ============================================================

results = []


print("\n" + "=" * 80)
print("BACKGROUND RATE SWEEP")
print("=" * 80)


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
# ÖZET
# ============================================================

print("\n" + "=" * 110)
print("SONUÇ")
print("=" * 110)


print(

    f"{'Rate':>7}"

    f" {'Events':>8}"

    f" {'Spikes':>10}"

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

        f" {result['stimulus']:>8}"

        f" {result['spikes']:>10}"

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
# ACTION TRANSITION ÖRNEKLERİ
# ============================================================

print("\n" + "=" * 110)
print("ACTION TRANSITION ÖRNEKLERİ")
print("=" * 110)


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


    for transition in transitions[:20]:

        brain_time_ms, old, new = transition


        print(

            f"{brain_time_ms:7.0f} ms"

            f" | {old}"

            f" -> {new}"

        )


    if len(transitions) > 20:

        print(
            f"... toplam {len(transitions)} transition"
        )


print("\n" + "=" * 80)
print("TEST BİTTİ")
print("=" * 80)