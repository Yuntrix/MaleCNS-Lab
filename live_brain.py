import time

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session import ContinuousBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# AYARLAR
# ============================================================

CHUNK_MS = 20.0

TOTAL_RUN_SECONDS = 6.0

LOOMING_RATE_HZ = 40.0

LOOMING_PULSE_MS = 60.0

TEST_EVENT_AT_SECONDS = 2.0

RUNAWAY_SPIKES_PER_CHUNK = 10_000


# ============================================================
# BRAIN
# ============================================================

brain = RealtimeMaleCNSLIF()


# ============================================================
# MALECNS İÇİN KALİBRE EDİLMİŞ GLOBAL SYNAPTIC GAIN
# ============================================================

brain.w_syn = 0.110


session = ContinuousBrainSession(

    brain,

    source_types=[
        "LPLC2",
        "LC4"
    ],

    seed=101
)


decoder = MotorDecoder()


# ============================================================
# WARMUP
# ============================================================

print("\n" + "=" * 70)
print("LIVE BRAIN WARMUP")
print("=" * 70)

print(
    "W_SYN:",
    brain.w_syn
)


session.step(
    rate_hz=0,
    chunk_ms=1.0
)


session.reset(
    seed=101
)


print("Compile tamam.")


# ============================================================
# EVENT STATE
# ============================================================

stimulus_remaining_ms = 0.0

test_event_sent = False


def trigger_looming():

    global stimulus_remaining_ms

    stimulus_remaining_ms = max(

        stimulus_remaining_ms,

        LOOMING_PULSE_MS

    )


    print(

        "\n>>> LOOMING EVENT GELDİ! "

        f"{LOOMING_PULSE_MS:.0f} ms sensory pulse başlatıldı."

    )


# ============================================================
# LIVE LOOP
# ============================================================

print("\n" + "=" * 70)
print("LIVE MALECNS BAŞLADI")
print("=" * 70)

print(
    "Chunk:",
    CHUNK_MS,
    "ms"
)

print(
    "Looming pulse:",
    LOOMING_RATE_HZ,
    "Hz x",
    LOOMING_PULSE_MS,
    "ms"
)

print(
    "Test event:",
    TEST_EVENT_AT_SECONDS,
    "s"
)


chunk_seconds = (

    CHUNK_MS

    /

    1000.0

)


start_time = time.perf_counter()

next_tick = start_time

chunk_number = 0

max_backlog_ms = 0.0

runaway = False


# ============================================================
# LOOP
# ============================================================

while True:

    wall_now = time.perf_counter()


    wall_elapsed = (

        wall_now

        -

        start_time

    )


    if wall_elapsed >= TOTAL_RUN_SECONDS:

        break


    # ========================================================
    # SAHTE KICK EVENT
    # ========================================================

    if (

        not test_event_sent

        and

        wall_elapsed >= TEST_EVENT_AT_SECONDS

    ):

        trigger_looming()

        test_event_sent = True


    # ========================================================
    # BEYNE VERİLECEK INPUT
    # ========================================================

    if stimulus_remaining_ms > 0:

        input_rate = LOOMING_RATE_HZ

    else:

        input_rate = 0.0


    # ========================================================
    # BRAIN
    # ========================================================

    compute_start = time.perf_counter()


    result = session.step(

        rate_hz=input_rate,

        chunk_ms=CHUNK_MS

    )


    compute_ms = (

        time.perf_counter()

        -

        compute_start

    ) * 1000.0


    # ========================================================
    # PULSE TIMER
    #
    # WALL CLOCK DEĞİL, BEYİN ZAMANI
    # ========================================================

    if stimulus_remaining_ms > 0:

        stimulus_remaining_ms -= CHUNK_MS


        stimulus_remaining_ms = max(

            0.0,

            stimulus_remaining_ms

        )


    # ========================================================
    # MOTOR DECODER
    # ========================================================

    movement = decoder.decode(

        neurons=
            brain.neurons,

        spike_counts=
            result["spike_counts"],

        motor_mask=
            brain.motor_mask

    )


    # ========================================================
    # RUNAWAY KONTROLÜ
    # ========================================================

    if (

        result["total_spikes"]

        >

        RUNAWAY_SPIKES_PER_CHUNK

    ):

        runaway = True


        print("\n" + "!" * 70)

        print(
            "RUNAWAY ALGILANDI!"
        )

        print(
            "Chunk spike:",
            result["total_spikes"]
        )

        print(
            "Test güvenlik nedeniyle durduruldu."
        )

        print(
            "!" * 70
        )

        break


    # ========================================================
    # PRINT
    # ========================================================

    should_print = (

        input_rate > 0

        or

        result["total_spikes"] > 0

        or

        chunk_number % 10 == 0

    )


    if should_print:

        brain_time_ms = (

            chunk_number

            *

            CHUNK_MS

        )


        print(

            f"wall {wall_elapsed:5.2f}s"

            f" | brain {brain_time_ms:7.0f} ms"

            f" | input {input_rate:4.0f} Hz"

            f" | spike {result['total_spikes']:5d}"

            f" | DN {result['descending_spikes']:4d}"

            f" | motor {result['motor_spikes']:4d}"

            f" | action {movement['action']:5s}"

            f" | flight {movement['flight_spikes']:3d}"

            f" | jump {movement['jump_spikes']:3d}"

            f" | calc {compute_ms:6.1f} ms"

        )


    # ========================================================
    # REAL-TIME PACING
    # ========================================================

    chunk_number += 1


    next_tick += (

        chunk_seconds

    )


    after_compute = time.perf_counter()


    sleep_time = (

        next_tick

        -

        after_compute

    )


    if sleep_time > 0:

        time.sleep(
            sleep_time
        )

    else:

        current_backlog_ms = (

            -sleep_time

            *

            1000.0

        )


        max_backlog_ms = max(

            max_backlog_ms,

            current_backlog_ms

        )


# ============================================================
# SONUÇ
# ============================================================

actual_duration = (

    time.perf_counter()

    -

    start_time

)


print("\n" + "=" * 70)
print("LIVE TEST BİTTİ")
print("=" * 70)


print(
    "W_SYN:",
    brain.w_syn
)


print(
    "Gerçek geçen süre:",
    round(
        actual_duration,
        3
    ),
    "s"
)


print(
    "Chunk sayısı:",
    chunk_number
)


print(
    "Max backlog:",
    round(
        max_backlog_ms,
        2
    ),
    "ms"
)


print(
    "Runaway:",
    runaway
)


if (

    not runaway

    and

    max_backlog_ms < 100

):

    print(
        "SONUÇ: LIVE BRAIN LOOP STABİL."
    )

else:

    print(
        "SONUÇ: Bir sonraki optimizasyon/test gerekiyor."
    )