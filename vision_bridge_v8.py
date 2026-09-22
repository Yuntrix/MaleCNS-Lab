import queue
import threading
import time
import urllib.request
from collections import deque

import mss
import numpy as np


# ============================================================
# MALECNS WORLD VISION BRIDGE V8
#
# LIGHTWEIGHT / PRODUCTION SENSOR
#
# WORLD Projector
#       ↓
# screen capture
#       ↓
# engineered visual motion / expansion detector
#       ↓
# asynchronous /trigger
#       ↓
# modeled LPLC2 + LC4 drive
#       ↓
# MaleCNS
#       ↓
# motor output
#
# IMPORTANT:
# - Vision doğrudan FLY komutu vermez.
# - Bu gerçek Drosophila retina modeli değildir.
# - Visual preprocessing engineering modelidir.
# ============================================================


print("=" * 95)
print("MALECNS WORLD VISION BRIDGE V8")
print("LIGHTWEIGHT PRODUCTION SENSOR")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# CORRECT MONITOR
#
# MSS:
#
# [0] virtual desktop
# [1] 1920x1080
# [2] 2560x1440
#
# Senin doğru ekranın artık [2].
# ============================================================

MONITOR_INDEX = 2


# ============================================================
# VISION RATE
#
# 8 FPS çalışan / stabil değere geri dönüyoruz.
# ============================================================

VISION_FPS = 8.0

FRAME_INTERVAL = 1.0 / VISION_FPS


# ============================================================
# TARGET PROCESSING SIZE
#
# Ekran kaç piksel olursa olsun yaklaşık 160 px genişlikte
# analiz edeceğiz.
#
# 2560 / 160 = 16
#
# Yani 2560x1440 -> yaklaşık 160x90.
# ============================================================

TARGET_PROCESS_WIDTH = 160


# ============================================================
# PIXEL MOTION
# ============================================================

MOTION_THRESHOLD = 20


# ============================================================
# NORMAL APPROACH DETECTOR
#
# V3'te çalışan temel değerler.
# ============================================================

MIN_MOTION_MEAN = 1.5

MIN_CURRENT_ACTIVE_RATIO = 0.020

MIN_AREA_GROWTH = 1.45

MIN_MOTION_GROWTH = 1.20


# ============================================================
# TEMPORAL CONFIRMATION
# ============================================================

HISTORY_FRAMES = 5

CONFIRM_WINDOW_FRAMES = 5

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# STARTUP
#
# Projector / scene ilk açıldığında frame değişimi olabilir.
# ============================================================

WARMUP_SECONDS = 2.0


# ============================================================
# SCENE CUT REJECTION
#
# Ekranın %40+'ı bir anda değişirse bu çoğu durumda:
#
# - pencere taşıma
# - alt-tab
# - scene change
# - projector geçişi
#
# gibi global bir değişimdir.
#
# Bunu looming saymıyoruz.
# ============================================================

SCENE_CUT_ACTIVE_RATIO = 0.40


# ============================================================
# TRIGGER REFRACTORY
#
# Tek visual event sırasında beyni spamlemiyoruz.
# ============================================================

TRIGGER_COOLDOWN_MS = 3000.0


# ============================================================
# QUIET RE-ARM
#
# Trigger sonrası sensör tekrar aktif olmak için birkaç
# sakin frame bekler.
# ============================================================

QUIET_FRAMES_REQUIRED = 4

QUIET_MAX_MOTION = 0.60

QUIET_MAX_ACTIVE_RATIO = 0.006


# ============================================================
# BACKGROUND TRIGGER QUEUE
#
# HTTP request vision loop'unu bloklamayacak.
#
# Queue size 1:
# server yavaşsa trigger biriktirmiyoruz.
# ============================================================

trigger_queue = queue.Queue(
    maxsize=1
)


# ============================================================
# STATS
# ============================================================

trigger_count = 0

trigger_success = 0

trigger_failures = 0

scene_cut_count = 0

dropped_trigger_count = 0


capture_sum_ms = 0.0

process_sum_ms = 0.0

total_sum_ms = 0.0

frame_count = 0


# ============================================================
# TRIGGER WORKER
# ============================================================

def trigger_worker():

    global trigger_success
    global trigger_failures


    while True:

        item = trigger_queue.get()


        if item is None:

            trigger_queue.task_done()

            break


        number = item["number"]

        region = item["region"]


        try:

            with urllib.request.urlopen(
                TRIGGER_URL,
                timeout=0.75
            ) as response:

                response.read()


            trigger_success += 1


            print(
                f"[MALECNS INPUT OK] "
                f"trigger #{number} "
                f"| region={region}"
            )


        except Exception as error:

            trigger_failures += 1


            print(
                f"[MALECNS INPUT FAIL] "
                f"trigger #{number} "
                f"| {error}"
            )


        finally:

            trigger_queue.task_done()


# ============================================================
# START WORKER
# ============================================================

worker = threading.Thread(
    target=trigger_worker,
    daemon=True
)

worker.start()


# ============================================================
# NON-BLOCKING TRIGGER
# ============================================================

def enqueue_trigger(
    number,
    region
):

    global dropped_trigger_count


    try:

        trigger_queue.put_nowait(
            {
                "number": number,
                "region": region
            }
        )


        return True


    except queue.Full:

        dropped_trigger_count += 1

        return False


# ============================================================
# FRAME -> LOW RES GRAYSCALE
# ============================================================

def frame_to_gray(
    screenshot,
    step
):

    frame = np.asarray(
        screenshot,
        dtype=np.uint8
    )


    small = frame[
        ::step,
        ::step,
        :3
    ]


    # MSS = BGRA

    blue = small[
        :,
        :,
        0
    ].astype(
        np.uint16
    )


    green = small[
        :,
        :,
        1
    ].astype(
        np.uint16
    )


    red = small[
        :,
        :,
        2
    ].astype(
        np.uint16
    )


    gray = (
        (
            77 * red
            +
            150 * green
            +
            29 * blue
        )
        >>
        8
    ).astype(
        np.uint8
    )


    return gray


# ============================================================
# GLOBAL BRIGHTNESS COMPENSATION
# ============================================================

def compensate_brightness(
    previous,
    current
):

    delta = float(
        np.median(
            current.astype(
                np.int16
            )
            -
            previous.astype(
                np.int16
            )
        )
    )


    corrected = (
        current.astype(
            np.float32
        )
        -
        delta
    )


    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(
        np.uint8
    )


    return corrected


# ============================================================
# LEFT / CENTER / RIGHT
#
# Şimdilik sadece debug information.
# Henüz MaleCNS'e ayrı left/right command göndermiyoruz.
# ============================================================

def dominant_region(
    active
):

    height, width = active.shape


    third = width // 3


    left = int(
        active[
            :,
            :third
        ].sum()
    )


    center = int(
        active[
            :,
            third:third * 2
        ].sum()
    )


    right = int(
        active[
            :,
            third * 2:
        ].sum()
    )


    counts = {
        "LEFT": left,
        "CENTER": center,
        "RIGHT": right
    }


    total = (
        left
        +
        center
        +
        right
    )


    if total <= 0:

        return (
            "NONE",
            0.0
        )


    region = max(
        counts,
        key=counts.get
    )


    dominance = (
        counts[region]
        /
        total
    )


    return (
        region,
        dominance
    )


# ============================================================
# HISTORY
# ============================================================

active_history = deque(
    maxlen=HISTORY_FRAMES
)


motion_history = deque(
    maxlen=HISTORY_FRAMES
)


candidate_history = deque(
    maxlen=CONFIRM_WINDOW_FRAMES
)


# ============================================================
# STATE
# ============================================================

previous_gray = None

last_trigger_ms = -1000000.0


armed = True

quiet_frames = 0


# ============================================================
# START CAPTURE
# ============================================================

with mss.MSS() as screen_capture:

    print()

    print(
        "Detected monitors:"
    )


    for index, monitor in enumerate(
        screen_capture.monitors
    ):

        print(
            f"[{index}] "
            f"{monitor['width']}x{monitor['height']} "
            f"left={monitor['left']} "
            f"top={monitor['top']}"
        )


    if MONITOR_INDEX >= len(
        screen_capture.monitors
    ):

        raise RuntimeError(
            f"MONITOR_INDEX={MONITOR_INDEX} yok."
        )


    monitor = screen_capture.monitors[
        MONITOR_INDEX
    ]


    monitor_width = int(
        monitor["width"]
    )


    monitor_height = int(
        monitor["height"]
    )


    # ========================================================
    # AUTO DOWNSAMPLE
    # ========================================================

    downsample = max(
        1,
        int(
            round(
                monitor_width
                /
                TARGET_PROCESS_WIDTH
            )
        )
    )


    processed_width = len(
        range(
            0,
            monitor_width,
            downsample
        )
    )


    processed_height = len(
        range(
            0,
            monitor_height,
            downsample
        )
    )


    print()

    print("=" * 95)
    print("VISION BRIDGE V8 ACTIVE")
    print("=" * 95)


    print(
        "Monitor:",
        MONITOR_INDEX
    )


    print(
        "Capture:",
        monitor_width,
        "x",
        monitor_height
    )


    print(
        "Downsample:",
        downsample
    )


    print(
        "Processed:",
        processed_width,
        "x",
        processed_height
    )


    print(
        "FPS:",
        VISION_FPS
    )


    print(
        "Scene-cut rejection:",
        f"{SCENE_CUT_ACTIVE_RATIO * 100:.0f}%"
    )


    print(
        "Cooldown:",
        TRIGGER_COOLDOWN_MS,
        "ms"
    )


    print(
        "Quiet re-arm:",
        QUIET_FRAMES_REQUIRED,
        "frames"
    )


    print()

    print(
        "Realtime PNG debug saving: OFF"
    )


    print(
        "HTTP trigger: ASYNC"
    )


    print(
        "Frame catch-up: OFF"
    )


    print()

    print(
        "Vision direct FLY vermez."
    )


    print("=" * 95)


    start_time = time.perf_counter()

    next_frame = start_time

    next_report = (
        start_time
        +
        1.0
    )


    try:

        while True:

            # =================================================
            # PACING
            # =================================================

            now = time.perf_counter()


            if now < next_frame:

                time.sleep(
                    next_frame
                    -
                    now
                )


            frame_start = time.perf_counter()


            # =================================================
            # CAPTURE
            # =================================================

            capture_start = time.perf_counter()


            screenshot = screen_capture.grab(
                monitor
            )


            capture_ms = (
                time.perf_counter()
                -
                capture_start
            ) * 1000.0


            # =================================================
            # PROCESS
            # =================================================

            process_start = time.perf_counter()


            current_gray = frame_to_gray(
                screenshot,
                downsample
            )


            # =================================================
            # FIRST FRAME
            # =================================================

            if previous_gray is None:

                previous_gray = current_gray


                active_history.append(
                    0.0
                )


                motion_history.append(
                    0.0
                )


                candidate_history.append(
                    False
                )


                process_ms = (
                    time.perf_counter()
                    -
                    process_start
                ) * 1000.0


                total_ms = (
                    time.perf_counter()
                    -
                    frame_start
                ) * 1000.0


                frame_count += 1

                capture_sum_ms += capture_ms

                process_sum_ms += process_ms

                total_sum_ms += total_ms


                next_frame = (
                    frame_start
                    +
                    FRAME_INTERVAL
                )


                continue


            # =================================================
            # BRIGHTNESS
            # =================================================

            corrected = compensate_brightness(
                previous_gray,
                current_gray
            )


            # =================================================
            # DIFFERENCE
            # =================================================

            difference = np.abs(
                corrected.astype(
                    np.int16
                )
                -
                previous_gray.astype(
                    np.int16
                )
            ).astype(
                np.uint8
            )


            active = (
                difference
                >=
                MOTION_THRESHOLD
            )


            # =================================================
            # METRICS
            # =================================================

            motion_mean = float(
                difference.mean()
            )


            active_ratio = float(
                active.mean()
            )


            (
                region,
                region_dominance
            ) = dominant_region(
                active
            )


            # =================================================
            # SCENE CUT
            # =================================================

            scene_cut = (
                active_ratio
                >=
                SCENE_CUT_ACTIVE_RATIO
            )


            if scene_cut:

                scene_cut_count += 1


                candidate_history.clear()

                active_history.clear()

                motion_history.clear()


                # Büyük scene change sonrası sensörü tekrar
                # sakinleşene kadar disarm ediyoruz.

                armed = False

                quiet_frames = 0


                previous_gray = current_gray


                process_ms = (
                    time.perf_counter()
                    -
                    process_start
                ) * 1000.0


                total_ms = (
                    time.perf_counter()
                    -
                    frame_start
                ) * 1000.0


                frame_count += 1

                capture_sum_ms += capture_ms

                process_sum_ms += process_ms

                total_sum_ms += total_ms


                print(
                    "vision"
                    f" | SCENE CUT"
                    f" | active={active_ratio * 100:5.1f}%"
                    f" | region={region}"
                )


                # =================================================
                # NO BACKLOG CATCH-UP
                # =================================================

                now_end = time.perf_counter()


                target_next = (
                    frame_start
                    +
                    FRAME_INTERVAL
                )


                if target_next <= now_end:

                    next_frame = (
                        now_end
                        +
                        FRAME_INTERVAL
                    )

                else:

                    next_frame = target_next


                continue


            # =================================================
            # BASELINE
            # =================================================

            if len(
                active_history
            ) >= 2:

                baseline_active = float(
                    np.median(
                        np.asarray(
                            active_history,
                            dtype=np.float64
                        )
                    )
                )


                baseline_motion = float(
                    np.median(
                        np.asarray(
                            motion_history,
                            dtype=np.float64
                        )
                    )
                )


            else:

                baseline_active = 0.0

                baseline_motion = 0.0


            active_reference = max(
                baseline_active,
                0.004
            )


            motion_reference = max(
                baseline_motion,
                1.0
            )


            area_growth = (
                active_ratio
                /
                active_reference
            )


            motion_growth = (
                motion_mean
                /
                motion_reference
            )


            # =================================================
            # QUIET REARM
            # =================================================

            quiet = (

                motion_mean
                <=
                QUIET_MAX_MOTION

                and

                active_ratio
                <=
                QUIET_MAX_ACTIVE_RATIO

            )


            if not armed:

                if quiet:

                    quiet_frames += 1

                else:

                    quiet_frames = 0


                if (
                    quiet_frames
                    >=
                    QUIET_FRAMES_REQUIRED
                ):

                    armed = True

                    quiet_frames = 0

                    candidate_history.clear()


                    print(
                        "[VISION] re-armed"
                    )


            # =================================================
            # CANDIDATE
            # =================================================

            candidate = (

                armed

                and

                motion_mean
                >=
                MIN_MOTION_MEAN

                and

                active_ratio
                >=
                MIN_CURRENT_ACTIVE_RATIO

                and

                area_growth
                >=
                MIN_AREA_GROWTH

                and

                motion_growth
                >=
                MIN_MOTION_GROWTH

            )


            # =================================================
            # HISTORY
            # =================================================

            active_history.append(
                active_ratio
            )


            motion_history.append(
                motion_mean
            )


            warmup_done = (

                frame_start
                -
                start_time

                >=

                WARMUP_SECONDS

            )


            if warmup_done:

                candidate_history.append(
                    bool(
                        candidate
                    )
                )


            else:

                candidate_history.clear()


            hits = int(
                sum(
                    candidate_history
                )
            )


            # =================================================
            # COOLDOWN
            # =================================================

            now_ms = (
                time.perf_counter()
                *
                1000.0
            )


            cooldown_ok = (

                now_ms
                -
                last_trigger_ms

                >=

                TRIGGER_COOLDOWN_MS

            )


            # =================================================
            # CONFIRM
            # =================================================

            confirmed = (

                warmup_done

                and

                armed

                and

                cooldown_ok

                and

                len(
                    candidate_history
                )
                >=
                CONFIRM_WINDOW_FRAMES

                and

                hits
                >=
                CONFIRM_HITS_REQUIRED

            )


            # =================================================
            # TRIGGER
            # =================================================

            if confirmed:

                trigger_count += 1


                queued = enqueue_trigger(
                    trigger_count,
                    region
                )


                if queued:

                    print()

                    print("=" * 95)

                    print(
                        ">>> [VISION -> MALECNS]"
                    )


                    print(
                        f"trigger #{trigger_count}"
                    )


                    print(
                        f"motion={motion_mean:.2f}"
                        f" | active="
                        f"{active_ratio * 100:.2f}%"
                    )


                    print(
                        f"areaX={area_growth:.2f}"
                        f" | motionX="
                        f"{motion_growth:.2f}"
                    )


                    print(
                        f"region={region}"
                        f" | dominance="
                        f"{region_dominance * 100:.1f}%"
                    )


                    print(
                        "Trigger queued asynchronously."
                    )


                    print(
                        "Vision direct FLY vermedi."
                    )


                    print("=" * 95)

                    print()


                    last_trigger_ms = now_ms


                    # Tek visual event'ten sonra yeniden
                    # sakinlik bekle.

                    armed = False

                    quiet_frames = 0


                candidate_history.clear()


            # =================================================
            # PERFORMANCE
            # =================================================

            process_ms = (
                time.perf_counter()
                -
                process_start
            ) * 1000.0


            total_ms = (
                time.perf_counter()
                -
                frame_start
            ) * 1000.0


            frame_count += 1

            capture_sum_ms += capture_ms

            process_sum_ms += process_ms

            total_sum_ms += total_ms


            # =================================================
            # STATUS
            # =================================================

            if frame_start >= next_report:

                print(
                    "vision"
                    f" | motion={motion_mean:6.2f}"
                    f" | active={active_ratio * 100:6.2f}%"
                    f" | areaX={area_growth:5.2f}"
                    f" | hits={hits}/{CONFIRM_WINDOW_FRAMES}"
                    f" | armed={'YES' if armed else 'no '}"
                    f" | region={region:6s}"
                    f" | cap={capture_ms:5.2f}ms"
                    f" | proc={process_ms:5.2f}ms"
                )


                next_report = (
                    frame_start
                    +
                    1.0
                )


            # =================================================
            # PREVIOUS FRAME
            # =================================================

            previous_gray = current_gray


            # =================================================
            # IMPORTANT:
            # NO CATCH-UP BURST
            #
            # Eğer bir frame gecikirse kaçırılan frame'leri
            # hızlı hızlı yetiştirmiyoruz.
            # =================================================

            now_end = time.perf_counter()


            target_next = (
                frame_start
                +
                FRAME_INTERVAL
            )


            if target_next <= now_end:

                next_frame = (
                    now_end
                    +
                    FRAME_INTERVAL
                )

            else:

                next_frame = target_next


    except KeyboardInterrupt:

        print()

        print(
            "Vision bridge V8 durduruldu."
        )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 95)
print("VISION V8 SUMMARY")
print("=" * 95)


print(
    "Frames:",
    frame_count
)


print(
    "Triggers queued:",
    trigger_count
)


print(
    "Triggers sent successfully:",
    trigger_success
)


print(
    "Trigger failures:",
    trigger_failures
)


print(
    "Dropped trigger requests:",
    dropped_trigger_count
)


print(
    "Scene cuts rejected:",
    scene_cut_count
)


if frame_count > 0:

    print()

    print(
        "Mean capture:",
        round(
            capture_sum_ms
            /
            frame_count,
            3
        ),
        "ms"
    )


    print(
        "Mean processing:",
        round(
            process_sum_ms
            /
            frame_count,
            3
        ),
        "ms"
    )


    print(
        "Mean total:",
        round(
            total_sum_ms
            /
            frame_count,
            3
        ),
        "ms"
    )


print("=" * 95)