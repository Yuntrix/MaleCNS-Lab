import json
import time
import urllib.request
from collections import deque

import mss
import numpy as np


# ============================================================
# MALECNS OBS VISION BRIDGE V5
#
# PROVEN V3 DETECTOR + LOW-LATENCY FAST PATH
#
# OBS Program
#     ↓
# engineered motion preprocessing
#     ↓
# strong approach OR temporal confirmation
#     ↓
# /trigger
#     ↓
# modeled LPLC2 + LC4 visual drive
#     ↓
# MaleCNS connectome + modeled LIF dynamics
#     ↓
# motor output
#
# IMPORTANT SCIENTIFIC LIMIT:
# Bu gerçek Drosophila retina modeli değildir.
# OBS görüntüsünü MaleCNS'e bağlayan engineered sensory
# preprocessing katmanıdır.
#
# Vision doğrudan FLY komutu VERMEZ.
# ============================================================


print("=" * 95)
print("MALECNS OBS VISION BRIDGE V5")
print("PROVEN V3 DETECTOR + FAST APPROACH PATH")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

STATE_URL = SERVER + "/state"
TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# MONITOR
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# VISION RATE
#
# V3'te çalışan değer.
# ============================================================

VISION_FPS = 8.0

FRAME_INTERVAL = 1.0 / VISION_FPS


# ============================================================
# DOWNSAMPLE
#
# 1920x1080 -> yaklaşık 160x90
# ============================================================

DOWNSAMPLE = 12


# ============================================================
# MOTION
# ============================================================

MOTION_THRESHOLD = 20


# ============================================================
# FLY SELF MASK
# ============================================================

FLY_MASK_WIDTH = 200
FLY_MASK_HEIGHT = 170


# ============================================================
# NORMAL APPROACH DETECTOR
#
# Bunlar çalışan V3 değerleri.
# ============================================================

NORMAL_MIN_MOTION = 1.5

NORMAL_MIN_ACTIVE_RATIO = 0.020

NORMAL_MIN_AREA_GROWTH = 1.45

NORMAL_MIN_MOTION_GROWTH = 1.20


# ============================================================
# FAST APPROACH PATH
#
# V3 gerçek testinde başarılı trigger'larda yaklaşık:
#
# motion: 5.49 / 9.07
# active: 9.47% / 11.06%
# area growth: 14.83x / 27.66x
#
# FAST eşikleri bunların belirgin altında ama idle seviyesinin
# çok üstünde tutulmuştur.
#
# Bunlar engineering thresholds'tur.
# ============================================================

FAST_MIN_MOTION = 4.0

FAST_MIN_ACTIVE_RATIO = 0.060

FAST_MIN_AREA_GROWTH = 8.0

FAST_MIN_MOTION_GROWTH = 3.0


# ============================================================
# HISTORY
# ============================================================

BASELINE_HISTORY_FRAMES = 5

CONFIRM_WINDOW_FRAMES = 5

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# COOLDOWN
# ============================================================

TRIGGER_COOLDOWN_MS = 1800.0


# ============================================================
# SERVER STATE
# ============================================================

STATE_POLL_INTERVAL = 0.25


fly_x = 0.50
fly_y = 0.92

last_state_poll = -1000.0

state_failures = 0


# ============================================================
# COUNTERS
# ============================================================

trigger_count = 0

fast_trigger_count = 0

confirmed_trigger_count = 0

last_trigger_ms = -1000000.0


# ============================================================
# LATENCY DEBUG
# ============================================================

ONSET_MIN_MOTION = 1.0

ONSET_MIN_ACTIVE = 0.010

QUIET_RESET_FRAMES = 4

motion_onset_ms = None

quiet_frames = 0


# ============================================================
# UPDATE FLY POSITION
# ============================================================

def update_fly_position(now):

    global fly_x
    global fly_y

    global last_state_poll
    global state_failures


    if now - last_state_poll < STATE_POLL_INTERVAL:
        return


    last_state_poll = now


    try:

        with urllib.request.urlopen(
            STATE_URL,
            timeout=0.05
        ) as response:

            state = json.loads(
                response.read().decode("utf-8")
            )


        fly_x = float(
            state.get(
                "x",
                fly_x
            )
        )


        fly_y = float(
            state.get(
                "y",
                fly_y
            )
        )


        fly_x = max(
            0.0,
            min(
                1.0,
                fly_x
            )
        )


        fly_y = max(
            0.0,
            min(
                1.0,
                fly_y
            )
        )


    except Exception:

        state_failures += 1


# ============================================================
# SEND VISUAL STIMULUS
# ============================================================

def send_looming():

    try:

        with urllib.request.urlopen(
            TRIGGER_URL,
            timeout=0.10
        ) as response:

            response.read()


        return True


    except Exception as error:

        print()
        print(
            "[VISION] Trigger error:",
            error
        )
        print()

        return False


# ============================================================
# SCREENSHOT -> GRAYSCALE
# ============================================================

def frame_to_gray(screenshot):

    frame = np.asarray(
        screenshot,
        dtype=np.uint8
    )


    small = frame[
        ::DOWNSAMPLE,
        ::DOWNSAMPLE,
        :3
    ]


    blue = small[:, :, 0].astype(
        np.uint16
    )

    green = small[:, :, 1].astype(
        np.uint16
    )

    red = small[:, :, 2].astype(
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

    difference = (
        current.astype(np.int16)
        -
        previous.astype(np.int16)
    )


    delta = float(
        np.median(
            difference
        )
    )


    corrected = (
        current.astype(np.float32)
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


    return corrected, delta


# ============================================================
# MASK FLY SPRITE
# ============================================================

def mask_fly(
    active,
    difference
):

    height, width = active.shape


    center_x = int(
        round(
            fly_x
            *
            (width - 1)
        )
    )


    center_y = int(
        round(
            fly_y
            *
            (height - 1)
        )
    )


    mask_width = max(
        2,
        int(
            round(
                FLY_MASK_WIDTH
                /
                DOWNSAMPLE
            )
        )
    )


    mask_height = max(
        2,
        int(
            round(
                FLY_MASK_HEIGHT
                /
                DOWNSAMPLE
            )
        )
    )


    half_width = mask_width // 2
    half_height = mask_height // 2


    x1 = max(
        0,
        center_x - half_width
    )

    x2 = min(
        width,
        center_x + half_width + 1
    )


    y1 = max(
        0,
        center_y - half_height
    )

    y2 = min(
        height,
        center_y + half_height + 1
    )


    active[
        y1:y2,
        x1:x2
    ] = False


    difference[
        y1:y2,
        x1:x2
    ] = 0


# ============================================================
# HISTORY
# ============================================================

active_history = deque(
    maxlen=BASELINE_HISTORY_FRAMES
)


motion_history = deque(
    maxlen=BASELINE_HISTORY_FRAMES
)


candidate_history = deque(
    maxlen=CONFIRM_WINDOW_FRAMES
)


# ============================================================
# PREVIOUS IMAGE
# ============================================================

previous_gray = None


# ============================================================
# PERFORMANCE
# ============================================================

capture_times = []

process_times = []

total_times = []


# ============================================================
# START
# ============================================================

with mss.MSS() as screen_capture:

    print()
    print("Detected monitors:")


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


    processed_width = len(
        range(
            0,
            monitor["width"],
            DOWNSAMPLE
        )
    )


    processed_height = len(
        range(
            0,
            monitor["height"],
            DOWNSAMPLE
        )
    )


    print()
    print("=" * 95)
    print("VISION BRIDGE V5 ACTIVE")
    print("=" * 95)

    print(
        "Monitor:",
        MONITOR_INDEX
    )

    print(
        "Capture:",
        monitor["width"],
        "x",
        monitor["height"]
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

    print()

    print("FAST PATH:")

    print(
        f"motion >= {FAST_MIN_MOTION}"
        f" | active >= {FAST_MIN_ACTIVE_RATIO * 100:.1f}%"
        f" | areaX >= {FAST_MIN_AREA_GROWTH:.1f}"
        f" | motionX >= {FAST_MIN_MOTION_GROWTH:.1f}"
    )

    print()

    print("NORMAL PATH:")

    print(
        f"{CONFIRM_HITS_REQUIRED}"
        f" / "
        f"{CONFIRM_WINDOW_FRAMES}"
        " candidate frames"
    )

    print()

    print(
        "Cooldown:",
        TRIGGER_COOLDOWN_MS,
        "ms"
    )

    print()

    print(
        "FAST veya confirmed approach -> /trigger"
    )

    print(
        "/trigger -> modeled LPLC2 + LC4 visual input"
    )

    print(
        "Vision doğrudan FLY komutu vermez."
    )

    print("=" * 95)


    next_frame = time.perf_counter()

    next_report = (
        time.perf_counter()
        +
        1.0
    )


    try:

        while True:

            now = time.perf_counter()


            # =================================================
            # FPS PACING
            # =================================================

            if now < next_frame:

                time.sleep(
                    min(
                        0.002,
                        next_frame - now
                    )
                )

                continue


            frame_start = time.perf_counter()


            # =================================================
            # STATE
            # =================================================

            update_fly_position(
                now
            )


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
                screenshot
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


                capture_times.append(
                    capture_ms
                )

                process_times.append(
                    process_ms
                )

                total_times.append(
                    total_ms
                )


                next_frame += FRAME_INTERVAL

                continue


            # =================================================
            # BRIGHTNESS COMPENSATION
            # =================================================

            corrected, brightness_delta = (
                compensate_brightness(
                    previous_gray,
                    current_gray
                )
            )


            # =================================================
            # FRAME DIFFERENCE
            # =================================================

            difference = np.abs(
                corrected.astype(np.int16)
                -
                previous_gray.astype(np.int16)
            ).astype(
                np.uint8
            )


            active = (
                difference
                >=
                MOTION_THRESHOLD
            )


            # =================================================
            # FLY MASK
            # =================================================

            mask_fly(
                active,
                difference
            )


            # =================================================
            # CURRENT MOTION
            # =================================================

            motion_mean = float(
                difference.mean()
            )


            active_ratio = float(
                active.mean()
            )


            # =================================================
            # PAST BASELINE
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
            # NORMAL CANDIDATE
            # =================================================

            normal_candidate = (

                motion_mean
                >=
                NORMAL_MIN_MOTION

                and

                active_ratio
                >=
                NORMAL_MIN_ACTIVE_RATIO

                and

                area_growth
                >=
                NORMAL_MIN_AREA_GROWTH

                and

                motion_growth
                >=
                NORMAL_MIN_MOTION_GROWTH

            )


            # =================================================
            # FAST CANDIDATE
            #
            # Strong looming-like visual event.
            # Temporal second hit beklenmez.
            # =================================================

            fast_candidate = (

                motion_mean
                >=
                FAST_MIN_MOTION

                and

                active_ratio
                >=
                FAST_MIN_ACTIVE_RATIO

                and

                area_growth
                >=
                FAST_MIN_AREA_GROWTH

                and

                motion_growth
                >=
                FAST_MIN_MOTION_GROWTH

            )


            # =================================================
            # MOTION ONSET / LATENCY
            # =================================================

            now_ms = (
                time.perf_counter()
                *
                1000.0
            )


            activity_present = (

                motion_mean
                >=
                ONSET_MIN_MOTION

                and

                active_ratio
                >=
                ONSET_MIN_ACTIVE

            )


            if activity_present:

                quiet_frames = 0


                if motion_onset_ms is None:

                    motion_onset_ms = now_ms


            else:

                quiet_frames += 1


                if quiet_frames >= QUIET_RESET_FRAMES:

                    motion_onset_ms = None


            # =================================================
            # TEMPORAL NORMAL CONFIRMATION
            # =================================================

            candidate_history.append(
                bool(
                    normal_candidate
                )
            )


            hits = int(
                sum(
                    candidate_history
                )
            )


            normal_confirmed = (

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
            # COOLDOWN
            # =================================================

            cooldown_remaining_ms = max(
                0.0,
                TRIGGER_COOLDOWN_MS
                -
                (
                    now_ms
                    -
                    last_trigger_ms
                )
            )


            cooldown_ok = (
                cooldown_remaining_ms
                <=
                0.0
            )


            # =================================================
            # SELECT PATH
            # =================================================

            trigger_mode = None


            if cooldown_ok:

                if fast_candidate:

                    trigger_mode = "FAST"


                elif normal_confirmed:

                    trigger_mode = "CONFIRMED"


            # =================================================
            # SEND
            # =================================================

            if trigger_mode is not None:

                sent = send_looming()


                if sent:

                    trigger_count += 1

                    last_trigger_ms = now_ms


                    if trigger_mode == "FAST":

                        fast_trigger_count += 1


                    else:

                        confirmed_trigger_count += 1


                    if motion_onset_ms is None:

                        latency_text = "n/a"


                    else:

                        latency_ms = (
                            now_ms
                            -
                            motion_onset_ms
                        )


                        latency_text = (
                            f"{latency_ms:.0f} ms"
                        )


                    print()
                    print("=" * 95)

                    print(
                        ">>> [VISION -> MALECNS]"
                    )

                    print(
                        f"trigger #{trigger_count}"
                        f" | mode={trigger_mode}"
                    )

                    print(
                        f"motion={motion_mean:.2f}"
                        f" | active="
                        f"{active_ratio * 100:.2f}%"
                    )

                    print(
                        f"area growth="
                        f"{area_growth:.2f}x"
                        f" | motion growth="
                        f"{motion_growth:.2f}x"
                    )

                    print(
                        "visual detection latency ≈",
                        latency_text
                    )

                    print(
                        "Modeled LPLC2 + LC4 stimulus gönderildi."
                    )

                    print(
                        "Motor sonucu MaleCNS belirleyecek."
                    )

                    print("=" * 95)
                    print()


                candidate_history.clear()


            # =================================================
            # UPDATE HISTORY
            # =================================================

            active_history.append(
                active_ratio
            )


            motion_history.append(
                motion_mean
            )


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


            capture_times.append(
                capture_ms
            )

            process_times.append(
                process_ms
            )

            total_times.append(
                total_ms
            )


            # =================================================
            # STATUS
            # =================================================

            if now >= next_report:

                if fast_candidate:

                    mode_text = "FAST"


                elif normal_candidate:

                    mode_text = "normal"


                else:

                    mode_text = "-"


                print(
                    "vision"
                    f" | motion={motion_mean:6.2f}"
                    f" | active={active_ratio * 100:6.2f}%"
                    f" | areaX={area_growth:6.2f}"
                    f" | motionX={motion_growth:5.2f}"
                    f" | mode={mode_text:6s}"
                    f" | hits={hits}/{CONFIRM_WINDOW_FRAMES}"
                    f" | cool={cooldown_remaining_ms:5.0f}ms"
                    f" | light={brightness_delta:+5.1f}"
                    f" | cap={capture_ms:5.2f}ms"
                    f" | proc={process_ms:5.2f}ms"
                )


                next_report = (
                    now
                    +
                    1.0
                )


            previous_gray = current_gray

            next_frame += FRAME_INTERVAL


    except KeyboardInterrupt:

        print()
        print(
            "Vision bridge V5 durduruldu."
        )


# ============================================================
# SUMMARY
# ============================================================

if len(
    capture_times
) > 0:

    capture_array = np.asarray(
        capture_times,
        dtype=np.float64
    )


    process_array = np.asarray(
        process_times,
        dtype=np.float64
    )


    total_array = np.asarray(
        total_times,
        dtype=np.float64
    )


    print()
    print("=" * 95)
    print("VISION V5 PERFORMANCE")
    print("=" * 95)


    print(
        "Frames:",
        len(
            capture_array
        )
    )


    print(
        "Triggers:",
        trigger_count
    )


    print(
        "FAST triggers:",
        fast_trigger_count
    )


    print(
        "Confirmed triggers:",
        confirmed_trigger_count
    )


    print(
        "State failures:",
        state_failures
    )


    print()


    print(
        "Mean capture:",
        round(
            float(
                capture_array.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 capture:",
        round(
            float(
                np.percentile(
                    capture_array,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    print()


    print(
        "Mean processing:",
        round(
            float(
                process_array.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 processing:",
        round(
            float(
                np.percentile(
                    process_array,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    print()


    print(
        "Mean total:",
        round(
            float(
                total_array.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 total:",
        round(
            float(
                np.percentile(
                    total_array,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    frame_budget_ms = (
        1000.0
        /
        VISION_FPS
    )


    frame_duty = (
        total_array.mean()
        /
        frame_budget_ms
        *
        100.0
    )


    print()


    print(
        "Vision frame duty:",
        round(
            float(
                frame_duty
            ),
            2
        ),
        "%"
    )


    print("=" * 95)