import json
import time
import urllib.request

from collections import deque

import mss
import numpy as np


# ============================================================
# MALECNS OBS VISION BRIDGE V3
#
# OBS Program
#     ↓
# low-resolution screen vision
#     ↓
# temporal motion-area expansion
#     ↓
# /trigger
#     ↓
# modeled LPLC2 + LC4 input
#     ↓
# MaleCNS
#
# SCIENTIFIC LIMIT:
# Bu gerçek Drosophila retina modeli değildir.
# Engineered visual preprocessing katmanıdır.
# ============================================================


print("=" * 95)
print("MALECNS OBS VISION BRIDGE V3")
print("TEMPORAL MOTION-AREA EXPANSION")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

STATE_URL = SERVER + "/state"

TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# DISPLAY
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# VISION
# ============================================================

VISION_FPS = 8.0

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
# BASIC MOTION GATE
# ============================================================

MIN_MOTION_MEAN = 1.5

MIN_ACTIVE_RATIO = 0.008


# ============================================================
# EXPANSION DETECTOR
#
# Component tracking YOK.
#
# Son motion-area değerlerine bakılır.
# ============================================================

AREA_HISTORY_FRAMES = 5


# Önceki birkaç frame'e göre hareket alanı
# belirgin büyümeli.

MIN_AREA_GROWTH = 1.45


# Motion mean de artmalı.

MIN_MOTION_GROWTH = 1.20


# En az bu active ratio'ya ulaşmalı.

MIN_CURRENT_ACTIVE_RATIO = 0.020


# ============================================================
# TEMPORAL CONFIRMATION
#
# Son 5 frame içinde 2 expansion hit.
# ============================================================

CONFIRM_WINDOW_FRAMES = 5

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# COOLDOWN
# ============================================================

TRIGGER_COOLDOWN_MS = 1600.0


# ============================================================
# STATE POLL
# ============================================================

STATE_POLL_INTERVAL = 0.25


# ============================================================
# FLY STATE
# ============================================================

fly_x = 0.50

fly_y = 0.92

last_state_poll = -1000.0

state_failures = 0


# ============================================================
# STATS
# ============================================================

trigger_count = 0


# ============================================================
# FLY POSITION
# ============================================================

def update_fly_position(now):

    global fly_x
    global fly_y

    global last_state_poll
    global state_failures


    if (
        now
        -
        last_state_poll
        <
        STATE_POLL_INTERVAL
    ):

        return


    last_state_poll = now


    try:

        with urllib.request.urlopen(
            STATE_URL,
            timeout=0.05
        ) as response:

            state = json.loads(
                response
                .read()
                .decode("utf-8")
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
# SEND VISUAL INPUT
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
# FRAME -> GRAYSCALE
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
# BRIGHTNESS COMPENSATION
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


    return (
        corrected,
        delta
    )


# ============================================================
# MASK FLY
# ============================================================

def mask_fly(
    active,
    difference
):

    height, width = (
        active.shape
    )


    center_x = int(
        round(
            fly_x
            *
            (
                width - 1
            )
        )
    )


    center_y = int(
        round(
            fly_y
            *
            (
                height - 1
            )
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


    x1 = max(
        0,
        center_x
        -
        mask_width // 2
    )


    x2 = min(
        width,
        center_x
        +
        mask_width // 2
    )


    y1 = max(
        0,
        center_y
        -
        mask_height // 2
    )


    y2 = min(
        height,
        center_y
        +
        mask_height // 2
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
    maxlen=
        AREA_HISTORY_FRAMES
)


motion_history = deque(
    maxlen=
        AREA_HISTORY_FRAMES
)


candidate_history = deque(
    maxlen=
        CONFIRM_WINDOW_FRAMES
)


# ============================================================
# PREVIOUS FRAME
# ============================================================

previous_gray = None


# ============================================================
# TRIGGER STATE
# ============================================================

last_trigger_ms = -1000000.0


# ============================================================
# TIMING
# ============================================================

frame_interval = (
    1.0
    /
    VISION_FPS
)


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


    if (
        MONITOR_INDEX
        >=
        len(
            screen_capture.monitors
        )
    ):

        raise RuntimeError(
            f"MONITOR_INDEX={MONITOR_INDEX} yok."
        )


    monitor = screen_capture.monitors[
        MONITOR_INDEX
    ]


    print()

    print("=" * 95)
    print("VISION BRIDGE V3 ACTIVE")
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
        monitor["width"] // DOWNSAMPLE,
        "x",
        monitor["height"] // DOWNSAMPLE
    )


    print(
        "FPS:",
        VISION_FPS
    )


    print(
        "Area history:",
        AREA_HISTORY_FRAMES,
        "frames"
    )


    print(
        "Confirm hits:",
        CONFIRM_HITS_REQUIRED,
        "/",
        CONFIRM_WINDOW_FRAMES
    )


    print(
        "Cooldown:",
        TRIGGER_COOLDOWN_MS,
        "ms"
    )


    print()

    print(
        "Confirmed expansion -> V4 /trigger"
    )


    print(
        "/trigger -> modeled LPLC2 + LC4 input"
    )


    print(
        "Vision FLY komutu vermez."
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
                        next_frame
                        -
                        now
                    )
                )

                continue


            frame_start = (
                time.perf_counter()
            )


            # =================================================
            # FLY POSITION
            # =================================================

            update_fly_position(
                now
            )


            # =================================================
            # CAPTURE
            # =================================================

            capture_start = (
                time.perf_counter()
            )


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

            process_start = (
                time.perf_counter()
            )


            current_gray = frame_to_gray(
                screenshot
            )


            # =================================================
            # FIRST FRAME
            # =================================================

            if previous_gray is None:

                previous_gray = (
                    current_gray
                )


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


                next_frame += (
                    frame_interval
                )


                continue


            # =================================================
            # BRIGHTNESS
            # =================================================

            (
                corrected,
                brightness_delta

            ) = compensate_brightness(

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
            # SELF MASK
            # =================================================

            mask_fly(
                active,
                difference
            )


            # =================================================
            # MOTION METRICS
            # =================================================

            motion_mean = float(
                difference.mean()
            )


            active_ratio = float(
                active.mean()
            )


            # =================================================
            # REFERENCE HISTORY
            #
            # Current frame'i eklemeden önce eski history
            # kullanılır.
            # =================================================

            if len(
                active_history
            ) >= 2:

                previous_active_values = np.asarray(
                    active_history,
                    dtype=np.float64
                )


                previous_motion_values = np.asarray(
                    motion_history,
                    dtype=np.float64
                )


                baseline_active = float(
                    np.median(
                        previous_active_values
                    )
                )


                baseline_motion = float(
                    np.median(
                        previous_motion_values
                    )
                )


            else:

                baseline_active = 0.0

                baseline_motion = 0.0


            # =================================================
            # SAFE DIVISORS
            # =================================================

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
            # EXPANSION CANDIDATE
            # =================================================

            candidate = (

                motion_mean
                >=
                MIN_MOTION_MEAN

                and

                active_ratio
                >=
                MIN_ACTIVE_RATIO

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
            # ADD HISTORY
            # =================================================

            active_history.append(
                active_ratio
            )


            motion_history.append(
                motion_mean
            )


            candidate_history.append(
                bool(
                    candidate
                )
            )


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
            # CONFIRM
            # =================================================

            confirmed = (

                len(
                    candidate_history
                )
                >=
                CONFIRM_WINDOW_FRAMES

                and

                hits
                >=
                CONFIRM_HITS_REQUIRED

                and

                cooldown_ok

            )


            # =================================================
            # SEND
            # =================================================

            if confirmed:

                sent = send_looming()


                if sent:

                    trigger_count += 1

                    last_trigger_ms = (
                        now_ms
                    )


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
                        f" | motion growth={motion_growth:.2f}x"
                    )


                    print(
                        f"active={active_ratio * 100:.2f}%"
                        f" | area growth={area_growth:.2f}x"
                    )


                    print(
                        f"temporal hits="
                        f"{hits}/"
                        f"{CONFIRM_WINDOW_FRAMES}"
                    )


                    print(
                        "Modeled LPLC2 + LC4 stimulus gönderildi."
                    )


                    print(
                        "Motor sonucu şimdi MaleCNS belirleyecek."
                    )


                    print("=" * 95)

                    print()


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
            # REPORT
            # =================================================

            if now >= next_report:

                candidate_text = (
                    "YES"
                    if candidate
                    else
                    "no"
                )


                print(

                    "vision"

                    f" | motion={motion_mean:6.2f}"

                    f" | active={active_ratio * 100:6.2f}%"

                    f" | baseA={baseline_active * 100:5.2f}%"

                    f" | areaX={area_growth:5.2f}"

                    f" | motionX={motion_growth:5.2f}"

                    f" | cand={candidate_text:3s}"

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


            # =================================================
            # SAVE PREVIOUS
            # =================================================

            previous_gray = (
                current_gray
            )


            # =================================================
            # NEXT FRAME
            # =================================================

            next_frame += (
                frame_interval
            )


    except KeyboardInterrupt:

        print()

        print(
            "Vision bridge V3 durduruldu."
        )


# ============================================================
# PERFORMANCE SUMMARY
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
    print("VISION V3 PERFORMANCE")
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


    duty = (
        total_array.mean()
        /
        frame_budget_ms
    )


    print()


    print(
        "Vision frame duty:",
        round(
            float(
                duty
                *
                100.0
            ),
            2
        ),
        "%"
    )


    print("=" * 95)