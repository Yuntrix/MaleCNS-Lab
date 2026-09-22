import os
import time
import urllib.request
from collections import deque
from datetime import datetime

import mss
from mss.tools import to_png
import numpy as np


# ============================================================
# MALECNS WORLD VISION BRIDGE V7
#
# CLEAN WORLD PROJECTOR
#     ↓
# low-resolution motion preprocessing
#     ↓
# approach / expansion detector
#     ↓
# V4 /trigger
#     ↓
# modeled LPLC2 + LC4 stimulus
#     ↓
# MaleCNS
#
# DEBUG:
# Trigger olduğunda:
#
# vision_debug/
#   trigger_001_before.png
#   trigger_001_after.png
#   trigger_001_motion.png
#   trigger_001_meta.txt
#
# IMPORTANT:
# WORLD sahnesinde sinek yok.
# Bu yüzden artık fly self-mask YOK.
#
# Vision doğrudan FLY komutu vermez.
# ============================================================


print("=" * 95)
print("MALECNS WORLD VISION BRIDGE V7")
print("TRIGGER SNAPSHOT + LEFT/CENTER/RIGHT DEBUG")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# WORLD PROJECTOR MONITOR
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# WORKING V3 SETTINGS
# ============================================================

VISION_FPS = 8.0

FRAME_INTERVAL = 1.0 / VISION_FPS

DOWNSAMPLE = 12

MOTION_THRESHOLD = 20


# ============================================================
# APPROACH DETECTOR
#
# Proven V3 baseline.
# ============================================================

MIN_MOTION_MEAN = 1.5

MIN_ACTIVE_RATIO = 0.008

MIN_CURRENT_ACTIVE_RATIO = 0.020

MIN_AREA_GROWTH = 1.45

MIN_MOTION_GROWTH = 1.20


# ============================================================
# TEMPORAL CONFIRMATION
# ============================================================

AREA_HISTORY_FRAMES = 5

CONFIRM_WINDOW_FRAMES = 5

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# COOLDOWN
# ============================================================

TRIGGER_COOLDOWN_MS = 1600.0


# ============================================================
# STARTUP WARMUP
#
# Scene projector açılırken büyük frame değişimi olabilir.
# İlk 2 saniye trigger göndermiyoruz.
# ============================================================

WARMUP_SECONDS = 2.0


# ============================================================
# DEBUG OUTPUT
# ============================================================

DEBUG_FOLDER = "vision_debug"

os.makedirs(
    DEBUG_FOLDER,
    exist_ok=True
)


# ============================================================
# STATS
# ============================================================

trigger_count = 0

last_trigger_ms = -1000000.0


# ============================================================
# SEND STIMULUS
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
# SCREENSHOT -> LOW RES GRAYSCALE
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


    # MSS = BGRA

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

    delta = float(
        np.median(
            current.astype(np.int16)
            -
            previous.astype(np.int16)
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
# DOMINANT MOTION REGION
#
# WORLD görüntüsü:
#
# LEFT | CENTER | RIGHT
# ============================================================

def dominant_region(active):

    height, width = active.shape


    third = width // 3


    left_count = int(
        active[
            :,
            0:third
        ].sum()
    )


    center_count = int(
        active[
            :,
            third:third * 2
        ].sum()
    )


    right_count = int(
        active[
            :,
            third * 2:width
        ].sum()
    )


    counts = {
        "LEFT": left_count,
        "CENTER": center_count,
        "RIGHT": right_count
    }


    region = max(
        counts,
        key=counts.get
    )


    total = (
        left_count
        +
        center_count
        +
        right_count
    )


    if total <= 0:

        dominance = 0.0

    else:

        dominance = (
            counts[region]
            /
            total
        )


    return (
        region,
        dominance,
        counts
    )


# ============================================================
# SAVE MOTION MASK
# ============================================================

def save_motion_image(
    active,
    filename
):

    # Low-res mask:
    # active = white
    # inactive = black

    mask = (
        active.astype(
            np.uint8
        )
        *
        255
    )


    # 160x90 çok küçük.
    # Görmek kolay olsun diye büyüt.

    scale = 6


    large = np.repeat(
        np.repeat(
            mask,
            scale,
            axis=0
        ),
        scale,
        axis=1
    )


    rgb = np.stack(
        [
            large,
            large,
            large
        ],
        axis=2
    )


    height, width, _ = rgb.shape


    to_png(
        rgb.tobytes(),
        (
            width,
            height
        ),
        output=filename
    )


# ============================================================
# SAVE TRIGGER DEBUG
# ============================================================

def save_trigger_debug(
    trigger_number,
    previous_rgb,
    current_rgb,
    monitor_width,
    monitor_height,
    active,
    motion_mean,
    active_ratio,
    area_growth,
    motion_growth,
    brightness_delta,
    region,
    dominance,
    region_counts
):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )


    prefix = os.path.join(
        DEBUG_FOLDER,
        f"trigger_{trigger_number:03d}_{timestamp}"
    )


    before_file = (
        prefix
        +
        "_before.png"
    )


    after_file = (
        prefix
        +
        "_after.png"
    )


    motion_file = (
        prefix
        +
        "_motion.png"
    )


    meta_file = (
        prefix
        +
        "_meta.txt"
    )


    # ========================================================
    # FULL RES BEFORE
    # ========================================================

    if previous_rgb is not None:

        to_png(
            previous_rgb,
            (
                monitor_width,
                monitor_height
            ),
            output=before_file
        )


    # ========================================================
    # FULL RES AFTER
    # ========================================================

    to_png(
        current_rgb,
        (
            monitor_width,
            monitor_height
        ),
        output=after_file
    )


    # ========================================================
    # MOTION MAP
    # ========================================================

    save_motion_image(
        active,
        motion_file
    )


    # ========================================================
    # METADATA
    # ========================================================

    with open(
        meta_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"trigger={trigger_number}\n"
        )

        file.write(
            f"motion_mean={motion_mean:.4f}\n"
        )

        file.write(
            f"active_ratio={active_ratio:.6f}\n"
        )

        file.write(
            f"active_percent={active_ratio * 100:.3f}\n"
        )

        file.write(
            f"area_growth={area_growth:.4f}\n"
        )

        file.write(
            f"motion_growth={motion_growth:.4f}\n"
        )

        file.write(
            f"brightness_delta={brightness_delta:.4f}\n"
        )

        file.write(
            f"dominant_region={region}\n"
        )

        file.write(
            f"dominance={dominance:.4f}\n"
        )

        file.write(
            f"left_pixels={region_counts['LEFT']}\n"
        )

        file.write(
            f"center_pixels={region_counts['CENTER']}\n"
        )

        file.write(
            f"right_pixels={region_counts['RIGHT']}\n"
        )


    return (
        before_file,
        after_file,
        motion_file,
        meta_file
    )


# ============================================================
# HISTORY
# ============================================================

active_history = deque(
    maxlen=AREA_HISTORY_FRAMES
)


motion_history = deque(
    maxlen=AREA_HISTORY_FRAMES
)


candidate_history = deque(
    maxlen=CONFIRM_WINDOW_FRAMES
)


# ============================================================
# PREVIOUS FRAME
# ============================================================

previous_gray = None

previous_rgb = None


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


    print()
    print("=" * 95)
    print("WORLD VISION BRIDGE V7 ACTIVE")
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
        "Processed yaklaşık:",
        len(
            range(
                0,
                monitor_width,
                DOWNSAMPLE
            )
        ),
        "x",
        len(
            range(
                0,
                monitor_height,
                DOWNSAMPLE
            )
        )
    )

    print(
        "FPS:",
        VISION_FPS
    )

    print(
        "Warmup:",
        WARMUP_SECONDS,
        "seconds"
    )

    print()

    print(
        "WORLD sahnesinde fly mask YOK."
    )

    print(
        "Trigger olduğunda debug PNG'leri kaydedilecek."
    )

    print(
        "Folder:",
        DEBUG_FOLDER
    )

    print()

    print(
        "Vision direct FLY vermez."
    )

    print(
        "Confirmed visual event -> modeled LPLC2 + LC4."
    )

    print("=" * 95)


    start_time = (
        time.perf_counter()
    )


    next_frame = start_time

    next_report = (
        start_time
        +
        1.0
    )


    try:

        while True:

            now = (
                time.perf_counter()
            )


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


            frame_start = (
                time.perf_counter()
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


            current_rgb = screenshot.rgb


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

                previous_rgb = (
                    current_rgb
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


                next_frame += (
                    FRAME_INTERVAL
                )

                continue


            # =================================================
            # BRIGHTNESS COMPENSATION
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
            # CURRENT METRICS
            # =================================================

            motion_mean = float(
                difference.mean()
            )


            active_ratio = float(
                active.mean()
            )


            # =================================================
            # OLD BASELINE
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
            # APPROACH CANDIDATE
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
            # HISTORY
            # =================================================

            active_history.append(
                active_ratio
            )


            motion_history.append(
                motion_mean
            )


            # =================================================
            # WARMUP
            # =================================================

            warmup_done = (

                now
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
            # CONFIRMATION
            # =================================================

            confirmed = (

                warmup_done

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

                and

                cooldown_ok

            )


            # =================================================
            # DOMINANT REGION
            # =================================================

            (
                region,
                dominance,
                region_counts

            ) = dominant_region(
                active
            )


            # =================================================
            # TRIGGER
            # =================================================

            if confirmed:

                sent = send_looming()


                if sent:

                    trigger_count += 1

                    last_trigger_ms = (
                        now_ms
                    )


                    (
                        before_file,
                        after_file,
                        motion_file,
                        meta_file

                    ) = save_trigger_debug(

                        trigger_number=
                            trigger_count,

                        previous_rgb=
                            previous_rgb,

                        current_rgb=
                            current_rgb,

                        monitor_width=
                            monitor_width,

                        monitor_height=
                            monitor_height,

                        active=
                            active,

                        motion_mean=
                            motion_mean,

                        active_ratio=
                            active_ratio,

                        area_growth=
                            area_growth,

                        motion_growth=
                            motion_growth,

                        brightness_delta=
                            brightness_delta,

                        region=
                            region,

                        dominance=
                            dominance,

                        region_counts=
                            region_counts

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
                        f"DOMINANT REGION = {region}"
                        f" ({dominance * 100:.1f}%)"
                    )

                    print()

                    print(
                        "DEBUG SAVED:"
                    )

                    print(
                        before_file
                    )

                    print(
                        after_file
                    )

                    print(
                        motion_file
                    )

                    print(
                        meta_file
                    )

                    print()

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
                    f" | areaX={area_growth:6.2f}"
                    f" | motionX={motion_growth:5.2f}"
                    f" | cand={candidate_text:3s}"
                    f" | hits={hits}/{CONFIRM_WINDOW_FRAMES}"
                    f" | region={region:6s}"
                    f" | cool={cooldown_remaining_ms:5.0f}ms"
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

            previous_rgb = (
                current_rgb
            )


            next_frame += (
                FRAME_INTERVAL
            )


    except KeyboardInterrupt:

        print()
        print(
            "Vision bridge V7 durduruldu."
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
    print("VISION V7 PERFORMANCE")
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
        "Mean total:",
        round(
            float(
                total_array.mean()
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
        *
        100.0
    )


    print(
        "Vision frame duty:",
        round(
            float(
                duty
            ),
            2
        ),
        "%"
    )

    print("=" * 95)