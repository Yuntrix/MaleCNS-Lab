import json
import math
import time
import urllib.request

import mss
import numpy as np

from scipy import ndimage


# ============================================================
# MALECNS OBS VISION BRIDGE
#
# OBS Program
#   -> low-resolution visual preprocessing
#   -> expansion / approach detection
#   -> V4 /trigger
#   -> LPLC2 + LC4 modeled input
#   -> MaleCNS
#
# IMPORTANT:
# This is engineered visual preprocessing.
# It is NOT a biological retina simulation.
# ============================================================


print("=" * 90)
print("MALECNS OBS VISION BRIDGE")
print("=" * 90)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

STATE_URL = SERVER + "/state"

TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# OBS PROJECTOR MONITOR
#
# Önceki sonuç:
#
# [1]
# left=2560
# top=0
# width=1920
# height=1080
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# SENSOR RATE
# ============================================================

VISION_FPS = 8.0


# ============================================================
# DOWNSAMPLE
#
# 1920x1080
#     ↓ /12
# yaklaşık
# 160x90
# ============================================================

DOWNSAMPLE = 12


# ============================================================
# MOTION
# ============================================================

MOTION_THRESHOLD = 20


# ============================================================
# FLY SELF MASK
#
# OBS Program sineği de içeriyor.
# Kendi sprite'ını visual motion saymayacağız.
#
# Original monitor pixels.
# ============================================================

FLY_MASK_WIDTH = 180

FLY_MASK_HEIGHT = 150


# ============================================================
# COMPONENT FILTER
#
# Küçük webcam noise bölgelerini at.
# ============================================================

MIN_COMPONENT_PIXELS = 18


# ============================================================
# APPROACH / EXPANSION
#
# Bunlar engineering thresholds.
# Biyolojik ölçüm değildir.
# ============================================================

MIN_MOTION_MEAN = 1.5

MIN_ACTIVE_RATIO = 0.002


# Aynı hareketli bölgenin alanının frame-to-frame büyümesi.

MIN_PIXEL_GROWTH = 1.12

MIN_BBOX_GROWTH = 1.10


# Lateral translation'ı looming sanmamak için
# component centroid çok fazla sıçramamalı.

MAX_CENTROID_SHIFT = 0.10


# En az kaç ardışık expansion frame?

CONFIRM_FRAMES = 2


# Aynı olayın sürekli stimulus spamlemesini engeller.

TRIGGER_COOLDOWN_MS = 1400.0


# ============================================================
# MORPHOLOGY
#
# Frame-difference hareketi çoğu zaman sadece cismin
# kenarlarını üretir.
#
# Ufak dilation bunları birleştirir.
# ============================================================

DILATION_ITERATIONS = 1


# ============================================================
# STATE POLL
#
# Fly position her vision frame'de HTTP ile çekilmek zorunda
# değil.
#
# 4 Hz yeterli.
# ============================================================

STATE_POLL_INTERVAL = 0.25


# ============================================================
# FALLBACK FLY STATE
# ============================================================

fly_x = 0.50
fly_y = 0.92

last_state_poll = -1000.0


# ============================================================
# GET FLY POSITION
# ============================================================

def update_fly_position(now):

    global fly_x
    global fly_y
    global last_state_poll


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

        pass


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

        print(
            "[VISION] Trigger error:",
            error
        )

        return False


# ============================================================
# SCREEN -> SMALL GRAYSCALE
# ============================================================

def frame_to_gray(
    screenshot
):

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
# GLOBAL BRIGHTNESS COMPENSATION
#
# Webcam auto exposure gibi tüm görüntüyü etkileyen
# parlaklık değişimlerini azaltır.
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
# FLY MASK
# ============================================================

def mask_fly(
    active_mask,
    difference
):

    height, width = (
        active_mask.shape
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


    active_mask[
        y1:y2,
        x1:x2
    ] = False


    difference[
        y1:y2,
        x1:x2
    ] = 0


# ============================================================
# LARGEST MOTION COMPONENT
# ============================================================

def largest_component(
    active
):

    # --------------------------------------------------------
    # Dilation
    # --------------------------------------------------------

    structure = np.ones(
        (
            3,
            3
        ),
        dtype=np.bool_
    )


    expanded = ndimage.binary_dilation(
        active,
        structure=structure,
        iterations=DILATION_ITERATIONS
    )


    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    labels, count = ndimage.label(
        expanded
    )


    if count <= 0:

        return None


    component_sizes = np.bincount(
        labels.ravel()
    )


    # label 0 = background

    if len(
        component_sizes
    ) <= 1:

        return None


    component_sizes[
        0
    ] = 0


    label_id = int(
        np.argmax(
            component_sizes
        )
    )


    component_pixels = int(
        component_sizes[
            label_id
        ]
    )


    if (
        component_pixels
        <
        MIN_COMPONENT_PIXELS
    ):

        return None


    ys, xs = np.nonzero(
        labels
        ==
        label_id
    )


    if len(xs) == 0:

        return None


    x_min = int(
        xs.min()
    )


    x_max = int(
        xs.max()
    )


    y_min = int(
        ys.min()
    )


    y_max = int(
        ys.max()
    )


    width = (
        x_max
        -
        x_min
        +
        1
    )


    height = (
        y_max
        -
        y_min
        +
        1
    )


    bbox_area = (
        width
        *
        height
    )


    centroid_x = float(
        xs.mean()
    )


    centroid_y = float(
        ys.mean()
    )


    return {

        "pixels":
            component_pixels,

        "bbox_area":
            bbox_area,

        "x_min":
            x_min,

        "x_max":
            x_max,

        "y_min":
            y_min,

        "y_max":
            y_max,

        "cx":
            centroid_x,

        "cy":
            centroid_y,

    }


# ============================================================
# MAIN
# ============================================================

previous_gray = None

previous_component = None


expansion_streak = 0


last_trigger_ms = -1000000.0


frame_interval = (
    1.0
    /
    VISION_FPS
)


# ============================================================
# PERFORMANCE STATS
# ============================================================

capture_times = []

process_times = []


# ============================================================
# START MSS
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

    print("=" * 90)
    print("VISION BRIDGE ACTIVE")
    print("=" * 90)

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
        "Processed yaklaşık:",
        math.ceil(
            monitor["width"]
            /
            DOWNSAMPLE
        ),
        "x",
        math.ceil(
            monitor["height"]
            /
            DOWNSAMPLE
        )
    )

    print(
        "FPS:",
        VISION_FPS
    )

    print()

    print(
        "Bu sefer vision GERÇEKTEN V4 /trigger'a bağlı."
    )

    print(
        "Confirmed approach -> modeled LPLC2/LC4 stimulus."
    )

    print()

    print(
        "Çıkmak için Ctrl+C."
    )

    print("=" * 90)


    next_frame = time.perf_counter()

    next_report = time.perf_counter() + 1.0


    try:

        while True:

            now = time.perf_counter()


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


            # =================================================
            # UPDATE FLY POSITION
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


            process_start = time.perf_counter()


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

                next_frame += (
                    frame_interval
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
            # REMOVE FLY
            # =================================================

            mask_fly(
                active,
                difference
            )


            # =================================================
            # GLOBAL MOTION
            # =================================================

            motion_mean = float(
                difference.mean()
            )


            active_ratio = float(
                active.mean()
            )


            # =================================================
            # LARGEST MOTION REGION
            # =================================================

            component = largest_component(
                active
            )


            pixel_growth = 1.0
            bbox_growth = 1.0
            centroid_shift = 1.0


            expansion_candidate = False


            if (
                component is not None
                and
                previous_component is not None
            ):

                previous_pixels = max(
                    1,
                    previous_component[
                        "pixels"
                    ]
                )


                previous_bbox = max(
                    1,
                    previous_component[
                        "bbox_area"
                    ]
                )


                pixel_growth = (
                    component[
                        "pixels"
                    ]
                    /
                    previous_pixels
                )


                bbox_growth = (
                    component[
                        "bbox_area"
                    ]
                    /
                    previous_bbox
                )


                frame_height, frame_width = (
                    active.shape
                )


                dx = (
                    component[
                        "cx"
                    ]
                    -
                    previous_component[
                        "cx"
                    ]
                ) / frame_width


                dy = (
                    component[
                        "cy"
                    ]
                    -
                    previous_component[
                        "cy"
                    ]
                ) / frame_height


                centroid_shift = math.sqrt(
                    dx * dx
                    +
                    dy * dy
                )


                expansion_candidate = (

                    motion_mean
                    >=
                    MIN_MOTION_MEAN

                    and

                    active_ratio
                    >=
                    MIN_ACTIVE_RATIO

                    and

                    pixel_growth
                    >=
                    MIN_PIXEL_GROWTH

                    and

                    bbox_growth
                    >=
                    MIN_BBOX_GROWTH

                    and

                    centroid_shift
                    <=
                    MAX_CENTROID_SHIFT

                )


            # =================================================
            # TEMPORAL CONFIRMATION
            # =================================================

            if expansion_candidate:

                expansion_streak += 1

            else:

                expansion_streak = max(
                    0,
                    expansion_streak - 1
                )


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


            confirmed = (

                expansion_streak
                >=
                CONFIRM_FRAMES

                and

                cooldown_ok

            )


            if confirmed:

                sent = send_looming()


                if sent:

                    last_trigger_ms = (
                        now_ms
                    )


                    print()

                    print(
                        ">>> [VISION -> MALECNS]"
                        f" motion={motion_mean:.2f}"
                        f" active={active_ratio * 100:.2f}%"
                        f" pixelsX={pixel_growth:.2f}"
                        f" bboxX={bbox_growth:.2f}"
                        f" shift={centroid_shift:.3f}"
                    )

                    print()


                expansion_streak = 0


            # =================================================
            # PERFORMANCE
            # =================================================

            process_ms = (
                time.perf_counter()
                -
                process_start
            ) * 1000.0


            capture_times.append(
                capture_ms
            )


            process_times.append(
                process_ms
            )


            # =================================================
            # REPORT
            # =================================================

            if now >= next_report:

                if component is None:

                    component_text = (
                        "none"
                    )

                else:

                    component_text = (

                        f"{component['pixels']:4d}px"

                    )


                print(

                    f"vision"

                    f" | motion={motion_mean:6.2f}"

                    f" | active={active_ratio * 100:6.2f}%"

                    f" | component={component_text:>8s}"

                    f" | pixelX={pixel_growth:5.2f}"

                    f" | bboxX={bbox_growth:5.2f}"

                    f" | shift={centroid_shift:5.3f}"

                    f" | streak={expansion_streak}"

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
            # PREVIOUS
            # =================================================

            previous_gray = (
                current_gray
            )


            previous_component = (
                component
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
            "Vision bridge durduruldu."
        )


# ============================================================
# FINAL PERFORMANCE
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


    print()

    print("=" * 90)
    print("VISION PERFORMANCE")
    print("=" * 90)

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

    print("=" * 90)