import json
import math
import time
import urllib.request

from collections import deque

import mss
import numpy as np

from scipy import ndimage


# ============================================================
# MALECNS OBS VISION BRIDGE V2
#
# OBS Program
#     ↓
# low-resolution visual preprocessing
#     ↓
# approach / expansion detection
#     ↓
# V4 /trigger
#     ↓
# modeled LPLC2 + LC4 input
#     ↓
# MaleCNS
#     ↓
# motor output
#
# IMPORTANT SCIENTIFIC LIMIT:
#
# Bu gerçek Drosophila retina biyofizik modeli değildir.
# OBS görüntüsünü connectome visual input'una bağlayan
# engineered sensory preprocessing katmanıdır.
# ============================================================


print("=" * 95)
print("MALECNS OBS VISION BRIDGE V2")
print("TEMPORAL APPROACH DETECTOR")
print("=" * 95)


# ============================================================
# MALECNS SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

STATE_URL = SERVER + "/state"

TRIGGER_URL = SERVER + "/trigger"


# ============================================================
# OBS FULLSCREEN PROJECTOR MONITOR
#
# Senin sisteminde:
#
# [1]
# 1920x1080
# left=2560
# top=0
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# VISION RATE
# ============================================================

VISION_FPS = 8.0


# ============================================================
# DOWNSAMPLE
#
# 1920 x 1080
#      ↓ /12
# yaklaşık
# 160 x 90
#
# Analiz maliyetini düşük tutar.
# ============================================================

DOWNSAMPLE = 12


# ============================================================
# MOTION THRESHOLD
# ============================================================

MOTION_THRESHOLD = 20


# ============================================================
# FLY SELF MASK
#
# OBS Program sineğin kendisini içerdiği için sprite'ın
# visual motion olarak algılanmasını azaltır.
#
# Original monitor pixels.
# ============================================================

FLY_MASK_WIDTH = 200

FLY_MASK_HEIGHT = 170


# ============================================================
# MOTION COMPONENT
# ============================================================

MIN_COMPONENT_PIXELS = 18


# ============================================================
# BASIC APPROACH THRESHOLDS
#
# Engineering thresholds.
# Bunlar biyolojik ölçüm değildir.
# ============================================================

MIN_MOTION_MEAN = 1.5

MIN_ACTIVE_RATIO = 0.002


# Hareketli component frame-to-frame büyümeli.

MIN_PIXEL_GROWTH = 1.12

MIN_BBOX_GROWTH = 1.10


# ============================================================
# CENTROID MOVEMENT
#
# V1: 0.10 idi.
#
# Gerçek yaklaşan obje kamerada tam sabit merkezden gelmez.
# Önceki testte güçlü approach sırasında 0.30 civarı
# displacement görüldü.
#
# Bu yüzden tolerans artırıldı.
# ============================================================

MAX_CENTROID_SHIFT = 0.35


# ============================================================
# TEMPORAL CONFIRMATION
#
# ARTIK:
#
# "tam art arda 2 frame"
#
# şartı yok.
#
# Son 4 vision frame içinde en az 2 güçlü expansion
# candidate varsa event doğrulanır.
#
# 8 FPS:
#
# 4 frame ≈ 500 ms
#
# ============================================================

CONFIRM_WINDOW_FRAMES = 4

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# COOLDOWN
#
# Tek yaklaşma sırasında MaleCNS'e sürekli stimulus
# göndermeyelim.
# ============================================================

TRIGGER_COOLDOWN_MS = 1400.0


# ============================================================
# MORPHOLOGY
# ============================================================

DILATION_ITERATIONS = 1


# ============================================================
# SERVER STATE POLL
#
# Fly konumunu her vision frame HTTP ile çekmeye gerek yok.
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
# STATE STATS
# ============================================================

state_failures = 0

trigger_count = 0


# ============================================================
# GET FLY POSITION
# ============================================================

def update_fly_position(
    now
):

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
# SEND MODELED VISUAL STIMULUS
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
# SCREEN -> LOW-RES GRAYSCALE
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


    # MSS format:
    #
    # B G R A

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
# Webcam auto-exposure veya tüm görüntünün parlaklık
# değişimini kısmen bastırır.
# ============================================================

def compensate_brightness(
    previous,
    current
):

    difference = (

        current.astype(
            np.int16
        )

        -

        previous.astype(
            np.int16
        )

    )


    delta = float(

        np.median(
            difference
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

    # ========================================================
    # SMALL DILATION
    #
    # Frame-difference çoğunlukla cismin kenarlarını verir.
    # Komşu kenarları biraz birleştiriyoruz.
    # ========================================================

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

        iterations=
            DILATION_ITERATIONS

    )


    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    labels, count = ndimage.label(
        expanded
    )


    if count <= 0:

        return None


    component_sizes = np.bincount(
        labels.ravel()
    )


    if len(
        component_sizes
    ) <= 1:

        return None


    # Label 0 = background

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


    if len(
        xs
    ) == 0:

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


    bbox_width = (

        x_max

        -

        x_min

        +

        1

    )


    bbox_height = (

        y_max

        -

        y_min

        +

        1

    )


    bbox_area = (

        bbox_width

        *

        bbox_height

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
# APPROACH CLASSIFICATION
# ============================================================

def classify_approach(
    motion_mean,
    active_ratio,
    component,
    previous_component,
    frame_width,
    frame_height
):

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    pixel_growth = 1.0

    bbox_growth = 1.0

    centroid_shift = 1.0

    candidate = False


    # ========================================================
    # NEED TWO COMPONENTS
    # ========================================================

    if (

        component is None

        or

        previous_component is None

    ):

        return (

            candidate,

            pixel_growth,

            bbox_growth,

            centroid_shift

        )


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


    # ========================================================
    # GROWTH
    # ========================================================

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


    # ========================================================
    # CENTROID SHIFT
    # ========================================================

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


    # ========================================================
    # APPROACH CANDIDATE
    # ========================================================

    candidate = (

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


    return (

        bool(
            candidate
        ),

        float(
            pixel_growth
        ),

        float(
            bbox_growth
        ),

        float(
            centroid_shift
        )

    )


# ============================================================
# TEMPORAL HISTORY
# ============================================================

candidate_history = deque(

    maxlen=
        CONFIRM_WINDOW_FRAMES

)


# ============================================================
# PREVIOUS FRAME STATE
# ============================================================

previous_gray = None

previous_component = None


# ============================================================
# COOLDOWN
# ============================================================

last_trigger_ms = -1000000.0


# ============================================================
# FRAME TIMING
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

    print("=" * 95)
    print("VISION BRIDGE V2 ACTIVE")
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


    print(
        "Confirmation window:",
        CONFIRM_WINDOW_FRAMES,
        "frames"
    )


    print(
        "Required hits:",
        CONFIRM_HITS_REQUIRED
    )


    print(
        "Max centroid shift:",
        MAX_CENTROID_SHIFT
    )


    print(
        "Cooldown:",
        TRIGGER_COOLDOWN_MS,
        "ms"
    )


    print()

    print(
        "Confirmed approach -> V4 /trigger"
    )


    print(
        "/trigger -> modeled LPLC2 + LC4 input"
    )


    print(
        "FLY komutu vision tarafından doğrudan verilmez."
    )


    print()

    print(
        "Çıkmak için Ctrl+C."
    )


    print("=" * 95)


    # ========================================================
    # LOOP TIMERS
    # ========================================================

    next_frame = (
        time.perf_counter()
    )


    next_report = (

        time.perf_counter()

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
            # SCREEN CAPTURE
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
            # PROCESS START
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


                total_times.append(

                    (
                        time.perf_counter()

                        -

                        frame_start
                    )

                    *

                    1000.0

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
            # FRAME DIFFERENCE
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
            # REMOVE FLY SPRITE
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
            # LARGEST MOVING REGION
            # =================================================

            component = largest_component(
                active
            )


            frame_height, frame_width = (
                active.shape
            )


            # =================================================
            # APPROACH
            # =================================================

            (

                candidate,

                pixel_growth,

                bbox_growth,

                centroid_shift

            ) = classify_approach(

                motion_mean=
                    motion_mean,

                active_ratio=
                    active_ratio,

                component=
                    component,

                previous_component=
                    previous_component,

                frame_width=
                    frame_width,

                frame_height=
                    frame_height

            )


            # =================================================
            # TEMPORAL WINDOW
            # =================================================

            candidate_history.append(
                candidate
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
            # CONFIRMATION
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
            # SEND TO MALECNS
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

                        f" | active={active_ratio * 100:.2f}%"

                    )


                    print(

                        f"pixel growth={pixel_growth:.2f}x"

                        f" | bbox growth={bbox_growth:.2f}x"

                    )


                    print(

                        f"centroid shift={centroid_shift:.3f}"

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

                        "Şimdi motor kararı MaleCNS'den gelecek."

                    )


                    print("=" * 95)

                    print()


                # =================================================
                # CLEAR TEMPORAL HISTORY
                #
                # Aynı expansion frame'leri tekrar kullanmayalım.
                # =================================================

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

                if component is None:

                    component_text = (
                        "none"
                    )


                else:

                    component_text = (

                        f"{component['pixels']}px"

                    )


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

                    f" | comp={component_text:>8s}"

                    f" | pxX={pixel_growth:5.2f}"

                    f" | boxX={bbox_growth:5.2f}"

                    f" | shift={centroid_shift:5.3f}"

                    f" | cand={candidate_text:3s}"

                    f" | hits={hits}/{CONFIRM_WINDOW_FRAMES}"

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
            # PREVIOUS STATE
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
            "Vision bridge V2 durduruldu."
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
    print("VISION V2 PERFORMANCE")
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