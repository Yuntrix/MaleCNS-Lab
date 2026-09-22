import json
import time
import urllib.request

import mss
import numpy as np


# ============================================================
# MALECNS LIGHTWEIGHT SCREEN VISION V2
#
# V2:
# - ego-motion compensation
# - correct fly self-mask
# - quantized ROI
# - confirmed looming detection
# - performance benchmark
#
# Brain'e henüz stimulus göndermez.
# ============================================================

print("=" * 95)
print("MALECNS LIGHTWEIGHT SCREEN VISION SENSOR V2")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

STATE_URL = "http://127.0.0.1:8765/state"


# ============================================================
# DISPLAY
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# SENSOR RATE
# ============================================================

SENSOR_FPS = 8.0

TEST_SECONDS = 30.0


# ============================================================
# ROI
#
# DOWNSAMPLE ile tam bölünecek ölçüler.
# ============================================================

ROI_WIDTH = 480

ROI_HEIGHT = 272

DOWNSAMPLE = 4


# ============================================================
# MOTION
# ============================================================

MOTION_PIXEL_THRESHOLD = 18


# ============================================================
# SELF MASK
#
# Cartoon fly + biraz çevresi.
# ============================================================

SELF_MASK_WIDTH = 130

SELF_MASK_HEIGHT = 100


# ============================================================
# LOOMING ENGINEERING THRESHOLDS
#
# Bunlar biyolojik ölçüm değildir.
# Lightweight visual preprocessing eşikleridir.
# ============================================================

MIN_ACTIVE_RATIO = 0.025

MIN_BBOX_GROWTH = 0.020

MIN_CENTER_MOTION = 5.0


# Tek karelik spike yerine iki ardışık
# expansion frame istiyoruz.

LOOMING_CONFIRM_FRAMES = 2


# Bir looming onaylandıktan sonra tekrar alarm için bekleme.

LOOMING_COOLDOWN_MS = 1000.0


# ============================================================
# FALLBACK FLY POSITION
# ============================================================

last_fly_x = 0.50

last_fly_y = 0.50


# ============================================================
# GET SERVER STATE
# ============================================================

def get_fly_position():

    global last_fly_x
    global last_fly_y


    start = time.perf_counter()


    success = False


    try:

        with urllib.request.urlopen(

            STATE_URL,

            timeout=0.05

        ) as response:

            data = json.loads(

                response
                .read()
                .decode("utf-8")

            )


        x = float(

            data.get(
                "x",
                last_fly_x
            )

        )


        y = float(

            data.get(
                "y",
                last_fly_y
            )

        )


        x = max(
            0.0,
            min(
                1.0,
                x
            )
        )


        y = max(
            0.0,
            min(
                1.0,
                y
            )
        )


        last_fly_x = x

        last_fly_y = y


        success = True


    except Exception:

        pass


    fetch_ms = (

        time.perf_counter()

        -

        start

    ) * 1000.0


    return (

        last_fly_x,

        last_fly_y,

        fetch_ms,

        success

    )


# ============================================================
# ALIGN VALUE TO DOWNSAMPLE GRID
# ============================================================

def align_coordinate(
    value,
    minimum,
    maximum,
    origin
):

    value = max(
        minimum,
        min(
            maximum,
            value
        )
    )


    max_aligned = (

        origin

        +

        (
            (
                maximum
                -
                origin
            )

            //
            DOWNSAMPLE

        )

        *
        DOWNSAMPLE

    )


    offset = (

        value

        -

        origin

    )


    aligned = (

        origin

        +

        (
            offset
            //
            DOWNSAMPLE
        )

        *
        DOWNSAMPLE

    )


    aligned = max(
        minimum,
        min(
            max_aligned,
            aligned
        )
    )


    return int(
        aligned
    )


# ============================================================
# BUILD CAPTURE REGION
# ============================================================

def build_capture_region(
    monitor,
    normalized_x,
    normalized_y
):

    monitor_left = int(
        monitor["left"]
    )


    monitor_top = int(
        monitor["top"]
    )


    monitor_width = int(
        monitor["width"]
    )


    monitor_height = int(
        monitor["height"]
    )


    # ========================================================
    # ACTUAL FLY SCREEN PIXEL
    # ========================================================

    fly_screen_x = (

        monitor_left

        +

        int(
            normalized_x
            *
            monitor_width
        )

    )


    fly_screen_y = (

        monitor_top

        +

        int(
            normalized_y
            *
            monitor_height
        )

    )


    desired_left = (

        fly_screen_x

        -

        ROI_WIDTH // 2

    )


    desired_top = (

        fly_screen_y

        -

        ROI_HEIGHT // 2

    )


    min_left = (
        monitor_left
    )


    max_left = (

        monitor_left

        +

        monitor_width

        -

        ROI_WIDTH

    )


    min_top = (
        monitor_top
    )


    max_top = (

        monitor_top

        +

        monitor_height

        -

        ROI_HEIGHT

    )


    # ========================================================
    # QUANTIZE ROI POSITION
    #
    # Çok önemli:
    # ROI her hareket ettiğinde DOWNSAMPLE grid'i aynı kalır.
    # Böylece previous/current görüntüleri doğru hizalayabiliriz.
    # ========================================================

    left = align_coordinate(

        desired_left,

        min_left,

        max_left,

        monitor_left

    )


    top = align_coordinate(

        desired_top,

        min_top,

        max_top,

        monitor_top

    )


    # ========================================================
    # FLY POSITION INSIDE ROI
    #
    # Artık self mask körlemesine ROI ortasında değil.
    # ========================================================

    fly_local_x = (

        fly_screen_x

        -

        left

    )


    fly_local_y = (

        fly_screen_y

        -

        top

    )


    region = {

        "left":
            int(
                left
            ),

        "top":
            int(
                top
            ),

        "width":
            ROI_WIDTH,

        "height":
            ROI_HEIGHT,

    }


    return (

        region,

        fly_local_x,

        fly_local_y

    )


# ============================================================
# FRAME -> SMALL GRAYSCALE
# ============================================================

def process_frame(
    screenshot
):

    frame = np.asarray(

        screenshot,

        dtype=np.uint8

    )


    # BGRA -> downsampled BGR

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
# SELF MASK
# ============================================================

def apply_self_mask(
    difference,
    valid_mask,
    fly_local_x,
    fly_local_y
):

    height, width = (
        difference.shape
    )


    center_x = int(

        round(

            fly_local_x

            /

            DOWNSAMPLE

        )

    )


    center_y = int(

        round(

            fly_local_y

            /

            DOWNSAMPLE

        )

    )


    mask_width = max(

        1,

        int(

            round(

                SELF_MASK_WIDTH

                /

                DOWNSAMPLE

            )

        )

    )


    mask_height = max(

        1,

        int(

            round(

                SELF_MASK_HEIGHT

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


    difference[
        y1:y2,
        x1:x2
    ] = 0


    valid_mask[
        y1:y2,
        x1:x2
    ] = False


# ============================================================
# ALIGN PREVIOUS FRAME TO CURRENT ROI
# ============================================================

def aligned_difference(
    previous_gray,
    current_gray,
    previous_region,
    current_region
):

    height, width = (
        current_gray.shape
    )


    delta_x_pixels = (

        current_region[
            "left"
        ]

        -

        previous_region[
            "left"
        ]

    )


    delta_y_pixels = (

        current_region[
            "top"
        ]

        -

        previous_region[
            "top"
        ]

    )


    # Quantized regions mean these should divide exactly.

    delta_x = int(

        delta_x_pixels

        //
        DOWNSAMPLE

    )


    delta_y = int(

        delta_y_pixels

        //
        DOWNSAMPLE

    )


    # ========================================================
    # NO OVERLAP
    # ========================================================

    if (

        abs(
            delta_x
        )
        >=
        width

        or

        abs(
            delta_y
        )
        >=
        height

    ):

        return (

            None,

            None,

            delta_x_pixels,

            delta_y_pixels

        )


    # ========================================================
    # X OVERLAP
    # ========================================================

    if delta_x >= 0:

        current_x1 = 0

        current_x2 = (
            width
            -
            delta_x
        )


        previous_x1 = (
            delta_x
        )

        previous_x2 = width


    else:

        current_x1 = (
            -delta_x
        )

        current_x2 = width


        previous_x1 = 0

        previous_x2 = (

            width

            +
            delta_x

        )


    # ========================================================
    # Y OVERLAP
    # ========================================================

    if delta_y >= 0:

        current_y1 = 0

        current_y2 = (

            height

            -
            delta_y

        )


        previous_y1 = (
            delta_y
        )

        previous_y2 = height


    else:

        current_y1 = (
            -delta_y
        )

        current_y2 = height


        previous_y1 = 0

        previous_y2 = (

            height

            +
            delta_y

        )


    # ========================================================
    # DIFFERENCE ARRAY
    # ========================================================

    difference = np.zeros(

        current_gray.shape,

        dtype=np.uint8

    )


    valid_mask = np.zeros(

        current_gray.shape,

        dtype=np.bool_

    )


    current_part = current_gray[

        current_y1:current_y2,

        current_x1:current_x2

    ].astype(
        np.int16
    )


    previous_part = previous_gray[

        previous_y1:previous_y2,

        previous_x1:previous_x2

    ].astype(
        np.int16
    )


    part_difference = np.abs(

        current_part

        -

        previous_part

    ).astype(
        np.uint8
    )


    difference[

        current_y1:current_y2,

        current_x1:current_x2

    ] = part_difference


    valid_mask[

        current_y1:current_y2,

        current_x1:current_x2

    ] = True


    return (

        difference,

        valid_mask,

        delta_x_pixels,

        delta_y_pixels

    )


# ============================================================
# MOTION ANALYSIS
# ============================================================

def analyze_motion(
    previous_gray,
    current_gray,
    previous_region,
    current_region,
    fly_local_x,
    fly_local_y,
    previous_bbox_area
):

    # ========================================================
    # FIRST FRAME
    # ========================================================

    if (

        previous_gray is None

        or

        previous_region is None

    ):

        return {

            "motion_mean":
                0.0,

            "active_ratio":
                0.0,

            "center_motion":
                0.0,

            "bbox_area":
                0.0,

            "bbox_growth":
                0.0,

            "raw_looming":
                False,

            "ego_dx":
                0,

            "ego_dy":
                0,

        }


    # ========================================================
    # EGO-MOTION COMPENSATED DIFFERENCE
    # ========================================================

    (

        difference,

        valid_mask,

        ego_dx,

        ego_dy

    ) = aligned_difference(

        previous_gray,

        current_gray,

        previous_region,

        current_region

    )


    if difference is None:

        return {

            "motion_mean":
                0.0,

            "active_ratio":
                0.0,

            "center_motion":
                0.0,

            "bbox_area":
                0.0,

            "bbox_growth":
                0.0,

            "raw_looming":
                False,

            "ego_dx":
                ego_dx,

            "ego_dy":
                ego_dy,

        }


    # ========================================================
    # REMOVE FLY ITSELF
    # ========================================================

    apply_self_mask(

        difference,

        valid_mask,

        fly_local_x,

        fly_local_y

    )


    valid_count = int(
        valid_mask.sum()
    )


    if valid_count <= 0:

        return {

            "motion_mean":
                0.0,

            "active_ratio":
                0.0,

            "center_motion":
                0.0,

            "bbox_area":
                0.0,

            "bbox_growth":
                0.0,

            "raw_looming":
                False,

            "ego_dx":
                ego_dx,

            "ego_dy":
                ego_dy,

        }


    # ========================================================
    # GLOBAL MOTION
    # ========================================================

    motion_mean = float(

        difference[
            valid_mask
        ].mean()

    )


    active = (

        difference

        >=

        MOTION_PIXEL_THRESHOLD

    )


    active &= (
        valid_mask
    )


    active_ratio = (

        float(
            active.sum()
        )

        /

        float(
            valid_count
        )

    )


    # ========================================================
    # LOCAL REGION AROUND FLY
    #
    # Not necessarily geometrical ROI center because at screen
    # edges the ROI is clipped.
    # ========================================================

    height, width = (
        difference.shape
    )


    fly_small_x = int(

        round(

            fly_local_x

            /

            DOWNSAMPLE

        )

    )


    fly_small_y = int(

        round(

            fly_local_y

            /

            DOWNSAMPLE

        )

    )


    local_half_width = max(

        10,

        width // 4

    )


    local_half_height = max(

        8,

        height // 4

    )


    local_x1 = max(

        0,

        fly_small_x
        -
        local_half_width

    )


    local_x2 = min(

        width,

        fly_small_x
        +
        local_half_width

    )


    local_y1 = max(

        0,

        fly_small_y
        -
        local_half_height

    )


    local_y2 = min(

        height,

        fly_small_y
        +
        local_half_height

    )


    local_difference = difference[

        local_y1:local_y2,

        local_x1:local_x2

    ]


    local_valid = valid_mask[

        local_y1:local_y2,

        local_x1:local_x2

    ]


    if np.any(
        local_valid
    ):

        center_motion = float(

            local_difference[
                local_valid
            ].mean()

        )

    else:

        center_motion = 0.0


    # ========================================================
    # MOTION BOUNDING BOX
    # ========================================================

    coordinates = np.argwhere(
        active
    )


    if len(coordinates) >= 5:

        y_min = int(
            coordinates[:, 0].min()
        )


        y_max = int(
            coordinates[:, 0].max()
        )


        x_min = int(
            coordinates[:, 1].min()
        )


        x_max = int(
            coordinates[:, 1].max()
        )


        bbox_pixels = (

            (
                y_max
                -
                y_min
                +
                1
            )

            *

            (
                x_max
                -
                x_min
                +
                1
            )

        )


        total_pixels = (

            height

            *

            width

        )


        bbox_area = (

            bbox_pixels

            /

            total_pixels

        )


    else:

        bbox_area = 0.0


    bbox_growth = (

        bbox_area

        -

        previous_bbox_area

    )


    # ========================================================
    # RAW LOOMING CANDIDATE
    # ========================================================

    raw_looming = (

        active_ratio
        >=
        MIN_ACTIVE_RATIO

        and

        bbox_growth
        >=
        MIN_BBOX_GROWTH

        and

        center_motion
        >=
        MIN_CENTER_MOTION

    )


    return {

        "motion_mean":
            motion_mean,

        "active_ratio":
            active_ratio,

        "center_motion":
            center_motion,

        "bbox_area":
            bbox_area,

        "bbox_growth":
            bbox_growth,

        "raw_looming":
            raw_looming,

        "ego_dx":
            ego_dx,

        "ego_dy":
            ego_dy,

    }


# ============================================================
# TIMING
# ============================================================

frame_interval = (

    1.0

    /

    SENSOR_FPS

)


# ============================================================
# STATS
# ============================================================

fetch_times = []

capture_times = []

processing_times = []

total_times = []


raw_looming_count = 0

confirmed_looming_count = 0


state_failures = 0


frame_count = 0


# ============================================================
# TEMPORAL LOOMING STATE
# ============================================================

looming_streak = 0

last_confirmed_looming_ms = (
    -1000000.0
)


# ============================================================
# PREVIOUS FRAME
# ============================================================

previous_gray = None

previous_region = None

previous_bbox_area = 0.0


# ============================================================
# MSS
#
# New API — eski mss.mss() deprecation warning yok.
# ============================================================

with mss.MSS() as screen_capture:

    print()

    print("=" * 95)
    print("MONITORLER")
    print("=" * 95)


    for index, monitor in enumerate(
        screen_capture.monitors
    ):

        print(

            f"[{index}]"

            f" left={monitor['left']}"

            f" top={monitor['top']}"

            f" width={monitor['width']}"

            f" height={monitor['height']}"

        )


    if (

        MONITOR_INDEX

        >=

        len(
            screen_capture.monitors
        )

    ):

        raise RuntimeError(

            f"MONITOR_INDEX={MONITOR_INDEX} bulunamadı."

        )


    monitor = screen_capture.monitors[
        MONITOR_INDEX
    ]


    print()

    print("=" * 95)
    print("VISION V2 TEST")
    print("=" * 95)

    print(
        "Monitor:",
        MONITOR_INDEX
    )

    print(
        "FPS:",
        SENSOR_FPS
    )

    print(
        "ROI:",
        ROI_WIDTH,
        "x",
        ROI_HEIGHT
    )

    print(
        "Processed:",
        ROI_WIDTH // DOWNSAMPLE,
        "x",
        ROI_HEIGHT // DOWNSAMPLE
    )

    print(
        "Test:",
        TEST_SECONDS,
        "s"
    )

    print()

    print(
        "0-10 sn  : mümkünse ekranı sabit bırak."
    )

    print(
        "10-20 sn : pencereyi sadece sağa/sola taşı."
    )

    print(
        "20-30 sn : bir pencere/objeyi sineğe doğru büyüt."
    )

    print()

    print(
        "Brain'e hâlâ stimulus gönderilmiyor."
    )

    print()


    # ========================================================
    # LOOP
    # ========================================================

    start_time = (
        time.perf_counter()
    )


    next_frame_time = (
        start_time
    )


    next_print_time = (
        start_time
    )


    while True:

        now = (
            time.perf_counter()
        )


        elapsed = (

            now

            -

            start_time

        )


        if elapsed >= TEST_SECONDS:

            break


        # ====================================================
        # FRAME RATE
        # ====================================================

        if now < next_frame_time:

            time.sleep(

                min(

                    0.002,

                    next_frame_time
                    -
                    now

                )

            )

            continue


        frame_start = (
            time.perf_counter()
        )


        # ====================================================
        # STATE
        # ====================================================

        (

            fly_x,

            fly_y,

            fetch_ms,

            state_ok

        ) = get_fly_position()


        fetch_times.append(
            fetch_ms
        )


        if not state_ok:

            state_failures += 1


        # ====================================================
        # REGION
        # ====================================================

        (

            region,

            fly_local_x,

            fly_local_y

        ) = build_capture_region(

            monitor,

            fly_x,

            fly_y

        )


        # ====================================================
        # CAPTURE
        # ====================================================

        capture_start = (
            time.perf_counter()
        )


        screenshot = screen_capture.grab(
            region
        )


        capture_ms = (

            time.perf_counter()

            -

            capture_start

        ) * 1000.0


        # ====================================================
        # PROCESS
        # ====================================================

        process_start = (
            time.perf_counter()
        )


        current_gray = process_frame(
            screenshot
        )


        result = analyze_motion(

            previous_gray,

            current_gray,

            previous_region,

            region,

            fly_local_x,

            fly_local_y,

            previous_bbox_area

        )


        processing_ms = (

            time.perf_counter()

            -

            process_start

        ) * 1000.0


        # ====================================================
        # RAW CANDIDATE
        # ====================================================

        if result[
            "raw_looming"
        ]:

            raw_looming_count += 1

            looming_streak += 1

        else:

            looming_streak = max(

                0,

                looming_streak - 1

            )


        # ====================================================
        # CONFIRMED LOOMING
        # ====================================================

        current_test_ms = (

            elapsed

            *

            1000.0

        )


        cooldown_ok = (

            current_test_ms

            -

            last_confirmed_looming_ms

            >=

            LOOMING_COOLDOWN_MS

        )


        confirmed = (

            looming_streak

            >=

            LOOMING_CONFIRM_FRAMES

            and

            cooldown_ok

        )


        if confirmed:

            confirmed_looming_count += 1


            last_confirmed_looming_ms = (
                current_test_ms
            )


            looming_streak = 0


        # ====================================================
        # SAVE PREVIOUS
        # ====================================================

        previous_gray = (
            current_gray
        )


        previous_region = dict(
            region
        )


        previous_bbox_area = (
            result[
                "bbox_area"
            ]
        )


        # ====================================================
        # STATS
        # ====================================================

        total_ms = (

            time.perf_counter()

            -

            frame_start

        ) * 1000.0


        capture_times.append(
            capture_ms
        )


        processing_times.append(
            processing_ms
        )


        total_times.append(
            total_ms
        )


        frame_count += 1


        # ====================================================
        # PRINT ~1 Hz
        # ====================================================

        now_after = (
            time.perf_counter()
        )


        if now_after >= next_print_time:

            raw_text = (

                "YES"

                if result[
                    "raw_looming"
                ]

                else

                "no"

            )


            confirmed_text = (

                "YES"

                if confirmed

                else

                "no"

            )


            print(

                f"{elapsed:5.1f}s"

                f" | fly=({fly_x:.3f},{fly_y:.3f})"

                f" | ego=({result['ego_dx']:+4d},{result['ego_dy']:+4d})"

                f" | motion={result['motion_mean']:6.2f}"

                f" | local={result['center_motion']:6.2f}"

                f" | active={result['active_ratio'] * 100:6.2f}%"

                f" | area={result['bbox_area'] * 100:6.2f}%"

                f" | growth={result['bbox_growth'] * 100:+6.2f}%"

                f" | raw={raw_text:3s}"

                f" | CONF={confirmed_text:3s}"

                f" | cap={capture_ms:5.2f}ms"

                f" | proc={processing_ms:5.2f}ms"

            )


            next_print_time = (

                now_after

                +

                1.0

            )


        # ====================================================
        # NEXT FRAME
        # ====================================================

        next_frame_time += (
            frame_interval
        )


# ============================================================
# ARRAYS
# ============================================================

fetch_times = np.asarray(
    fetch_times,
    dtype=np.float64
)


capture_times = np.asarray(
    capture_times,
    dtype=np.float64
)


processing_times = np.asarray(
    processing_times,
    dtype=np.float64
)


total_times = np.asarray(
    total_times,
    dtype=np.float64
)


# ============================================================
# RESULT
# ============================================================

print()

print("=" * 95)
print("VISION V2 BENCHMARK SONUCU")
print("=" * 95)


print(
    "Frames:",
    frame_count
)


print(
    "State failures:",
    state_failures
)


print(
    "Raw looming frames:",
    raw_looming_count
)


print(
    "CONFIRMED looming events:",
    confirmed_looming_count
)


if frame_count > 0:

    print()

    print(
        "Mean state fetch:",
        round(
            float(
                fetch_times.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 state fetch:",
        round(
            float(
                np.percentile(
                    fetch_times,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    print()

    print(
        "Mean capture:",
        round(
            float(
                capture_times.mean()
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
                    capture_times,
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
                processing_times.mean()
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
                    processing_times,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    print()

    print(
        "Mean total frame:",
        round(
            float(
                total_times.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 total frame:",
        round(
            float(
                np.percentile(
                    total_times,
                    95
                )
            ),
            3
        ),
        "ms"
    )


    frame_budget_ms = (

        frame_interval

        *

        1000.0

    )


    duty_fraction = (

        total_times.mean()

        /

        frame_budget_ms

    )


    integrated_estimate = (

        capture_times.mean()

        +

        processing_times.mean()

    )


    integrated_duty = (

        integrated_estimate

        /

        frame_budget_ms

    )


    print()

    print(
        "Current test duty:",
        round(
            float(
                duty_fraction
                *
                100.0
            ),
            2
        ),
        "%"
    )


    print(
        "Estimated integrated vision duty:",
        round(
            float(
                integrated_duty
                *
                100.0
            ),
            2
        ),
        "%"
    )


print()

print(
    "NOT:"
)

print(
    "Integrated tahminde localhost /state HTTP maliyeti çıkarılmıştır."
)

print(
    "Bu oran CPU yüzdesi değildir; 8 FPS frame bütçesinin çalışma oranıdır."
)

print(
    "Brain'e henüz visual stimulus gönderilmedi."
)

print("=" * 95)
print("TEST BİTTİ")
print("=" * 95)