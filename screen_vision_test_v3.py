import json
import time
import urllib.request

import mss
import numpy as np


# ============================================================
# MALECNS SCREEN VISION SENSOR V3
#
# Amaç:
#
# OBS projector görüntüsünden:
#
# - normal motion
# - webcam noise / brightness change
# - lateral translation
# - gerçek image expansion / looming
#
# ayrımı yapmak.
#
# Bu dosya henüz MaleCNS'e stimulus GÖNDERMEZ.
#
# Bilimsel sınır:
# Bu gerçek Drosophila retina/optic-flow biyofizik modeli değildir.
# Engineered visual preprocessing katmanıdır.
# ============================================================


print("=" * 100)
print("MALECNS SCREEN VISION SENSOR V3")
print("SCALE-CONSISTENT LOOMING DETECTOR")
print("=" * 100)


# ============================================================
# OBS SERVER
# ============================================================

STATE_URL = "http://127.0.0.1:8765/state"


# ============================================================
# MONITOR
#
# Önceki testte:
#
# [1] left=2560 top=0 width=1920 height=1080
#
# OBS Fullscreen Projector burada olmalı.
# ============================================================

MONITOR_INDEX = 1


# ============================================================
# SENSOR RATE
# ============================================================

SENSOR_FPS = 8.0

TEST_SECONDS = 30.0


# ============================================================
# FLY-CENTRIC FIELD OF VIEW
#
# V2:
# 480 x 272 idi.
#
# V3'te biraz büyütüyoruz ki OBS kamera/source içeriğinin
# görüş alanına girme ihtimali artsın.
# ============================================================

ROI_WIDTH = 640

ROI_HEIGHT = 360


# ============================================================
# DOWNSAMPLE
#
# 640 x 360
#      ↓
# 160 x 90
# ============================================================

DOWNSAMPLE = 4


# ============================================================
# SELF MASK
#
# Sineğin kendi sprite'ını görmesini engellemek için.
# ============================================================

SELF_MASK_WIDTH = 150

SELF_MASK_HEIGHT = 120


# ============================================================
# MOTION PARAMETERS
# ============================================================

MOTION_PIXEL_THRESHOLD = 18


# ============================================================
# EDGE / TEXTURE
#
# Scale fitting için tamamen düz alanları kullanmak istemiyoruz.
# ============================================================

EDGE_TEXTURE_THRESHOLD = 12

MIN_FIT_PIXELS = 250


# ============================================================
# SCALE MODELS
#
# Expansion:
# current frame previous frame'e göre büyümüş olabilir.
#
# Shrink:
# Tam ters hipotezi de hesaplıyoruz.
#
# Gerçek looming için:
#
# expansion fit
#     >
# shrink fit
#
# olmasını istiyoruz.
# ============================================================

EXPANSION_SCALES = (
    1.02,
    1.04,
    1.08,
    1.12,
)


SHRINK_SCALES = (
    0.98,
    0.96,
    0.92,
)


# ============================================================
# EXPANSION CENTER GRID
#
# Yaklaşan obje tam ekran merkezinde olmak zorunda değil.
#
# 3 x 3 olası expansion center deniyoruz.
# ============================================================

CENTER_FRACTIONS = (
    0.25,
    0.50,
    0.75,
)


# ============================================================
# LOOMING THRESHOLDS
#
# Bunlar measured fly biology değildir.
# Engineering classification thresholds.
# ============================================================

MIN_MOTION_MEAN = 2.0

MIN_ACTIVE_RATIO = 0.015

MIN_IDENTITY_ERROR = 4.0


# Warp edilmiş expansion modeli identity modelden
# en az bu kadar daha iyi olmalı.
MIN_EXPANSION_SCORE = 0.060


# Expansion modeli shrink modelden de daha iyi olmalı.
MIN_EXPANSION_MARGIN = 0.035


# Kaç frame ardışık candidate gerekli?
LOOMING_CONFIRM_FRAMES = 2


# Aynı visual event beyni ileride spamlemesin.
LOOMING_COOLDOWN_MS = 1200.0


# ============================================================
# FLY POSITION FALLBACK
# ============================================================

last_fly_x = 0.50
last_fly_y = 0.50


# ============================================================
# GET FLY POSITION
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
# ALIGN ROI POSITION
#
# ROI'yi downsample grid'e hizalıyoruz.
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
            maximum,
            aligned
        )
    )


    return int(
        aligned
    )


# ============================================================
# BUILD FLY-CENTRIC CAPTURE REGION
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


    min_left = monitor_left

    max_left = (
        monitor_left
        +
        monitor_width
        -
        ROI_WIDTH
    )


    min_top = monitor_top

    max_top = (
        monitor_top
        +
        monitor_height
        -
        ROI_HEIGHT
    )


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
        "left": int(left),
        "top": int(top),
        "width": ROI_WIDTH,
        "height": ROI_HEIGHT,
    }


    return (
        region,
        fly_local_x,
        fly_local_y
    )


# ============================================================
# MSS FRAME -> SMALL GRAYSCALE
# ============================================================

def process_frame(
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
# ALIGN PREVIOUS FRAME INTO CURRENT ROI COORDINATES
# ============================================================

def align_previous_frame(
    previous_gray,
    current_gray,
    previous_region,
    current_region
):

    height, width = (
        current_gray.shape
    )


    delta_x_pixels = (
        current_region["left"]
        -
        previous_region["left"]
    )


    delta_y_pixels = (
        current_region["top"]
        -
        previous_region["top"]
    )


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


    if (
        abs(delta_x) >= width
        or
        abs(delta_y) >= height
    ):

        return (
            None,
            None,
            delta_x_pixels,
            delta_y_pixels
        )


    if delta_x >= 0:

        current_x1 = 0
        current_x2 = width - delta_x

        previous_x1 = delta_x
        previous_x2 = width

    else:

        current_x1 = -delta_x
        current_x2 = width

        previous_x1 = 0
        previous_x2 = width + delta_x


    if delta_y >= 0:

        current_y1 = 0
        current_y2 = height - delta_y

        previous_y1 = delta_y
        previous_y2 = height

    else:

        current_y1 = -delta_y
        current_y2 = height

        previous_y1 = 0
        previous_y2 = height + delta_y


    aligned_previous = np.zeros(
        current_gray.shape,
        dtype=np.uint8
    )


    valid_mask = np.zeros(
        current_gray.shape,
        dtype=np.bool_
    )


    aligned_previous[
        current_y1:current_y2,
        current_x1:current_x2
    ] = previous_gray[
        previous_y1:previous_y2,
        previous_x1:previous_x2
    ]


    valid_mask[
        current_y1:current_y2,
        current_x1:current_x2
    ] = True


    return (
        aligned_previous,
        valid_mask,
        delta_x_pixels,
        delta_y_pixels
    )


# ============================================================
# SELF MASK
# ============================================================

def remove_fly_from_valid_mask(
    valid_mask,
    fly_local_x,
    fly_local_y
):

    height, width = (
        valid_mask.shape
    )


    fly_x = int(
        round(
            fly_local_x
            /
            DOWNSAMPLE
        )
    )


    fly_y = int(
        round(
            fly_local_y
            /
            DOWNSAMPLE
        )
    )


    mask_width = int(
        round(
            SELF_MASK_WIDTH
            /
            DOWNSAMPLE
        )
    )


    mask_height = int(
        round(
            SELF_MASK_HEIGHT
            /
            DOWNSAMPLE
        )
    )


    x1 = max(
        0,
        fly_x
        -
        mask_width // 2
    )


    x2 = min(
        width,
        fly_x
        +
        mask_width // 2
    )


    y1 = max(
        0,
        fly_y
        -
        mask_height // 2
    )


    y2 = min(
        height,
        fly_y
        +
        mask_height // 2
    )


    valid_mask[
        y1:y2,
        x1:x2
    ] = False


# ============================================================
# ERODE VALID MASK ONE PIXEL
#
# Edge calculation sınır artefact'larını azaltır.
# ============================================================

def erode_valid_mask(
    valid_mask
):

    result = valid_mask.copy()


    result[
        1:,
        :
    ] &= valid_mask[
        :-1,
        :
    ]


    result[
        :-1,
        :
    ] &= valid_mask[
        1:,
        :
    ]


    result[
        :,
        1:
    ] &= valid_mask[
        :,
        :-1
    ]


    result[
        :,
        :-1
    ] &= valid_mask[
        :,
        1:
    ]


    return result


# ============================================================
# PHOTOMETRIC COMPENSATION
#
# Webcam auto-exposure / genel brightness değişimi:
#
# current = previous + global brightness shift
#
# ise bunu motion/looming sanmayalım.
# ============================================================

def compensate_brightness(
    current_gray,
    previous_aligned,
    valid_mask
):

    if not np.any(
        valid_mask
    ):

        return (
            current_gray.copy(),
            0.0
        )


    current_values = current_gray[
        valid_mask
    ].astype(
        np.int16
    )


    previous_values = previous_aligned[
        valid_mask
    ].astype(
        np.int16
    )


    brightness_delta = float(
        np.median(
            current_values
            -
            previous_values
        )
    )


    corrected = (
        current_gray.astype(
            np.float32
        )
        -
        brightness_delta
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
        brightness_delta
    )


# ============================================================
# SIMPLE EDGE MAP
#
# OpenCV kullanmıyoruz.
# ============================================================

def edge_map(
    gray
):

    gray16 = gray.astype(
        np.int16
    )


    edges = np.zeros(
        gray.shape,
        dtype=np.int16
    )


    horizontal = np.abs(
        gray16[
            :,
            1:
        ]
        -
        gray16[
            :,
            :-1
        ]
    )


    vertical = np.abs(
        gray16[
            1:,
            :
        ]
        -
        gray16[
            :-1,
            :
        ]
    )


    edges[
        :,
        1:
    ] += horizontal


    edges[
        1:,
        :
    ] += vertical


    edges = np.clip(
        edges,
        0,
        255
    ).astype(
        np.uint8
    )


    return edges


# ============================================================
# SCALE FIT SCORE
#
# scale > 1:
# expansion hypothesis
#
# scale < 1:
# shrink hypothesis
#
# score > 0:
# scale-warp model identity modelden daha iyi.
# ============================================================

def scale_fit_score(
    current_edge,
    previous_edge,
    valid_mask,
    xx,
    yy,
    center_x,
    center_y,
    scale
):

    height, width = (
        current_edge.shape
    )


    source_x_float = (
        center_x
        +
        (
            xx
            -
            center_x
        )
        /
        scale
    )


    source_y_float = (
        center_y
        +
        (
            yy
            -
            center_y
        )
        /
        scale
    )


    source_x = np.rint(
        source_x_float
    ).astype(
        np.int32
    )


    source_y = np.rint(
        source_y_float
    ).astype(
        np.int32
    )


    in_bounds = (
        (source_x >= 0)
        &
        (source_x < width)
        &
        (source_y >= 0)
        &
        (source_y < height)
    )


    clipped_x = np.clip(
        source_x,
        0,
        width - 1
    )


    clipped_y = np.clip(
        source_y,
        0,
        height - 1
    )


    warped_previous = previous_edge[
        clipped_y,
        clipped_x
    ]


    source_valid = valid_mask[
        clipped_y,
        clipped_x
    ]


    texture_mask = (
        (current_edge >= EDGE_TEXTURE_THRESHOLD)
        |
        (warped_previous >= EDGE_TEXTURE_THRESHOLD)
    )


    fit_mask = (
        valid_mask
        &
        in_bounds
        &
        source_valid
        &
        texture_mask
    )


    fit_pixels = int(
        fit_mask.sum()
    )


    if fit_pixels < MIN_FIT_PIXELS:

        return (
            -1.0,
            0.0,
            0.0,
            fit_pixels
        )


    current_values = current_edge[
        fit_mask
    ].astype(
        np.int16
    )


    identity_values = previous_edge[
        fit_mask
    ].astype(
        np.int16
    )


    warped_values = warped_previous[
        fit_mask
    ].astype(
        np.int16
    )


    identity_error = float(
        np.mean(
            np.abs(
                current_values
                -
                identity_values
            )
        )
    )


    warped_error = float(
        np.mean(
            np.abs(
                current_values
                -
                warped_values
            )
        )
    )


    if identity_error <= 0.001:

        return (
            0.0,
            identity_error,
            warped_error,
            fit_pixels
        )


    score = (
        identity_error
        -
        warped_error
    ) / identity_error


    return (
        float(score),
        identity_error,
        warped_error,
        fit_pixels
    )


# ============================================================
# FIND BEST SCALE MODEL
# ============================================================

def find_best_scale_fit(
    current_edge,
    previous_edge,
    valid_mask,
    scales,
    xx,
    yy
):

    height, width = (
        current_edge.shape
    )


    best_score = -1.0

    best_scale = 1.0

    best_center_x = 0.0
    best_center_y = 0.0

    best_identity_error = 0.0

    best_warp_error = 0.0

    best_fit_pixels = 0


    for center_y_fraction in CENTER_FRACTIONS:

        center_y = (
            center_y_fraction
            *
            (
                height - 1
            )
        )


        for center_x_fraction in CENTER_FRACTIONS:

            center_x = (
                center_x_fraction
                *
                (
                    width - 1
                )
            )


            for scale in scales:

                (
                    score,
                    identity_error,
                    warp_error,
                    fit_pixels

                ) = scale_fit_score(

                    current_edge,
                    previous_edge,
                    valid_mask,
                    xx,
                    yy,
                    center_x,
                    center_y,
                    scale

                )


                if score > best_score:

                    best_score = score

                    best_scale = scale

                    best_center_x = center_x_fraction

                    best_center_y = center_y_fraction

                    best_identity_error = identity_error

                    best_warp_error = warp_error

                    best_fit_pixels = fit_pixels


    return {

        "score":
            float(
                best_score
            ),

        "scale":
            float(
                best_scale
            ),

        "center_x":
            float(
                best_center_x
            ),

        "center_y":
            float(
                best_center_y
            ),

        "identity_error":
            float(
                best_identity_error
            ),

        "warp_error":
            float(
                best_warp_error
            ),

        "fit_pixels":
            int(
                best_fit_pixels
            ),

    }


# ============================================================
# MOTION + SCALE ANALYSIS
# ============================================================

def analyze_frame(
    previous_gray,
    current_gray,
    previous_region,
    current_region,
    fly_local_x,
    fly_local_y
):

    if (
        previous_gray is None
        or
        previous_region is None
    ):

        return {

            "motion_mean": 0.0,
            "active_ratio": 0.0,
            "local_motion": 0.0,

            "brightness_delta": 0.0,

            "expansion_score": 0.0,
            "expansion_scale": 1.0,

            "shrink_score": 0.0,

            "expansion_margin": 0.0,

            "identity_error": 0.0,

            "fit_pixels": 0,

            "candidate": False,

            "ego_dx": 0,
            "ego_dy": 0,

        }


    (
        previous_aligned,
        valid_mask,
        ego_dx,
        ego_dy

    ) = align_previous_frame(

        previous_gray,
        current_gray,
        previous_region,
        current_region

    )


    if previous_aligned is None:

        return {

            "motion_mean": 0.0,
            "active_ratio": 0.0,
            "local_motion": 0.0,

            "brightness_delta": 0.0,

            "expansion_score": 0.0,
            "expansion_scale": 1.0,

            "shrink_score": 0.0,

            "expansion_margin": 0.0,

            "identity_error": 0.0,

            "fit_pixels": 0,

            "candidate": False,

            "ego_dx": ego_dx,
            "ego_dy": ego_dy,

        }


    # ========================================================
    # REMOVE FLY
    # ========================================================

    remove_fly_from_valid_mask(
        valid_mask,
        fly_local_x,
        fly_local_y
    )


    # ========================================================
    # PHOTOMETRIC COMPENSATION
    # ========================================================

    (
        corrected_current,
        brightness_delta

    ) = compensate_brightness(

        current_gray,
        previous_aligned,
        valid_mask

    )


    # ========================================================
    # MOTION
    # ========================================================

    difference = np.abs(

        corrected_current.astype(
            np.int16
        )

        -

        previous_aligned.astype(
            np.int16
        )

    ).astype(
        np.uint8
    )


    difference[
        ~valid_mask
    ] = 0


    valid_count = int(
        valid_mask.sum()
    )


    if valid_count > 0:

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


        active &= valid_mask


        active_ratio = (
            float(
                active.sum()
            )
            /
            float(
                valid_count
            )
        )


    else:

        motion_mean = 0.0
        active_ratio = 0.0


    # ========================================================
    # LOCAL MOTION AROUND FLY
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
        12,
        width // 4
    )


    local_half_height = max(
        8,
        height // 4
    )


    lx1 = max(
        0,
        fly_small_x
        -
        local_half_width
    )


    lx2 = min(
        width,
        fly_small_x
        +
        local_half_width
    )


    ly1 = max(
        0,
        fly_small_y
        -
        local_half_height
    )


    ly2 = min(
        height,
        fly_small_y
        +
        local_half_height
    )


    local_diff = difference[
        ly1:ly2,
        lx1:lx2
    ]


    local_valid = valid_mask[
        ly1:ly2,
        lx1:lx2
    ]


    if np.any(
        local_valid
    ):

        local_motion = float(

            local_diff[
                local_valid
            ].mean()

        )

    else:

        local_motion = 0.0


    # ========================================================
    # EDGE MAPS
    # ========================================================

    current_edge = edge_map(
        corrected_current
    )


    previous_edge = edge_map(
        previous_aligned
    )


    edge_valid = erode_valid_mask(
        valid_mask
    )


    # ========================================================
    # COORDINATE MAP
    # ========================================================

    yy, xx = np.indices(
        current_edge.shape,
        dtype=np.float32
    )


    # ========================================================
    # EXPANSION MODEL
    # ========================================================

    expansion = find_best_scale_fit(

        current_edge,
        previous_edge,
        edge_valid,

        EXPANSION_SCALES,

        xx,
        yy

    )


    # ========================================================
    # SHRINK MODEL
    # ========================================================

    shrink = find_best_scale_fit(

        current_edge,
        previous_edge,
        edge_valid,

        SHRINK_SCALES,

        xx,
        yy

    )


    expansion_margin = (

        expansion[
            "score"
        ]

        -

        shrink[
            "score"
        ]

    )


    # ========================================================
    # CANDIDATE
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

        expansion[
            "identity_error"
        ]
        >=
        MIN_IDENTITY_ERROR

        and

        expansion[
            "score"
        ]
        >=
        MIN_EXPANSION_SCORE

        and

        expansion_margin
        >=
        MIN_EXPANSION_MARGIN

        and

        expansion[
            "fit_pixels"
        ]
        >=
        MIN_FIT_PIXELS

    )


    return {

        "motion_mean":
            motion_mean,

        "active_ratio":
            active_ratio,

        "local_motion":
            local_motion,

        "brightness_delta":
            brightness_delta,

        "expansion_score":
            expansion[
                "score"
            ],

        "expansion_scale":
            expansion[
                "scale"
            ],

        "expansion_center_x":
            expansion[
                "center_x"
            ],

        "expansion_center_y":
            expansion[
                "center_y"
            ],

        "shrink_score":
            shrink[
                "score"
            ],

        "expansion_margin":
            expansion_margin,

        "identity_error":
            expansion[
                "identity_error"
            ],

        "fit_pixels":
            expansion[
                "fit_pixels"
            ],

        "candidate":
            bool(
                candidate
            ),

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


state_failures = 0

candidate_frames = 0

confirmed_events = 0

frame_count = 0


# ============================================================
# TEMPORAL CONFIRMATION
# ============================================================

candidate_streak = 0

last_confirmed_ms = -1000000.0


# ============================================================
# PREVIOUS FRAME
# ============================================================

previous_gray = None

previous_region = None


# ============================================================
# CAPTURE LOOP
# ============================================================

with mss.MSS() as screen_capture:

    print()

    print("=" * 100)
    print("MONITORLER")
    print("=" * 100)


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

    print("=" * 100)
    print("VISION V3 TEST")
    print("=" * 100)

    print(
        "Monitor:",
        MONITOR_INDEX
    )

    print(
        "Sensor FPS:",
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

    print()

    print(
        "TEST PLANI:"
    )

    print(
        "0-10 sn  : mümkün olduğunca normal / sabit kal."
    )

    print(
        "10-20 sn : webcam önünde elini SADECE sağa-sola hareket ettir."
    )

    print(
        "20-30 sn : elini kameradan uzakta başlat ve KAMERAYA DOĞRU yaklaştır."
    )

    print()

    print(
        "ÖNEMLİ: Kamera görüntüsünün sineğin ROI'sinde görünmesi gerekir."
    )

    print(
        "ROI artık 640x360 olduğu için V2'den daha geniş."
    )

    print()

    print(
        "Brain'e hâlâ visual stimulus gönderilmiyor."
    )

    print()


    start_time = time.perf_counter()

    next_frame_time = start_time

    next_print_time = start_time


    while True:

        now = time.perf_counter()


        elapsed = (
            now
            -
            start_time
        )


        if elapsed >= TEST_SECONDS:

            break


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


        frame_start = time.perf_counter()


        # ====================================================
        # SERVER STATE
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
        # SCREEN CAPTURE
        # ====================================================

        capture_start = time.perf_counter()


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

        processing_start = time.perf_counter()


        current_gray = process_frame(
            screenshot
        )


        result = analyze_frame(

            previous_gray,
            current_gray,

            previous_region,
            region,

            fly_local_x,
            fly_local_y

        )


        processing_ms = (
            time.perf_counter()
            -
            processing_start
        ) * 1000.0


        # ====================================================
        # TEMPORAL CANDIDATE
        # ====================================================

        if result[
            "candidate"
        ]:

            candidate_frames += 1

            candidate_streak += 1

        else:

            candidate_streak = max(
                0,
                candidate_streak - 1
            )


        current_test_ms = (
            elapsed
            *
            1000.0
        )


        cooldown_ok = (
            current_test_ms
            -
            last_confirmed_ms
            >=
            LOOMING_COOLDOWN_MS
        )


        confirmed = (
            candidate_streak
            >=
            LOOMING_CONFIRM_FRAMES

            and

            cooldown_ok
        )


        if confirmed:

            confirmed_events += 1

            last_confirmed_ms = (
                current_test_ms
            )

            candidate_streak = 0


            print()

            print(
                ">>> [VISION LOOMING CONFIRMED]"
                f" t={elapsed:.2f}s"
                f" | expansion={result['expansion_score']:.3f}"
                f" | shrink={result['shrink_score']:.3f}"
                f" | margin={result['expansion_margin']:.3f}"
                f" | scale={result['expansion_scale']:.3f}"
            )

            print()


        # ====================================================
        # SAVE PREVIOUS
        # ====================================================

        previous_gray = current_gray

        previous_region = dict(
            region
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
        # ~1 HZ PRINT
        # ====================================================

        now_after = time.perf_counter()


        if now_after >= next_print_time:

            candidate_text = (
                "YES"
                if result[
                    "candidate"
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

                f" | local={result['local_motion']:6.2f}"

                f" | active={result['active_ratio'] * 100:6.2f}%"

                f" | light={result['brightness_delta']:+6.1f}"

                f" | EXP={result['expansion_score']:+6.3f}"

                f" | SHR={result['shrink_score']:+6.3f}"

                f" | margin={result['expansion_margin']:+6.3f}"

                f" | scale={result['expansion_scale']:4.2f}"

                f" | fit={result['fit_pixels']:5d}"

                f" | cand={candidate_text:3s}"

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
# RESULTS
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


print()

print("=" * 100)
print("VISION V3 BENCHMARK SONUCU")
print("=" * 100)


print(
    "Frames:",
    frame_count
)


print(
    "State failures:",
    state_failures
)


print(
    "Expansion candidate frames:",
    candidate_frames
)


print(
    "CONFIRMED looming events:",
    confirmed_events
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
        1000.0
        /
        SENSOR_FPS
    )


    current_duty = (
        total_times.mean()
        /
        frame_budget_ms
    )


    integrated_estimate_ms = (
        capture_times.mean()
        +
        processing_times.mean()
    )


    integrated_duty = (
        integrated_estimate_ms
        /
        frame_budget_ms
    )


    print()

    print(
        "Current test duty:",
        round(
            float(
                current_duty
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

print("BEKLENEN:")

print(
    "0-10 sn:"
    " normal webcam noise -> CONF 0"
)

print(
    "10-20 sn:"
    " sağa-sola hareket -> motion olabilir ama CONF tercihen 0"
)

print(
    "20-30 sn:"
    " kameraya yaklaşma -> VISION LOOMING CONFIRMED çıkabilir"
)

print()

print(
    "NOT: Bu hâlâ engineered visual preprocessing katmanıdır."
)

print(
    "MaleCNS'e stimulus gönderilmedi."
)

print("=" * 100)
print("TEST BİTTİ")
print("=" * 100)