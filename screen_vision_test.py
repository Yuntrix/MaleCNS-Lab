import json
import time
import urllib.request

import mss
import numpy as np


# ============================================================
# MALECNS LIGHTWEIGHT SCREEN VISION TEST
# ============================================================

print("=" * 90)
print("MALECNS LIGHTWEIGHT SCREEN VISION SENSOR")
print("=" * 90)


# ============================================================
# AYARLAR
# ============================================================

STATE_URL = "http://127.0.0.1:8765/state"


# Windows'taki ilk gerçek monitor.
#
# mss.monitors[0] = bütün sanal desktop
# mss.monitors[1] = monitor 1
# mss.monitors[2] = monitor 2 ...
MONITOR_INDEX = 1


# Görsel sensör FPS.
#
# 60 FPS yapmıyoruz.
# Yayın performansı için bilinçli olarak düşük.
SENSOR_FPS = 8.0


# Test süresi.
TEST_SECONDS = 30.0


# ============================================================
# FIELD OF VIEW
#
# Sineğin çevresinden sadece küçük bir alan capture edilir.
#
# Full 1920x1080 capture yapmıyoruz.
# ============================================================

ROI_WIDTH = 480

ROI_HEIGHT = 270


# ============================================================
# DOWNSAMPLE
#
# 480x270
#   ↓ /4
# yaklaşık
# 120x68
#
# Çok hafif.
# ============================================================

DOWNSAMPLE = 4


# ============================================================
# MOTION SETTINGS
# ============================================================

MOTION_PIXEL_THRESHOLD = 18


# ============================================================
# SELF MASK
#
# Sineğin kendi sprite'ı ekran capture'da görünüyorsa
# sensörün kendisini hareket olarak algılamasını azaltır.
#
# Pixel cinsinden, ORIGINAL ROI ölçüsünde.
# ============================================================

SELF_MASK_WIDTH = 130

SELF_MASK_HEIGHT = 100


# ============================================================
# LOOMING CANDIDATE
#
# Bunlar şimdilik sadece engineering thresholds.
# Brain'e henüz bağlanmıyor.
# ============================================================

MIN_ACTIVE_RATIO = 0.025

MIN_BBOX_GROWTH = 0.015

MIN_CENTER_MOTION = 4.5


# ============================================================
# STATE FALLBACK
# ============================================================

last_fly_x = 0.50

last_fly_y = 0.50


# ============================================================
# SERVER'DAN SİNEK KONUMU
# ============================================================

def get_fly_position():

    global last_fly_x
    global last_fly_y

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


    except Exception:

        # Server bir frame cevap vermezse sensörü durdurma.
        # Son bilinen pozisyon kullanılır.
        pass


    return (
        last_fly_x,
        last_fly_y
    )


# ============================================================
# ROI HESAPLA
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


    center_x = (

        monitor_left

        +

        int(
            normalized_x
            *
            monitor_width
        )

    )


    center_y = (

        monitor_top

        +

        int(
            normalized_y
            *
            monitor_height
        )

    )


    left = int(

        center_x

        -

        ROI_WIDTH // 2

    )


    top = int(

        center_y

        -

        ROI_HEIGHT // 2

    )


    # ========================================================
    # MONITOR SINIRLARI
    # ========================================================

    minimum_left = (
        monitor_left
    )

    maximum_left = (

        monitor_left

        +

        monitor_width

        -

        ROI_WIDTH

    )


    minimum_top = (
        monitor_top
    )

    maximum_top = (

        monitor_top

        +

        monitor_height

        -

        ROI_HEIGHT

    )


    left = max(

        minimum_left,

        min(
            maximum_left,
            left
        )

    )


    top = max(

        minimum_top,

        min(
            maximum_top,
            top
        )

    )


    return {

        "left":
            left,

        "top":
            top,

        "width":
            ROI_WIDTH,

        "height":
            ROI_HEIGHT,

    }


# ============================================================
# FRAME -> SMALL GRAYSCALE
# ============================================================

def process_frame(
    screenshot
):

    # MSS:
    # BGRA uint8

    frame = np.asarray(
        screenshot,
        dtype=np.uint8
    )


    # ========================================================
    # DOWNSAMPLE
    #
    # OpenCV resize kullanmıyoruz.
    # Sadece stride sampling.
    # Çok ucuz.
    # ========================================================

    small = frame[

        ::DOWNSAMPLE,

        ::DOWNSAMPLE,

        :3

    ]


    # BGR

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


    # ========================================================
    # INTEGER GRAYSCALE
    #
    # yaklaşık:
    # 0.299 R
    # 0.587 G
    # 0.114 B
    # ========================================================

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
    difference
):

    height, width = (
        difference.shape
    )


    mask_width = max(

        1,

        SELF_MASK_WIDTH
        //
        DOWNSAMPLE

    )


    mask_height = max(

        1,

        SELF_MASK_HEIGHT
        //
        DOWNSAMPLE

    )


    center_x = (
        width // 2
    )

    center_y = (
        height // 2
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


# ============================================================
# MOTION ANALYSIS
# ============================================================

def analyze_motion(
    previous_gray,
    current_gray,
    previous_bbox_area
):

    if previous_gray is None:

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

            "looming_candidate":
                False,

        }


    # ========================================================
    # ABS FRAME DIFFERENCE
    # ========================================================

    difference = np.abs(

        current_gray.astype(
            np.int16
        )

        -

        previous_gray.astype(
            np.int16
        )

    ).astype(
        np.uint8
    )


    # ========================================================
    # REMOVE FLY ITSELF
    # ========================================================

    apply_self_mask(
        difference
    )


    motion_mean = float(
        difference.mean()
    )


    active = (

        difference

        >=

        MOTION_PIXEL_THRESHOLD

    )


    active_ratio = float(
        active.mean()
    )


    # ========================================================
    # CENTER MOTION
    #
    # Field of view'ın merkez kısmı.
    # ========================================================

    height, width = (
        difference.shape
    )


    center_y1 = (
        height // 4
    )

    center_y2 = (
        height
        -
        height // 4
    )


    center_x1 = (
        width // 4
    )

    center_x2 = (
        width
        -
        width // 4
    )


    center_region = difference[

        center_y1:center_y2,

        center_x1:center_x2

    ]


    center_motion = float(
        center_region.mean()
    )


    # ========================================================
    # MOTION BOUNDING AREA
    #
    # Hareket eden bölge ekran alanının ne kadarını kaplıyor?
    #
    # Yaklaşan cisimlerde bunun büyümesini daha sonra
    # looming signal olarak kullanabiliriz.
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
    # LOOMING CANDIDATE
    #
    # Henüz brain stimulus DEĞİL.
    # Sadece sensor test flag'i.
    # ========================================================

    looming_candidate = (

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

        "looming_candidate":
            looming_candidate,

    }


# ============================================================
# SENSOR
# ============================================================

frame_interval = (

    1.0

    /

    SENSOR_FPS

)


capture_times = []

processing_times = []

total_times = []


looming_count = 0


frame_count = 0


previous_gray = None

previous_bbox_area = 0.0


# ============================================================
# MSS
# ============================================================

with mss.mss() as screen_capture:

    print()

    print("=" * 90)
    print("MONITORLER")
    print("=" * 90)


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

            f"MONITOR_INDEX={MONITOR_INDEX} yok."

        )


    monitor = screen_capture.monitors[
        MONITOR_INDEX
    ]


    print()

    print("=" * 90)
    print("VISION TEST")
    print("=" * 90)


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
        "Processed yaklaşık:",
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
        "30 saniye boyunca ekranda bazı objeleri/pencereleri "
        "hareket ettirebilirsin."
    )


    print(
        "Brain'e henüz hiçbir stimulus gönderilmiyor."
    )


    print()


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
        # FRAME PACING
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
        # FLY POSITION
        # ====================================================

        fly_x, fly_y = (
            get_fly_position()
        )


        region = build_capture_region(

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
        # PROCESSING
        # ====================================================

        processing_start = (
            time.perf_counter()
        )


        current_gray = process_frame(
            screenshot
        )


        result = analyze_motion(

            previous_gray,

            current_gray,

            previous_bbox_area

        )


        processing_ms = (

            time.perf_counter()

            -

            processing_start

        ) * 1000.0


        # ====================================================
        # STORE
        # ====================================================

        previous_gray = (
            current_gray
        )


        previous_bbox_area = (
            result[
                "bbox_area"
            ]
        )


        if result[
            "looming_candidate"
        ]:

            looming_count += 1


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

            looming_text = (

                "YES"

                if result[
                    "looming_candidate"
                ]

                else
                "no"

            )


            print(

                f"{elapsed:5.1f}s"

                f" | fly=({fly_x:.3f},{fly_y:.3f})"

                f" | motion={result['motion_mean']:6.2f}"

                f" | center={result['center_motion']:6.2f}"

                f" | active={result['active_ratio'] * 100:6.2f}%"

                f" | area={result['bbox_area'] * 100:6.2f}%"

                f" | growth={result['bbox_growth'] * 100:+6.2f}%"

                f" | loom={looming_text:3s}"

                f" | capture={capture_ms:5.2f}ms"

                f" | process={processing_ms:5.2f}ms"

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
# SUMMARY
# ============================================================

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

print("=" * 90)
print("VISION BENCHMARK SONUCU")
print("=" * 90)


print(
    "Frames:",
    frame_count
)


print(
    "Looming candidates:",
    looming_count
)


if frame_count > 0:

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
        "Mean total sensor frame:",
        round(
            float(
                total_times.mean()
            ),
            3
        ),
        "ms"
    )


    print(
        "P95 total sensor frame:",
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


    sensor_work_fraction = (

        total_times.mean()

        /

        (
            frame_interval
            *
            1000.0
        )

    )


    print()

    print(
        "Sensor work fraction:",
        round(
            float(
                sensor_work_fraction
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
    "Bu test screen motion/expansion preprocessing testidir."
)

print(
    "Henüz MaleCNS'e visual stimulus göndermedi."
)

print("=" * 90)
print("TEST BİTTİ")
print("=" * 90)