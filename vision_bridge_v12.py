import queue
import threading
import time
import urllib.request
from collections import deque

import mss
import numpy as np


# ============================================================
# MALECNS WORLD VISION BRIDGE V12
#
# Daha hassas, lokal + global görsel olay algılama.
#
# WORLD Projector
#       ↓
# low-resolution visual preprocessing
#       ↓
# GLOBAL expansion + LOCAL 3x3 motion
#       ↓
# async /trigger_left, /trigger_center, /trigger_right
#       ↓
# modeled LPLC2 + LC4 drive
#       ↓
# MaleCNS
#       ↓
# motor output
#
# NOT:
# - raw retina simulation değildir
# - doğrudan FLY komutu vermez
# - left/right motor komutu henüz vermez
# ============================================================


print("=" * 95)
print("MALECNS WORLD VISION BRIDGE V12")
print("LOCAL + GLOBAL VISUAL EVENT SENSOR")
print("=" * 95)


# ============================================================
# SERVER
# ============================================================

SERVER = "http://127.0.0.1:8765"

# V6 anatomical visual inputs. NONE/unknown must never send a request.
TRIGGER_ENDPOINTS = {
    "LEFT": "/trigger_left",
    "CENTER": "/trigger_center",
    "RIGHT": "/trigger_right",
}


# ============================================================
# CORRECT PHYSICAL MONITOR
# ============================================================

MONITOR_INDEX = 2


# ============================================================
# VISION RATE
# ============================================================

VISION_FPS = 8.0

FRAME_INTERVAL = 1.0 / VISION_FPS


# ============================================================
# RESOLUTION
#
# V8 = yaklaşık 160x90 idi.
#
# V9 küçük hareketleri daha iyi korumak için yaklaşık
# 200 piksel genişlik kullanır.
# ============================================================

TARGET_PROCESS_WIDTH = 200


# ============================================================
# PIXEL DIFFERENCE
#
# V8 = 20
# V9 = 16
#
# Biraz daha küçük parlaklık değişimleri görülebilir.
# ============================================================

MOTION_THRESHOLD = 16


# ============================================================
# GLOBAL EVENT
#
# Tüm görüntüde belirgin yaklaşma / değişim.
# ============================================================

GLOBAL_MIN_MOTION = 1.00

GLOBAL_MIN_ACTIVE_RATIO = 0.010

GLOBAL_MIN_AREA_GROWTH = 1.35

GLOBAL_MIN_MOTION_GROWTH = 1.10


# ============================================================
# LOCAL EVENT
#
# Görüntüyü 3x3'e bölüyoruz.
#
# Küçük webcam / oyun nesnesi tüm ekranın %2'sini
# değiştirmek zorunda değil.
#
# Tek bir bölgede yeterli hareket varsa visual event olabilir.
# ============================================================

GRID_ROWS = 3

GRID_COLS = 3


LOCAL_MIN_MOTION = 0.90

LOCAL_MIN_ACTIVE_RATIO = 0.030

LOCAL_MIN_AREA_GROWTH = 1.35

LOCAL_MIN_MOTION_GROWTH = 1.10


# ============================================================
# TEMPORAL CONFIRMATION
#
# 4 frame içinde 2 pozitif frame.
# Ayrıca CURRENT frame de candidate olmak zorunda.
# ============================================================

HISTORY_FRAMES = 5

CONFIRM_WINDOW_FRAMES = 4

CONFIRM_HITS_REQUIRED = 2


# ============================================================
# COOLDOWN
#
# V8 = 3000 ms
# V9 = 1800 ms
#
# Aynı olayla beyni spamlemiyoruz ama hareketli sahnelerde
# gereksiz yere 3 saniye kör kalmıyoruz.
# ============================================================

TRIGGER_COOLDOWN_MS = 1800.0


# ============================================================
# SCENE CUT
#
# V8'de %40 üzeri değişimi kesiyorduk.
# Bu normal büyük el hareketlerini de reddedebiliyordu.
#
# Artık sadece ekranın yaklaşık %75'i bir anda değişirse
# "scene cut" kabul ediyoruz.
# ============================================================

SCENE_CUT_ACTIVE_RATIO = 0.75

SCENE_CUT_BLOCK_MS = 750.0


# ============================================================
# STARTUP
# ============================================================

WARMUP_SECONDS = 2.0


# ============================================================
# ASYNC TRIGGER QUEUE
# ============================================================

trigger_queue = queue.Queue(
    maxsize=1
)


# ============================================================
# STATS
# ============================================================

frame_count = 0

trigger_count = 0

trigger_success = 0

trigger_failures = 0

dropped_triggers = 0

scene_cut_count = 0


capture_sum_ms = 0.0

processing_sum_ms = 0.0

total_sum_ms = 0.0


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

        source = item["source"]


        endpoint = TRIGGER_ENDPOINTS.get(region)

        try:

            if endpoint is None:
                print(
                    f"[MALECNS INPUT SKIP] trigger #{number}"
                    f" | source={source} | region={region} | endpoint=NONE"
                )
                continue

            with urllib.request.urlopen(
                SERVER + endpoint,
                timeout=0.75
            ) as response:

                response.read()


            trigger_success += 1


            print(
                f"[MALECNS INPUT OK] "
                f"trigger #{number}"
                f" | source={source}"
                f" | region={region}"
                f" | endpoint={endpoint}"
            )


        except Exception as error:

            trigger_failures += 1


            print(
                f"[MALECNS INPUT FAIL] "
                f"trigger #{number}"
                f" | source={source}"
                f" | region={region}"
                f" | endpoint={endpoint}"
                f" | {error}"
            )


        finally:

            trigger_queue.task_done()


worker = threading.Thread(
    target=trigger_worker,
    daemon=True
)

worker.start()


# ============================================================
# QUEUE TRIGGER
# ============================================================

def enqueue_trigger(
    number,
    source,
    region
):

    global dropped_triggers

    endpoint = TRIGGER_ENDPOINTS.get(region)
    if endpoint is None:
        print(
            f"[VISION INPUT SKIP] trigger #{number}"
            f" | source={source} | region={region} | endpoint=NONE"
        )
        return False

    try:

        trigger_queue.put_nowait(
            {
                "number": number,
                "source": source,
                "region": region
            }
        )


        return True


    except queue.Full:

        dropped_triggers += 1

        return False


# ============================================================
# FRAME -> LOW RES GRAY
# ============================================================

def frame_to_gray(
    screenshot,
    downsample
):

    frame = np.asarray(
        screenshot,
        dtype=np.uint8
    )


    small = frame[
        ::downsample,
        ::downsample,
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
# 3x3 GRID METRICS
# ============================================================

def calculate_grid_metrics(
    active,
    difference
):

    height, width = active.shape


    active_values = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=np.float64
    )


    motion_values = np.zeros(
        (
            GRID_ROWS,
            GRID_COLS
        ),
        dtype=np.float64
    )


    for row in range(
        GRID_ROWS
    ):

        y1 = (
            row
            *
            height
            //
            GRID_ROWS
        )


        y2 = (
            (row + 1)
            *
            height
            //
            GRID_ROWS
        )


        for col in range(
            GRID_COLS
        ):

            x1 = (
                col
                *
                width
                //
                GRID_COLS
            )


            x2 = (
                (col + 1)
                *
                width
                //
                GRID_COLS
            )


            active_tile = active[
                y1:y2,
                x1:x2
            ]


            diff_tile = difference[
                y1:y2,
                x1:x2
            ]


            active_values[
                row,
                col
            ] = float(
                active_tile.mean()
            )


            motion_values[
                row,
                col
            ] = float(
                diff_tile.mean()
            )


    return (
        active_values,
        motion_values
    )


# ============================================================
# COLUMN -> LEFT/CENTER/RIGHT
# ============================================================

def column_region(
    col
):

    if col == 0:

        return "LEFT"


    if col == 1:

        return "CENTER"


    return "RIGHT"


# ============================================================
# GLOBAL DOMINANT REGION
# ============================================================

def dominant_global_region(
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
# LOCAL DETECTOR
# ============================================================

def find_local_candidate(
    tile_active,
    tile_motion,
    baseline_active,
    baseline_motion
):

    best_candidate = False

    best_score = -1.0

    best_row = 0

    best_col = 0

    best_active = 0.0

    best_motion = 0.0

    best_area_growth = 0.0

    best_motion_growth = 0.0


    for row in range(
        GRID_ROWS
    ):

        for col in range(
            GRID_COLS
        ):

            current_active = float(
                tile_active[
                    row,
                    col
                ]
            )


            current_motion = float(
                tile_motion[
                    row,
                    col
                ]
            )


            active_reference = max(
                float(
                    baseline_active[
                        row,
                        col
                    ]
                ),
                0.010
            )


            motion_reference = max(
                float(
                    baseline_motion[
                        row,
                        col
                    ]
                ),
                0.75
            )


            area_growth = (
                current_active
                /
                active_reference
            )


            motion_growth = (
                current_motion
                /
                motion_reference
            )


            candidate = (

                current_motion
                >=
                LOCAL_MIN_MOTION

                and

                current_active
                >=
                LOCAL_MIN_ACTIVE_RATIO

                and

                area_growth
                >=
                LOCAL_MIN_AREA_GROWTH

                and

                motion_growth
                >=
                LOCAL_MIN_MOTION_GROWTH

            )


            score = (
                current_motion
                *
                (
                    current_active
                    *
                    100.0
                )
            )


            if (
                candidate
                and
                score > best_score
            ):

                best_candidate = True

                best_score = score

                best_row = row

                best_col = col

                best_active = current_active

                best_motion = current_motion

                best_area_growth = area_growth

                best_motion_growth = motion_growth


    return (
        best_candidate,
        best_row,
        best_col,
        best_active,
        best_motion,
        best_area_growth,
        best_motion_growth
    )


# ============================================================
# HISTORY
# ============================================================

global_active_history = deque(
    maxlen=HISTORY_FRAMES
)


global_motion_history = deque(
    maxlen=HISTORY_FRAMES
)


tile_active_history = deque(
    maxlen=HISTORY_FRAMES
)


tile_motion_history = deque(
    maxlen=HISTORY_FRAMES
)


# Confirmation belongs to the WORLD screen region, not motion direction.
candidate_histories = {
    region: deque(maxlen=CONFIRM_WINDOW_FRAMES)
    for region in TRIGGER_ENDPOINTS
}


# ============================================================
# SENSOR STATE
# ============================================================

previous_gray = None


last_trigger_ms = -1000000.0


scene_cut_block_until_ms = 0.0


event_latches = {region: False for region in TRIGGER_ENDPOINTS}


# ============================================================
# CAPTURE
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
            f"Monitor {MONITOR_INDEX} bulunamadı."
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
    print("VISION BRIDGE V12 ACTIVE")
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
        "Grid:",
        f"{GRID_COLS}x{GRID_ROWS}"
    )


    print(
        "Pixel motion threshold:",
        MOTION_THRESHOLD
    )


    print(
        "Scene cut:",
        f"{SCENE_CUT_ACTIVE_RATIO * 100:.0f}%"
    )


    print(
        "Cooldown:",
        TRIGGER_COOLDOWN_MS,
        "ms"
    )


    print()

    print(
        "Sensor modes: GLOBAL + LOCAL"
    )


    print(
        "Async MaleCNS input: ON"
    )


    print(
        "Long quiet re-arm: OFF"
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
            # GRAYSCALE
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


                global_active_history.append(
                    0.0
                )


                global_motion_history.append(
                    0.0
                )


                zero_grid = np.zeros(
                    (
                        GRID_ROWS,
                        GRID_COLS
                    ),
                    dtype=np.float64
                )


                tile_active_history.append(
                    zero_grid.copy()
                )


                tile_motion_history.append(
                    zero_grid.copy()
                )


                for history in candidate_histories.values():
                    history.append(False)


                processing_ms = (
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

                processing_sum_ms += processing_ms

                total_sum_ms += total_ms


                next_frame = (
                    frame_start
                    +
                    FRAME_INTERVAL
                )


                continue


            # =================================================
            # BRIGHTNESS COMPENSATION
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
            # GLOBAL METRICS
            # =================================================

            global_motion = float(
                difference.mean()
            )


            global_active = float(
                active.mean()
            )


            (
                global_region,
                global_dominance
            ) = dominant_global_region(
                active
            )


            # =================================================
            # GRID METRICS
            # =================================================

            (
                tile_active,
                tile_motion
            ) = calculate_grid_metrics(
                active,
                difference
            )


            # =================================================
            # SCENE CUT
            # =================================================

            current_ms = (
                time.perf_counter()
                *
                1000.0
            )


            scene_cut = (
                global_active
                >=
                SCENE_CUT_ACTIVE_RATIO
            )


            if scene_cut:

                scene_cut_count += 1


                scene_cut_block_until_ms = (
                    current_ms
                    +
                    SCENE_CUT_BLOCK_MS
                )


                for region, history in candidate_histories.items():
                    history.clear()
                    event_latches[region] = True


                print(
                    "vision"
                    f" | SCENE CUT"
                    f" | active="
                    f"{global_active * 100:5.1f}%"
                    f" | region={global_region}"
                )


                previous_gray = current_gray


                processing_ms = (
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

                processing_sum_ms += processing_ms

                total_sum_ms += total_ms


                now_end = time.perf_counter()


                if (
                    frame_start
                    +
                    FRAME_INTERVAL
                    <=
                    now_end
                ):

                    next_frame = (
                        now_end
                        +
                        FRAME_INTERVAL
                    )

                else:

                    next_frame = (
                        frame_start
                        +
                        FRAME_INTERVAL
                    )


                continue


            # =================================================
            # GLOBAL BASELINE
            # =================================================

            if len(
                global_active_history
            ) >= 2:

                baseline_global_active = float(
                    np.median(
                        np.asarray(
                            global_active_history,
                            dtype=np.float64
                        )
                    )
                )


                baseline_global_motion = float(
                    np.median(
                        np.asarray(
                            global_motion_history,
                            dtype=np.float64
                        )
                    )
                )


            else:

                baseline_global_active = 0.0

                baseline_global_motion = 0.0


            global_active_reference = max(
                baseline_global_active,
                0.004
            )


            global_motion_reference = max(
                baseline_global_motion,
                0.75
            )


            global_area_growth = (
                global_active
                /
                global_active_reference
            )


            global_motion_growth = (
                global_motion
                /
                global_motion_reference
            )


            # =================================================
            # GLOBAL CANDIDATE
            # =================================================

            global_candidate = (

                global_motion
                >=
                GLOBAL_MIN_MOTION

                and

                global_active
                >=
                GLOBAL_MIN_ACTIVE_RATIO

                and

                global_area_growth
                >=
                GLOBAL_MIN_AREA_GROWTH

                and

                global_motion_growth
                >=
                GLOBAL_MIN_MOTION_GROWTH

            )


            # =================================================
            # TILE BASELINE
            # =================================================

            if len(
                tile_active_history
            ) >= 2:

                baseline_tile_active = np.median(
                    np.stack(
                        tuple(
                            tile_active_history
                        ),
                        axis=0
                    ),
                    axis=0
                )


                baseline_tile_motion = np.median(
                    np.stack(
                        tuple(
                            tile_motion_history
                        ),
                        axis=0
                    ),
                    axis=0
                )


            else:

                baseline_tile_active = np.zeros(
                    (
                        GRID_ROWS,
                        GRID_COLS
                    ),
                    dtype=np.float64
                )


                baseline_tile_motion = np.zeros(
                    (
                        GRID_ROWS,
                        GRID_COLS
                    ),
                    dtype=np.float64
                )


            # =================================================
            # LOCAL CANDIDATE
            # =================================================

            (
                local_candidate,
                local_row,
                local_col,
                local_active,
                local_motion,
                local_area_growth,
                local_motion_growth

            ) = find_local_candidate(

                tile_active,
                tile_motion,
                baseline_tile_active,
                baseline_tile_motion

            )


            local_region = column_region(
                local_col
            )


            # =================================================
            # COMBINED EVENT
            # =================================================

            if local_candidate:

                candidate = True

                candidate_source = "LOCAL"

                candidate_region = local_region


            elif global_candidate:

                candidate = True

                candidate_source = "GLOBAL"

                candidate_region = global_region


            else:

                candidate = False

                candidate_source = "NONE"

                candidate_region = global_region


            # =================================================
            # LATCH RESET
            #
            # Sürekli aynı hareket her 1.8 saniyede yeniden
            # trigger olmasın.
            #
            # Tek bir non-candidate frame yeni event'e izin verir.
            # =================================================

            # A candidate in another region does not keep this region latched.
            for region in TRIGGER_ENDPOINTS:
                if not (candidate and candidate_region == region):
                    event_latches[region] = False


            # =================================================
            # HISTORY
            # =================================================

            global_active_history.append(
                global_active
            )


            global_motion_history.append(
                global_motion
            )


            tile_active_history.append(
                tile_active.copy()
            )


            tile_motion_history.append(
                tile_motion.copy()
            )


            warmup_done = (

                frame_start
                -
                start_time

                >=

                WARMUP_SECONDS

            )


            blocked_by_scene_cut = (

                current_ms
                <
                scene_cut_block_until_ms

            )


            if (
                warmup_done
                and
                not blocked_by_scene_cut
            ):

                for region, history in candidate_histories.items():
                    history.append(bool(candidate and candidate_region == region))


            else:

                for history in candidate_histories.values():
                    history.clear()

            candidate_history = candidate_histories.get(candidate_region, ())
            hits = int(sum(candidate_history))


            # =================================================
            # COOLDOWN
            # Keep V9's GLOBAL 1800 ms limit across all regions.
            # =================================================

            cooldown_ok = (

                current_ms
                -
                last_trigger_ms

                >=

                TRIGGER_COOLDOWN_MS

            )


            # =================================================
            # CONFIRM
            #
            # Current frame MUST also be candidate.
            # =================================================

            confirmed = (

                warmup_done

                and

                not blocked_by_scene_cut

                and

                candidate

                and

                not event_latches.get(candidate_region, True)

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
                    candidate_source,
                    candidate_region
                )


                if queued:

                    last_trigger_ms = current_ms

                    event_latches[candidate_region] = True


                    print()

                    print("=" * 95)

                    print(
                        ">>> [VISION -> MALECNS]"
                    )


                    print(
                        f"trigger #{trigger_count}"
                    )


                    print(
                        f"source="
                        f"{candidate_source}"
                    )


                    print(
                        f"region="
                        f"{candidate_region}"
                        f" | endpoint={TRIGGER_ENDPOINTS.get(candidate_region)}"
                    )


                    if (
                        candidate_source
                        ==
                        "LOCAL"
                    ):

                        print(
                            f"tile row="
                            f"{local_row + 1}"
                            f" col="
                            f"{local_col + 1}"
                        )


                        print(
                            f"local motion="
                            f"{local_motion:.2f}"
                            f" | active="
                            f"{local_active * 100:.2f}%"
                        )


                        print(
                            f"local areaX="
                            f"{local_area_growth:.2f}"
                            f" | motionX="
                            f"{local_motion_growth:.2f}"
                        )


                    else:

                        print(
                            f"global motion="
                            f"{global_motion:.2f}"
                            f" | active="
                            f"{global_active * 100:.2f}%"
                        )


                        print(
                            f"global areaX="
                            f"{global_area_growth:.2f}"
                            f" | motionX="
                            f"{global_motion_growth:.2f}"
                        )


                        print(
                            f"dominance="
                            f"{global_dominance * 100:.1f}%"
                        )


                    print()

                    print(
                        "Modeled LPLC2 + LC4 input queued."
                    )


                    print(
                        "Vision direct FLY vermedi."
                    )


                    print("=" * 95)

                    print()


                candidate_histories[candidate_region].clear()


            # =================================================
            # PERFORMANCE
            # =================================================

            processing_ms = (
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

            processing_sum_ms += processing_ms

            total_sum_ms += total_ms


            # =================================================
            # STATUS
            # =================================================

            if frame_start >= next_report:

                max_tile_index = np.unravel_index(
                    int(
                        np.argmax(
                            tile_active
                        )
                    ),
                    tile_active.shape
                )


                strongest_row = int(
                    max_tile_index[0]
                )


                strongest_col = int(
                    max_tile_index[1]
                )


                strongest_active = float(
                    tile_active[
                        strongest_row,
                        strongest_col
                    ]
                )


                print(
                    "vision"
                    f" | globalM={global_motion:5.2f}"
                    f" | globalA={global_active * 100:5.2f}%"
                    f" | localMax="
                    f"{strongest_active * 100:5.2f}%"
                    f" | cand="
                    f"{'YES' if candidate else 'no '}"
                    f" | source="
                    f"{candidate_source:6s}"
                    f" | hits="
                    f"{hits}/{CONFIRM_WINDOW_FRAMES}"
                    f" | region="
                    f"{candidate_region:6s}"
                    f" | cap="
                    f"{capture_ms:5.2f}ms"
                    f" | proc="
                    f"{processing_ms:5.2f}ms"
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


    except KeyboardInterrupt:

        print()

        print(
            "Vision bridge V12 durduruldu."
        )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 95)
print("VISION V12 SUMMARY")
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
    "Dropped triggers:",
    dropped_triggers
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
            processing_sum_ms
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