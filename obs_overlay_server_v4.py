import json
import random
import threading
import time

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session_multi import MultiInputBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# SERVER
# ============================================================

HOST = "127.0.0.1"
PORT = 8765


# ============================================================
# BRAIN
# ============================================================

W_SYN = 0.110
CHUNK_MS = 20.0
SEED = 404

RUNAWAY_SPIKES_PER_CHUNK = 10000


# ============================================================
# LOOMING
# ============================================================

LOOMING_RATE_HZ = 40.0
LOOMING_DURATION_MS = 60.0


# ============================================================
# MODELED ENDOGENOUS DRIVE
# ============================================================

ENDOGENOUS_RATE_HZ = 2.0
ENDOGENOUS_DURATION_MS = 200.0
ENDOGENOUS_NEURON_COUNT = 128

FIRST_ENDOGENOUS_AT_MS = 3000.0

ENDOGENOUS_INTERVAL_MIN_MS = 8000.0
ENDOGENOUS_INTERVAL_MAX_MS = 18000.0


# ============================================================
# BODY / ACTUATOR MODEL
#
# Bunlar neural parametre değildir.
# Cartoon beden/world mechanics parametreleridir.
# ============================================================

MOVE_HOLD_MS = 450.0


# Bir neural FLY çıktısı kalkışı başlatınca
# bedenin uçuş kontrolünü minimum ne kadar sürdüreceği.
FLIGHT_CONTROL_MS = 1800.0


# Uçuş devam ederken yeni FLY sinyali gelirse
# flight controller'ın ne kadar uzatılacağı.
FLIGHT_REFRESH_MS = 700.0


# Normalized world coordinates.
GROUND_Y = 0.92
FLIGHT_TARGET_Y = 0.69

LEFT_WALL = 0.03
RIGHT_WALL = 0.96
CEILING_Y = 0.05


# Ground movement
MOVE_TARGET_SPEED = 0.075
MOVE_ACCELERATION = 0.30


# Air movement
TAKEOFF_VELOCITY = -0.52

AIR_HORIZONTAL_ACCELERATION = 0.055
MAX_AIR_HORIZONTAL_SPEED = 0.12


# Flight altitude controller
FLIGHT_VERTICAL_GAIN = 1.50
FLIGHT_VERTICAL_BLEND = 0.13

MAX_FLIGHT_UP_SPEED = -0.36
MAX_FLIGHT_DOWN_SPEED = 0.18


# Landing gravity
GRAVITY = 1.20


# Jump
JUMP_VELOCITY = -0.58


# ============================================================
# CACHE
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent


CACHE_DIR = (
    PROJECT_DIR
    /
    "data"
    /
    "processed"
)


CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ENDOGENOUS_CACHE = (
    CACHE_DIR
    /
    "endogenous_drive_indices.npy"
)


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()


shared_state = {

    "brain_time_ms": 0.0,

    "raw_action": "IDLE",

    "total_spikes": 0,
    "descending_spikes": 0,
    "motor_spikes": 0,

    "flight": 0,
    "jump": 0,
    "other": 0,

    "looming_active": False,
    "endogenous_active": False,

    "endogenous_bursts": 0,
    "next_endogenous_ms": FIRST_ENDOGENOUS_AT_MS,

    "body_state": "IDLE",

    "x": 0.22,
    "y": GROUND_Y,

    "vx": 0.0,
    "vy": 0.0,

    "facing": 1,

    "airborne": False,

    "flight_control_remaining_ms": 0.0,

    "compute_ms": 0.0,
    "max_compute_ms": 0.0,

    "backlog_ms": 0.0,

    "runaway": False,
}


commands = {

    "looming_ms": 0.0,

    "endogenous_ms": 0.0,
}


# ============================================================
# LOAD BRAIN
# ============================================================

print("=" * 80)
print("MALECNS AUTONOMOUS OBS SERVER V4")
print("=" * 80)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


print()
print("Nöron:", brain.N)
print("Bağlantı:", len(brain.indices))
print("W_SYN:", brain.w_syn)


# ============================================================
# ENDOGENOUS POPULATION
# ============================================================

def build_endogenous_population():

    # --------------------------------------------------------
    # CACHE
    # --------------------------------------------------------

    if ENDOGENOUS_CACHE.exists():

        cached = np.load(
            ENDOGENOUS_CACHE
        ).astype(
            np.int32
        )


        if len(cached) == ENDOGENOUS_NEURON_COUNT:

            print(
                "Endogenous population cache yüklendi:"
            )

            print(
                ENDOGENOUS_CACHE
            )

            return cached


    # --------------------------------------------------------
    # CACHE YOKSA CONNECTOME'DAN OLUŞTUR
    # --------------------------------------------------------

    print(
        "Endogenous population oluşturuluyor..."
    )


    neurons = brain.neurons


    out_degree = np.diff(
        brain.indptr
    ).astype(
        np.int32
    )


    superclass = (
        neurons["superclass"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )


    nt_role = (
        neurons["nt_role"]
        .fillna("")
        .astype(str)
        .to_numpy()
    )


    base_mask = (

        (
            superclass
            ==
            "cb_intrinsic"
        )

        &

        (
            nt_role
            ==
            "fast_excitatory"
        )

        &

        (
            ~brain.motor_mask
        )

        &

        (
            ~brain.descending_mask
        )

    )


    base_indices = np.flatnonzero(
        base_mask
    )


    base_out = out_degree[
        base_indices
    ]


    out_low = float(

        np.percentile(
            base_out,
            25
        )

    )


    out_high = float(

        np.percentile(
            base_out,
            90
        )

    )


    direct_dn_strength = np.zeros(

        brain.N,

        dtype=np.float64
    )


    direct_dn_targets = np.zeros(

        brain.N,

        dtype=np.int32
    )


    for neuron_idx in base_indices:

        degree = out_degree[
            neuron_idx
        ]


        if (
            degree < out_low
            or
            degree > out_high
        ):

            continue


        start = brain.indptr[
            neuron_idx
        ]


        end = brain.indptr[
            neuron_idx + 1
        ]


        if end <= start:

            continue


        targets = brain.indices[
            start:end
        ]


        weights = brain.weights[
            start:end
        ]


        dn_mask = (

            brain.descending_mask[
                targets
            ]

            &

            (
                weights > 0
            )

        )


        if not np.any(
            dn_mask
        ):

            continue


        direct_dn_targets[
            neuron_idx
        ] = int(

            np.count_nonzero(
                dn_mask
            )

        )


        direct_dn_strength[
            neuron_idx
        ] = float(

            np.sum(

                weights[
                    dn_mask
                ]

            )

        )


    candidate_mask = (

        base_mask

        &

        (
            direct_dn_targets > 0
        )

        &

        (
            out_degree >= out_low
        )

        &

        (
            out_degree <= out_high
        )

    )


    candidates = np.flatnonzero(
        candidate_mask
    )


    scores = np.zeros(

        brain.N,

        dtype=np.float64
    )


    scores[
        candidates
    ] = (

        direct_dn_strength[
            candidates
        ]

        /

        np.sqrt(

            np.maximum(

                out_degree[
                    candidates
                ],

                1

            )

        )

    )


    order = np.argsort(

        scores[
            candidates
        ]

    )[::-1]


    ranked = candidates[
        order
    ]


    selected = ranked[
        :ENDOGENOUS_NEURON_COUNT
    ].astype(
        np.int32
    )


    np.save(

        ENDOGENOUS_CACHE,

        selected
    )


    print(
        "Endogenous cache kaydedildi:"
    )

    print(
        ENDOGENOUS_CACHE
    )


    return selected


endogenous_indices = (
    build_endogenous_population()
)


# ============================================================
# LOOMING POPULATION
# ============================================================

types = (
    brain.neurons["type"]
    .fillna("")
    .astype(str)
)


looming_indices = np.flatnonzero(

    types.isin(
        [
            "LPLC2",
            "LC4",
        ]
    ).to_numpy()

).astype(
    np.int32
)


print(
    "Endogenous neurons:",
    len(endogenous_indices)
)


print(
    "Looming neurons:",
    len(looming_indices)
)


# ============================================================
# SESSION
# ============================================================

session = MultiInputBrainSession(

    brain,

    input_groups={

        "endogenous":
            endogenous_indices,

        "looming":
            looming_indices,

    },

    seed=SEED
)


# ============================================================
# WARMUP
# ============================================================

print(
    "Numba warmup..."
)


session.step(

    group_rates={

        "endogenous": 0.0,

        "looming": 0.0,

    },

    chunk_ms=1.0
)


session.reset(
    seed=SEED
)


print(
    "Warmup tamam."
)


# ============================================================
# BODY CONTROLLER
# ============================================================

class BodyState:

    def __init__(
        self
    ):

        self.x = 0.22
        self.y = GROUND_Y

        self.vx = 0.0
        self.vy = 0.0

        self.facing = 1

        self.state = "IDLE"

        self.airborne = False

        self.move_until_ms = 0.0

        self.flight_control_until_ms = 0.0


    # ========================================================
    # START / REFRESH FLIGHT
    # ========================================================

    def receive_flight_signal(
        self,
        brain_time_ms
    ):

        # İlk kalkış
        if not self.airborne:

            self.airborne = True

            self.state = "FLY"

            self.vy = (
                TAKEOFF_VELOCITY
            )


            self.vx += (

                0.025

                *

                self.facing

            )


            self.flight_control_until_ms = (

                brain_time_ms

                +

                FLIGHT_CONTROL_MS

            )


        # Zaten uçuyorsa neural FLY sinyali
        # uçuş kontrol süresini uzatabilir.
        else:

            self.flight_control_until_ms = max(

                self.flight_control_until_ms,

                brain_time_ms

                +

                FLIGHT_REFRESH_MS

            )


            self.state = "FLY"


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        raw_action,
        brain_time_ms,
        dt_seconds
    ):

        # ====================================================
        # RAW BRAIN OUTPUT
        # ====================================================

        if raw_action == "FLY":

            self.receive_flight_signal(
                brain_time_ms
            )


        elif raw_action == "JUMP":

            if not self.airborne:

                self.airborne = True

                self.state = "JUMP"

                self.vy = (
                    JUMP_VELOCITY
                )


        elif raw_action == "MOVE":

            # Havada gelen MOVE,
            # uçuşu kesmez.
            if not self.airborne:

                self.move_until_ms = (

                    brain_time_ms

                    +

                    MOVE_HOLD_MS

                )


        # ====================================================
        # FLIGHT CONTROLLER ACTIVE?
        # ====================================================

        flight_control_active = (

            self.airborne

            and

            brain_time_ms
            <
            self.flight_control_until_ms

        )


        # ====================================================
        # AIRBORNE
        # ====================================================

        if self.airborne:

            if flight_control_active:

                self.state = "FLY"


                # --------------------------------------------
                # Target altitude controller
                #
                # Sinek yalnızca balistik olarak yukarı fırlayıp
                # geri düşmüyor.
                #
                # Flight actuator target yüksekliğe doğru
                # kontrollü vertical velocity oluşturuyor.
                # --------------------------------------------

                altitude_error = (

                    FLIGHT_TARGET_Y

                    -

                    self.y

                )


                desired_vy = (

                    altitude_error

                    *

                    FLIGHT_VERTICAL_GAIN

                )


                desired_vy = max(

                    MAX_FLIGHT_UP_SPEED,

                    min(

                        MAX_FLIGHT_DOWN_SPEED,

                        desired_vy

                    )

                )


                self.vy += (

                    (
                        desired_vy
                        -
                        self.vy
                    )

                    *

                    FLIGHT_VERTICAL_BLEND

                )


            else:

                # --------------------------------------------
                # Flight controller bitti.
                # Artık ballistic landing.
                # --------------------------------------------

                self.state = "FLY"


                self.vy += (

                    GRAVITY

                    *

                    dt_seconds

                )


        # ====================================================
        # GROUNDED
        # ====================================================

        else:

            if (
                brain_time_ms
                <
                self.move_until_ms
            ):

                self.state = "MOVE"

            else:

                self.state = "IDLE"


        # ====================================================
        # HORIZONTAL MOTION
        # ====================================================

        if self.airborne:

            self.vx += (

                AIR_HORIZONTAL_ACCELERATION

                *

                self.facing

                *

                dt_seconds

            )


            self.vx = max(

                -MAX_AIR_HORIZONTAL_SPEED,

                min(

                    MAX_AIR_HORIZONTAL_SPEED,

                    self.vx

                )

            )


        else:

            if self.state == "MOVE":

                target_vx = (

                    MOVE_TARGET_SPEED

                    *

                    self.facing

                )


                if self.vx < target_vx:

                    self.vx += (

                        MOVE_ACCELERATION

                        *

                        dt_seconds

                    )


                    self.vx = min(

                        self.vx,

                        target_vx

                    )


                elif self.vx > target_vx:

                    self.vx -= (

                        MOVE_ACCELERATION

                        *

                        dt_seconds

                    )


                    self.vx = max(

                        self.vx,

                        target_vx

                    )


            else:

                self.vx *= 0.88


        # ====================================================
        # POSITION
        # ====================================================

        self.x += (

            self.vx

            *

            dt_seconds

        )


        self.y += (

            self.vy

            *

            dt_seconds

        )


        # ====================================================
        # LEFT WALL
        # ====================================================

        if self.x <= LEFT_WALL:

            self.x = LEFT_WALL

            self.facing = 1

            self.vx = abs(
                self.vx
            )


        # ====================================================
        # RIGHT WALL
        # ====================================================

        elif self.x >= RIGHT_WALL:

            self.x = RIGHT_WALL

            self.facing = -1

            self.vx = -abs(
                self.vx
            )


        # ====================================================
        # CEILING
        # ====================================================

        if self.y <= CEILING_Y:

            self.y = CEILING_Y


            if self.vy < 0:

                self.vy = 0.0


        # ====================================================
        # LANDING
        # ====================================================

        if self.y >= GROUND_Y:

            self.y = GROUND_Y


            if self.vy > 0:

                self.vy = 0.0


            # Flight controller hâlâ aktifse
            # yerde kalmasına izin verme.
            if flight_control_active:

                self.airborne = True

                self.state = "FLY"

                self.vy = -0.20


            else:

                if self.airborne:

                    self.airborne = False


                if (
                    brain_time_ms
                    <
                    self.move_until_ms
                ):

                    self.state = "MOVE"

                else:

                    self.state = "IDLE"


    # ========================================================
    # REMAINING FLIGHT CONTROL
    # ========================================================

    def flight_remaining(
        self,
        brain_time_ms
    ):

        return max(

            0.0,

            self.flight_control_until_ms

            -

            brain_time_ms

        )


# ============================================================
# BODY
# ============================================================

body = BodyState()


# ============================================================
# BRAIN LOOP
# ============================================================

def brain_loop():

    rng = random.Random(
        20260918
    )


    brain_time_ms = 0.0


    next_endogenous_ms = (
        FIRST_ENDOGENOUS_AT_MS
    )


    automatic_endogenous_remaining = 0.0
    manual_endogenous_remaining = 0.0

    looming_remaining = 0.0


    endogenous_burst_count = 0


    max_compute_ms = 0.0


    dt_seconds = (

        CHUNK_MS

        /

        1000.0

    )


    next_tick = (
        time.perf_counter()
    )


    last_raw = "IDLE"
    last_body = "IDLE"


    print()
    print("Brain + flight controller başladı.")


    while True:

        # ====================================================
        # COMMANDS
        # ====================================================

        with state_lock:

            if commands[
                "looming_ms"
            ] > 0:

                looming_remaining = max(

                    looming_remaining,

                    commands[
                        "looming_ms"
                    ]

                )


                commands[
                    "looming_ms"
                ] = 0.0


            if commands[
                "endogenous_ms"
            ] > 0:

                manual_endogenous_remaining = max(

                    manual_endogenous_remaining,

                    commands[
                        "endogenous_ms"
                    ]

                )


                commands[
                    "endogenous_ms"
                ] = 0.0


        # ====================================================
        # AUTOMATIC ENDOGENOUS BURST
        # ====================================================

        if (

            brain_time_ms
            >=
            next_endogenous_ms

            and

            automatic_endogenous_remaining
            <=
            0.0

        ):

            automatic_endogenous_remaining = (
                ENDOGENOUS_DURATION_MS
            )


            endogenous_burst_count += 1


            interval_ms = rng.uniform(

                ENDOGENOUS_INTERVAL_MIN_MS,

                ENDOGENOUS_INTERVAL_MAX_MS

            )


            next_endogenous_ms = (

                brain_time_ms

                +

                interval_ms

            )


            print(

                f"[AUTO INTERNAL] "

                f"brain={brain_time_ms:.0f} ms "

                f"| burst={endogenous_burst_count} "

                f"| next={next_endogenous_ms:.0f} ms"

            )


        # ====================================================
        # INPUT RATES
        # ====================================================

        endogenous_active = (

            automatic_endogenous_remaining > 0.0

            or

            manual_endogenous_remaining > 0.0

        )


        endogenous_rate = (

            ENDOGENOUS_RATE_HZ

            if endogenous_active

            else 0.0

        )


        looming_rate = (

            LOOMING_RATE_HZ

            if looming_remaining > 0.0

            else 0.0

        )


        # ====================================================
        # BRAIN STEP
        # ====================================================

        compute_start = (
            time.perf_counter()
        )


        result = session.step(

            group_rates={

                "endogenous":
                    endogenous_rate,

                "looming":
                    looming_rate,

            },

            chunk_ms=CHUNK_MS

        )


        compute_ms = (

            time.perf_counter()

            -

            compute_start

        ) * 1000.0


        max_compute_ms = max(

            max_compute_ms,

            compute_ms

        )


        # ====================================================
        # MOTOR DECODER
        # ====================================================

        movement = decoder.decode(

            neurons=
                brain.neurons,

            spike_counts=
                result[
                    "spike_counts"
                ],

            motor_mask=
                brain.motor_mask

        )


        raw_action = movement[
            "action"
        ]


        # ====================================================
        # BODY
        # ====================================================

        body.update(

            raw_action=
                raw_action,

            brain_time_ms=
                brain_time_ms,

            dt_seconds=
                dt_seconds

        )


        # ====================================================
        # TIMERS
        # ====================================================

        automatic_endogenous_remaining = max(

            0.0,

            automatic_endogenous_remaining

            -

            CHUNK_MS

        )


        manual_endogenous_remaining = max(

            0.0,

            manual_endogenous_remaining

            -

            CHUNK_MS

        )


        looming_remaining = max(

            0.0,

            looming_remaining

            -

            CHUNK_MS

        )


        # ====================================================
        # SAFETY
        # ====================================================

        runaway = (

            result[
                "total_spikes"
            ]

            >
            RUNAWAY_SPIKES_PER_CHUNK

        )


        # ====================================================
        # SHARED STATE
        # ====================================================

        with state_lock:

            shared_state[
                "brain_time_ms"
            ] = round(
                brain_time_ms,
                1
            )


            shared_state[
                "raw_action"
            ] = raw_action


            shared_state[
                "total_spikes"
            ] = result[
                "total_spikes"
            ]


            shared_state[
                "descending_spikes"
            ] = result[
                "descending_spikes"
            ]


            shared_state[
                "motor_spikes"
            ] = result[
                "motor_spikes"
            ]


            shared_state[
                "flight"
            ] = int(

                movement.get(
                    "flight",
                    0
                )

            )


            shared_state[
                "jump"
            ] = int(

                movement.get(
                    "jump",
                    0
                )

            )


            shared_state[
                "other"
            ] = int(

                movement.get(
                    "other",
                    0
                )

            )


            shared_state[
                "looming_active"
            ] = bool(
                looming_rate > 0
            )


            shared_state[
                "endogenous_active"
            ] = bool(
                endogenous_rate > 0
            )


            shared_state[
                "endogenous_bursts"
            ] = endogenous_burst_count


            shared_state[
                "next_endogenous_ms"
            ] = round(
                next_endogenous_ms,
                1
            )


            shared_state[
                "body_state"
            ] = body.state


            shared_state[
                "x"
            ] = round(
                body.x,
                6
            )


            shared_state[
                "y"
            ] = round(
                body.y,
                6
            )


            shared_state[
                "vx"
            ] = round(
                body.vx,
                6
            )


            shared_state[
                "vy"
            ] = round(
                body.vy,
                6
            )


            shared_state[
                "facing"
            ] = body.facing


            shared_state[
                "airborne"
            ] = body.airborne


            shared_state[
                "flight_control_remaining_ms"
            ] = round(

                body.flight_remaining(
                    brain_time_ms
                ),

                1

            )


            shared_state[
                "compute_ms"
            ] = round(
                compute_ms,
                3
            )


            shared_state[
                "max_compute_ms"
            ] = round(
                max_compute_ms,
                3
            )


            shared_state[
                "runaway"
            ] = bool(
                runaway
            )


        # ====================================================
        # LOG STATE CHANGES
        # ====================================================

        if (

            raw_action != last_raw

            or

            body.state != last_body

        ):

            print(

                f"[STATE] "

                f"{brain_time_ms:8.0f} ms"

                f" | raw={raw_action:5s}"

                f" | body={body.state:5s}"

                f" | spike={result['total_spikes']:4d}"

                f" | DN={result['descending_spikes']:3d}"

                f" | motor={result['motor_spikes']:3d}"

                f" | y={body.y:.3f}"

                f" | flightRemain="
                f"{body.flight_remaining(brain_time_ms):.0f} ms"

                f" | calc={compute_ms:.2f} ms"

            )


            last_raw = raw_action

            last_body = body.state


        # ====================================================
        # ADVANCE TIME
        # ====================================================

        brain_time_ms += (
            CHUNK_MS
        )


        # ====================================================
        # REALTIME PACING
        # ====================================================

        next_tick += (
            dt_seconds
        )


        sleep_seconds = (

            next_tick

            -

            time.perf_counter()

        )


        if sleep_seconds > 0:

            with state_lock:

                shared_state[
                    "backlog_ms"
                ] = 0.0


            time.sleep(
                sleep_seconds
            )


        else:

            backlog_ms = (

                -sleep_seconds

                *

                1000.0

            )


            with state_lock:

                shared_state[
                    "backlog_ms"
                ] = round(
                    backlog_ms,
                    3
                )


            if backlog_ms > 100.0:

                next_tick = (
                    time.perf_counter()
                )


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<style>

html,
body {

    margin: 0;
    padding: 0;

    width: 100%;
    height: 100%;

    overflow: hidden;

    background: transparent;

}


#world {

    position: fixed;

    inset: 0;

    overflow: hidden;

    background: transparent;

}


/* =========================================================
   FLY
   ========================================================= */

#fly {

    position: absolute;

    width: 76px;
    height: 58px;

    transform-origin:
        center
        center;

    will-change:
        transform;

}


.fly-body {

    position: absolute;

    width: 40px;
    height: 28px;

    left: 20px;
    top: 17px;

    border-radius:
        50%
        58%
        58%
        50%;

    border:
        2px solid
        #111;

    background:

        repeating-linear-gradient(

            90deg,

            #252525 0px,

            #252525 7px,

            #d89e25 7px,

            #d89e25 12px

        );

}


.head {

    position: absolute;

    width: 24px;
    height: 24px;

    left: 7px;
    top: 18px;

    border-radius: 50%;

    border:
        2px solid
        #111;

    background:
        #343434;

}


.eye {

    position: absolute;

    width: 9px;
    height: 13px;

    border-radius: 50%;

    background:

        radial-gradient(

            circle at 35% 25%,

            #ffc4c4 0%,

            #e02730 30%,

            #74080c 100%

        );

}


.eye.left {

    left: 3px;
    top: 4px;

}


.eye.right {

    left: 12px;
    top: 5px;

}


/* =========================================================
   WINGS
   ========================================================= */

.wing {

    position: absolute;

    width: 37px;
    height: 20px;

    left: 31px;

    border-radius:
        70%
        30%
        70%
        30%;

    border:

        1px solid

        rgba(
            90,
            150,
            180,
            0.75
        );

    background:

        rgba(
            190,
            235,
            255,
            0.58
        );

    transform-origin:
        4px
        10px;

}


.wing.top {

    top: 7px;

    transform:
        rotate(-24deg);

}


.wing.bottom {

    top: 30px;

    transform:
        rotate(24deg);

}


/* =========================================================
   LEGS
   ========================================================= */

.leg {

    position: absolute;

    width: 30px;
    height: 2px;

    left: 28px;

    background:
        #171717;

    transform-origin:
        2px
        1px;

}


.leg.one {

    top: 37px;

    transform:
        rotate(35deg);

}


.leg.two {

    top: 41px;

    transform:
        rotate(65deg);

}


.leg.three {

    top: 42px;

    transform:
        rotate(100deg);

}


/* =========================================================
   ANTENNA
   ========================================================= */

.antenna {

    position: absolute;

    width: 15px;
    height: 2px;

    background:
        #161616;

}


.antenna.one {

    left: -2px;
    top: 20px;

    transform:
        rotate(25deg);

}


.antenna.two {

    left: -1px;
    top: 29px;

    transform:
        rotate(-20deg);

}


/* =========================================================
   ANIMATIONS
   ========================================================= */

#fly.MOVE {

    animation:

        walkBob

        0.15s

        ease-in-out

        infinite

        alternate;

}


#fly.FLY .wing.top {

    animation:

        wingTop

        0.055s

        linear

        infinite;

}


#fly.FLY .wing.bottom {

    animation:

        wingBottom

        0.055s

        linear

        infinite;

}


#fly.JUMP {

    animation:

        jumpBody

        0.12s

        ease-in-out

        infinite

        alternate;

}


@keyframes walkBob {

    from {
        margin-top: 0px;
    }

    to {
        margin-top: -3px;
    }

}


@keyframes wingTop {

    0% {

        transform:
            rotate(-38deg);

    }

    50% {

        transform:
            rotate(17deg);

    }

    100% {

        transform:
            rotate(-38deg);

    }

}


@keyframes wingBottom {

    0% {

        transform:
            rotate(38deg);

    }

    50% {

        transform:
            rotate(-17deg);

    }

    100% {

        transform:
            rotate(38deg);

    }

}


@keyframes jumpBody {

    from {
        margin-top: 0px;
    }

    to {
        margin-top: -4px;
    }

}


/* =========================================================
   DEBUG
   ========================================================= */

.debug-ui {

    display: none;

}


body.debug .debug-ui {

    display: block;

}


#debugPanel {

    position: fixed;

    left: 15px;
    top: 15px;

    width: 370px;

    padding: 14px;

    border-radius: 10px;

    background:

        rgba(
            0,
            0,
            0,
            0.84
        );

    color:
        white;

    font-family:
        Consolas,
        monospace;

    font-size: 13px;

    line-height: 1.45;

    white-space:
        pre-wrap;

    z-index: 1000;

}


#buttons {

    position: fixed;

    right: 18px;
    top: 18px;

    z-index: 1000;

}


button {

    display: block;

    width: 150px;

    margin-bottom: 10px;

    padding: 11px;

    cursor: pointer;

    font-size: 14px;

}

</style>

</head>


<body>


<div id="world">

    <div
        id="fly"
        class="IDLE"
    >

        <div class="wing top"></div>
        <div class="wing bottom"></div>

        <div class="leg one"></div>
        <div class="leg two"></div>
        <div class="leg three"></div>

        <div class="fly-body"></div>

        <div class="head">

            <div class="eye left"></div>
            <div class="eye right"></div>

        </div>

        <div class="antenna one"></div>
        <div class="antenna two"></div>

    </div>

</div>


<div
    id="debugPanel"
    class="debug-ui"
>
Loading...
</div>


<div
    id="buttons"
    class="debug-ui"
>

    <button onclick="triggerLooming()">
        TEST LOOMING
    </button>

    <button onclick="triggerInternal()">
        TEST INTERNAL
    </button>

</div>


<script>


const params =

    new URLSearchParams(
        window.location.search
    );


const debug =

    params.get("debug")
    ===
    "1";


if (debug) {

    document.body.classList.add(
        "debug"
    );

}


const fly =

    document.getElementById(
        "fly"
    );


const panel =

    document.getElementById(
        "debugPanel"
    );


let state = {

    body_state: "IDLE",

    x: 0.22,

    y: 0.92,

    facing: 1

};


/* ==========================================================
   TEST BUTTONS
   ========================================================== */

async function triggerLooming() {

    try {

        await fetch(

            "/trigger",

            {
                cache: "no-store"
            }

        );

    }

    catch (error) {

    }

}


async function triggerInternal() {

    try {

        await fetch(

            "/internal",

            {
                cache: "no-store"
            }

        );

    }

    catch (error) {

    }

}


/* ==========================================================
   STATE
   ========================================================== */

async function updateState() {

    try {

        const response =

            await fetch(

                "/state",

                {
                    cache: "no-store"
                }

            );


        state =

            await response.json();


        if (debug) {

            panel.textContent =

                "MALECNS AUTONOMOUS OBS V4\n\n"

                +

                "brain: "
                + state.brain_time_ms
                + " ms\n"

                +

                "raw action: "
                + state.raw_action
                + "\n"

                +

                "BODY STATE: "
                + state.body_state
                + "\n"

                +

                "airborne: "
                + state.airborne
                + "\n"

                +

                "flight control: "
                + state.flight_control_remaining_ms
                + " ms\n\n"

                +

                "x: "
                + Number(
                    state.x
                ).toFixed(3)
                + "\n"

                +

                "y: "
                + Number(
                    state.y
                ).toFixed(3)
                + "\n"

                +

                "vx: "
                + Number(
                    state.vx
                ).toFixed(4)
                + "\n"

                +

                "vy: "
                + Number(
                    state.vy
                ).toFixed(4)
                + "\n\n"

                +

                "spikes: "
                + state.total_spikes
                + "\n"

                +

                "DN: "
                + state.descending_spikes
                + "\n"

                +

                "motor: "
                + state.motor_spikes
                + "\n"

                +

                "flight motor: "
                + state.flight
                + "\n"

                +

                "jump: "
                + state.jump
                + "\n"

                +

                "other: "
                + state.other
                + "\n\n"

                +

                "internal active: "
                + state.endogenous_active
                + "\n"

                +

                "internal bursts: "
                + state.endogenous_bursts
                + "\n"

                +

                "next internal: "
                + state.next_endogenous_ms
                + " ms\n"

                +

                "looming: "
                + state.looming_active
                + "\n\n"

                +

                "calc: "
                + state.compute_ms
                + " ms\n"

                +

                "max calc: "
                + state.max_compute_ms
                + " ms\n"

                +

                "backlog: "
                + state.backlog_ms
                + " ms\n"

                +

                "runaway: "
                + state.runaway;

        }

    }

    catch (error) {

    }

}


setInterval(

    updateState,

    40

);


/* ==========================================================
   RENDER
   ========================================================== */

function render() {

    const width =
        window.innerWidth;


    const height =
        window.innerHeight;


    const x =

        Number(
            state.x
        )

        *

        width;


    const y =

        Number(
            state.y
        )

        *

        height;


    const facing =

        Number(
            state.facing
        )

        >=
        0

        ?

        1

        :

        -1;


    const bodyState =

        state.body_state

        ||

        "IDLE";


    fly.className =
        bodyState;


    fly.style.transform =

        "translate("

        +

        (x - 38).toFixed(1)

        +

        "px,"

        +

        (y - 55).toFixed(1)

        +

        "px) "

        +

        "scaleX("

        +

        facing

        +

        ")";


    requestAnimationFrame(
        render
    );

}


requestAnimationFrame(
    render
);


</script>


</body>

</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class OverlayHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format,
        *args
    ):

        return


    def send_bytes(
        self,
        data,
        content_type,
        status=200
    ):

        self.send_response(
            status
        )


        self.send_header(

            "Content-Type",

            content_type

        )


        self.send_header(

            "Cache-Control",

            "no-store"

        )


        self.send_header(

            "Content-Length",

            str(
                len(data)
            )

        )


        self.end_headers()


        self.wfile.write(
            data
        )


    def send_json(
        self,
        obj
    ):

        data = json.dumps(
            obj
        ).encode(
            "utf-8"
        )


        self.send_bytes(

            data,

            "application/json; charset=utf-8"

        )


    def do_GET(
        self
    ):

        path = urlparse(
            self.path
        ).path


        # ====================================================
        # STATE
        # ====================================================

        if path == "/state":

            with state_lock:

                snapshot = dict(
                    shared_state
                )


            self.send_json(
                snapshot
            )

            return


        # ====================================================
        # LOOMING
        # ====================================================

        if path == "/trigger":

            with state_lock:

                commands[
                    "looming_ms"
                ] = max(

                    commands[
                        "looming_ms"
                    ],

                    LOOMING_DURATION_MS

                )


            self.send_json(

                {
                    "ok": True,
                    "event": "looming"
                }

            )

            return


        # ====================================================
        # ENDOGENOUS
        # ====================================================

        if path == "/internal":

            with state_lock:

                commands[
                    "endogenous_ms"
                ] = max(

                    commands[
                        "endogenous_ms"
                    ],

                    ENDOGENOUS_DURATION_MS

                )


            self.send_json(

                {
                    "ok": True,
                    "event": "endogenous"
                }

            )

            return


        # ====================================================
        # HTML
        # ====================================================

        if path == "/":

            self.send_bytes(

                HTML.encode(
                    "utf-8"
                ),

                "text/html; charset=utf-8"

            )

            return


        self.send_bytes(

            b"Not found",

            "text/plain; charset=utf-8",

            status=404

        )


# ============================================================
# START BRAIN THREAD
# ============================================================

brain_thread = threading.Thread(

    target=brain_loop,

    daemon=True

)


brain_thread.start()


# ============================================================
# SERVER
# ============================================================

server = ThreadingHTTPServer(

    (
        HOST,
        PORT
    ),

    OverlayHandler

)


print()
print("=" * 80)
print("SERVER V4 HAZIR")
print("=" * 80)

print()

print("OBS:")
print(
    f"http://{HOST}:{PORT}/"
)

print()

print("DEBUG:")
print(
    f"http://{HOST}:{PORT}/?debug=1"
)

print()

print(
    "Flight controller:",
    FLIGHT_CONTROL_MS,
    "ms"
)

print(
    "Flight target Y:",
    FLIGHT_TARGET_Y
)

print()

print(
    "Raw neural action ve body state birbirinden ayrı tutuluyor."
)

print("=" * 80)


try:

    server.serve_forever()


except KeyboardInterrupt:

    print(
        "\nServer kapatılıyor..."
    )


finally:

    server.server_close()