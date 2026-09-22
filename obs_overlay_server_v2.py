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


# ============================================================
# LOOMING
# ============================================================

LOOMING_RATE_HZ = 40.0

LOOMING_DURATION_MS = 60.0


# ============================================================
# MODELED ENDOGENOUS DRIVE
#
# Calibration sonucu:
#
# 2 Hz x 200 ms
#
# 1 Hz -> motor yok
# 5/10 Hz -> persistent activity
#
# Bu yüzden 2 Hz kullanıyoruz.
# ============================================================

ENDOGENOUS_RATE_HZ = 2.0

ENDOGENOUS_DURATION_MS = 200.0

ENDOGENOUS_NEURON_COUNT = 128


# ============================================================
# ENDOGENOUS BURST ARALIĞI
#
# İlk burst hızlı geliyor ki test ederken görelim.
#
# Sonrakiler:
# 8 - 18 saniye arasında modeled internal interval.
#
# Bu interval biyolojik ölçüm değildir.
# Autonomous agent için modeled internal drive'dır.
# ============================================================

FIRST_ENDOGENOUS_AT_MS = 3000.0

ENDOGENOUS_INTERVAL_MIN_MS = 8000.0

ENDOGENOUS_INTERVAL_MAX_MS = 18000.0


# ============================================================
# VISUAL ACTION HOLD
#
# Brain action'ı değiştirmez.
#
# Tek 20 ms motor cevabının animasyonda görülebilmesi için
# actuator / animation smoothing.
# ============================================================

VISUAL_ACTION_HOLD_MS = 450.0


# ============================================================
# SAFETY
# ============================================================

RUNAWAY_SPIKES_PER_CHUNK = 10000


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

    "action": "IDLE",

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
# BRAIN LOAD
# ============================================================

print("=" * 80)
print("MALECNS AUTONOMOUS OBS SERVER V2")
print("=" * 80)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


decoder = MotorDecoder()


print()

print(
    "Nöron:",
    brain.N
)

print(
    "Bağlantı:",
    len(
        brain.indices
    )
)

print(
    "W_SYN:",
    brain.w_syn
)


# ============================================================
# BUILD / LOAD ENDOGENOUS POPULATION
# ============================================================

def build_endogenous_population():

    # --------------------------------------------------------
    # Cache varsa direkt kullan.
    # Böylece her server başlangıcında connectome taramıyoruz.
    # --------------------------------------------------------

    if ENDOGENOUS_CACHE.exists():

        cached = np.load(
            ENDOGENOUS_CACHE
        ).astype(
            np.int32
        )


        if len(cached) == ENDOGENOUS_NEURON_COUNT:

            print(
                "Endogenous population cache yüklendi:",
                ENDOGENOUS_CACHE
            )

            return cached


    print(
        "Endogenous population ilk kez hazırlanıyor..."
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


    # --------------------------------------------------------
    # cb_intrinsic + fast_excitatory
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Hub filtresi
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Connectome'da DN bağlantısını ölç
    # --------------------------------------------------------

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
            direct_dn_targets
            >
            0
        )

        &

        (
            out_degree
            >=
            out_low
        )

        &

        (
            out_degree
            <=
            out_high
        )

    )


    candidate_indices = np.flatnonzero(
        candidate_mask
    )


    scores = np.zeros(

        brain.N,

        dtype=np.float64

    )


    scores[
        candidate_indices
    ] = (

        direct_dn_strength[
            candidate_indices
        ]

        /

        np.sqrt(

            np.maximum(

                out_degree[
                    candidate_indices
                ],

                1

            )

        )

    )


    order = np.argsort(

        scores[
            candidate_indices
        ]

    )[::-1]


    ranked = candidate_indices[
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
        "Endogenous population kaydedildi:",
        ENDOGENOUS_CACHE
    )


    print(
        "Selected:",
        len(
            selected
        )
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
    len(
        endogenous_indices
    )
)


print(
    "Looming neurons:",
    len(
        looming_indices
    )
)


# ============================================================
# MULTI INPUT SESSION
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
# NUMBA WARMUP
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


    visual_action = "IDLE"

    visual_action_until_ms = 0.0


    max_compute_ms = 0.0


    chunk_seconds = (

        CHUNK_MS

        /

        1000.0

    )


    next_tick = (
        time.perf_counter()
    )


    print()

    print(
        "Brain loop başladı."
    )


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


            interval = rng.uniform(

                ENDOGENOUS_INTERVAL_MIN_MS,

                ENDOGENOUS_INTERVAL_MAX_MS

            )


            next_endogenous_ms = (

                brain_time_ms

                +

                interval

            )


            print(

                f"[AUTO INTERNAL] "

                f"brain={brain_time_ms:.0f} ms "

                f"next={next_endogenous_ms:.0f} ms"

            )


        # ====================================================
        # CURRENT INPUT RATES
        # ====================================================

        endogenous_active = (

            automatic_endogenous_remaining
            >
            0.0

            or

            manual_endogenous_remaining
            >
            0.0

        )


        if endogenous_active:

            endogenous_rate = (
                ENDOGENOUS_RATE_HZ
            )

        else:

            endogenous_rate = 0.0


        if looming_remaining > 0.0:

            looming_rate = (
                LOOMING_RATE_HZ
            )

        else:

            looming_rate = 0.0


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
        # DECODER
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
        # VISUAL / ACTUATOR HOLD
        #
        # Neural state'e dokunmuyoruz.
        # Sadece animasyon görünür olsun diye output hold.
        # ====================================================

        if raw_action != "IDLE":

            visual_action = (
                raw_action
            )


            visual_action_until_ms = (

                brain_time_ms

                +

                VISUAL_ACTION_HOLD_MS

            )


        elif (
            brain_time_ms
            >=
            visual_action_until_ms
        ):

            visual_action = "IDLE"


        # ====================================================
        # RUNAWAY
        # ====================================================

        runaway = (

            result[
                "total_spikes"
            ]

            >
            RUNAWAY_SPIKES_PER_CHUNK

        )


        # ====================================================
        # INPUT TIMERS
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
                "action"
            ] = visual_action


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
            ] = (
                looming_rate > 0
            )


            shared_state[
                "endogenous_active"
            ] = (
                endogenous_rate > 0
            )


            shared_state[
                "endogenous_bursts"
            ] = (
                endogenous_burst_count
            )


            shared_state[
                "next_endogenous_ms"
            ] = round(
                next_endogenous_ms,
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
        # CONSOLE ACTION LOG
        # ====================================================

        if raw_action != "IDLE":

            print(

                f"[BRAIN] "

                f"{brain_time_ms:8.0f} ms"

                f" | raw={raw_action}"

                f" | spike={result['total_spikes']}"

                f" | DN={result['descending_spikes']}"

                f" | motor={result['motor_spikes']}"

                f" | calc={compute_ms:.2f} ms"

            )


        # ====================================================
        # ADVANCE BRAIN TIME
        # ====================================================

        brain_time_ms += (
            CHUNK_MS
        )


        # ====================================================
        # REALTIME PACING
        # ====================================================

        next_tick += (
            chunk_seconds
        )


        now = (
            time.perf_counter()
        )


        sleep_time = (

            next_tick

            -

            now

        )


        if sleep_time > 0:

            with state_lock:

                shared_state[
                    "backlog_ms"
                ] = 0.0


            time.sleep(
                sleep_time
            )


        else:

            backlog_ms = (

                -sleep_time

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

body {
    font-family: Arial, sans-serif;
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
    width: 72px;
    height: 54px;
    transform-origin: center center;
    will-change: transform;
}

.body {
    position: absolute;

    width: 38px;
    height: 27px;

    left: 17px;
    top: 16px;

    border-radius: 50% 55% 55% 50%;

    background:
        repeating-linear-gradient(
            90deg,
            #272727 0px,
            #272727 7px,
            #d69c25 7px,
            #d69c25 11px
        );

    border: 2px solid #111;
}

.head {
    position: absolute;

    width: 23px;
    height: 23px;

    left: 7px;
    top: 17px;

    border-radius: 50%;

    background: #353535;

    border: 2px solid #111;
}

.eye {
    position: absolute;

    width: 8px;
    height: 12px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle at 35% 30%,
            #ffb0b0 0%,
            #d51f28 30%,
            #72070b 100%
        );
}

.eye.left {
    left: 4px;
    top: 4px;
}

.eye.right {
    left: 12px;
    top: 5px;
}

.wing {
    position: absolute;

    width: 34px;
    height: 18px;

    border-radius: 70% 30% 70% 30%;

    background: rgba(
        185,
        235,
        255,
        0.58
    );

    border:
        1px solid
        rgba(
            90,
            150,
            180,
            0.75
        );

    transform-origin: 4px 9px;
}

.wing.left {
    left: 28px;
    top: 8px;

    transform:
        rotate(-24deg);
}

.wing.right {
    left: 28px;
    top: 28px;

    transform:
        rotate(24deg);
}

.leg {
    position: absolute;

    width: 29px;
    height: 2px;

    background: #181818;

    transform-origin: 2px 1px;
}

.leg.l1 {
    left: 24px;
    top: 35px;
    transform: rotate(35deg);
}

.leg.l2 {
    left: 28px;
    top: 39px;
    transform: rotate(65deg);
}

.leg.l3 {
    left: 31px;
    top: 40px;
    transform: rotate(100deg);
}

.antenna {
    position: absolute;

    width: 15px;
    height: 2px;

    background: #171717;

    transform-origin: right center;
}

.antenna.a1 {
    left: -4px;
    top: 18px;
    transform: rotate(25deg);
}

.antenna.a2 {
    left: -3px;
    top: 28px;
    transform: rotate(-20deg);
}


/* =========================================================
   ACTION ANIMATIONS
   ========================================================= */

#fly.FLY .wing.left {
    animation:
        wingLeft
        0.055s
        linear
        infinite;
}

#fly.FLY .wing.right {
    animation:
        wingRight
        0.055s
        linear
        infinite;
}

#fly.MOVE {
    animation:
        walkBob
        0.16s
        ease-in-out
        infinite alternate;
}

#fly.JUMP {
    animation:
        jumpStretch
        0.12s
        ease-out
        infinite alternate;
}


@keyframes wingLeft {

    0% {
        rotate: -35deg;
    }

    50% {
        rotate: 15deg;
    }

    100% {
        rotate: -35deg;
    }

}


@keyframes wingRight {

    0% {
        rotate: 35deg;
    }

    50% {
        rotate: -15deg;
    }

    100% {
        rotate: 35deg;
    }

}


@keyframes walkBob {

    from {
        margin-top: 0px;
    }

    to {
        margin-top: -3px;
    }

}


@keyframes jumpStretch {

    from {
        scale: 1 1;
    }

    to {
        scale: 0.94 1.08;
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

    left: 18px;
    top: 18px;

    width: 330px;

    padding: 14px;

    background:
        rgba(
            0,
            0,
            0,
            0.82
        );

    color: white;

    border-radius: 10px;

    font-family: Consolas, monospace;

    font-size: 13px;

    line-height: 1.45;

    z-index: 1000;

    white-space: pre-wrap;
}

#buttons {
    position: fixed;

    right: 20px;
    top: 20px;

    z-index: 1000;
}

button {
    display: block;

    margin-bottom: 10px;

    padding:
        12px
        18px;

    font-size: 15px;

    cursor: pointer;
}

</style>
</head>


<body>

<div id="world">

    <div id="fly">

        <div class="wing left"></div>
        <div class="wing right"></div>

        <div class="leg l1"></div>
        <div class="leg l2"></div>
        <div class="leg l3"></div>

        <div class="body"></div>

        <div class="head">

            <div class="eye left"></div>
            <div class="eye right"></div>

        </div>

        <div class="antenna a1"></div>
        <div class="antenna a2"></div>

    </div>

</div>


<div
    id="debugPanel"
    class="debug-ui"
>
loading...
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
    params.get("debug") === "1";


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


let brainState = {

    action: "IDLE",
    raw_action: "IDLE"

};


let x = 320;

let y = 0;


let vx = 0;

let vy = 0;


let facing = 1;


let initialized = false;


/* ==========================================================
   API
   ========================================================== */

async function triggerLooming() {

    try {

        await fetch(
            "/trigger"
        );

    } catch (e) {

    }

}


async function triggerInternal() {

    try {

        await fetch(
            "/internal"
        );

    } catch (e) {

    }

}


async function updateState() {

    try {

        const response =
            await fetch(
                "/state",
                {
                    cache: "no-store"
                }
            );


        brainState =
            await response.json();


        if (debug) {

            panel.textContent =

                "MALECNS AUTONOMOUS OBS\n\n"

                +

                "brain: "
                + brainState.brain_time_ms
                + " ms\n"

                +

                "raw action: "
                + brainState.raw_action
                + "\n"

                +

                "visual action: "
                + brainState.action
                + "\n\n"

                +

                "spikes: "
                + brainState.total_spikes
                + "\n"

                +

                "DN: "
                + brainState.descending_spikes
                + "\n"

                +

                "motor: "
                + brainState.motor_spikes
                + "\n"

                +

                "flight: "
                + brainState.flight
                + "\n"

                +

                "jump: "
                + brainState.jump
                + "\n"

                +

                "other: "
                + brainState.other
                + "\n\n"

                +

                "internal active: "
                + brainState.endogenous_active
                + "\n"

                +

                "internal bursts: "
                + brainState.endogenous_bursts
                + "\n"

                +

                "next internal: "
                + brainState.next_endogenous_ms
                + " ms\n"

                +

                "looming: "
                + brainState.looming_active
                + "\n\n"

                +

                "calc: "
                + brainState.compute_ms
                + " ms\n"

                +

                "max calc: "
                + brainState.max_compute_ms
                + " ms\n"

                +

                "backlog: "
                + brainState.backlog_ms
                + " ms\n"

                +

                "runaway: "
                + brainState.runaway;

        }

    } catch (e) {

    }

}


setInterval(

    updateState,

    50

);


/* ==========================================================
   WORLD PHYSICS
   ========================================================== */

function animate() {

    const width =
        window.innerWidth;


    const height =
        window.innerHeight;


    const ground =
        Math.max(
            80,
            height - 80
        );


    if (!initialized) {

        y = ground;

        initialized = true;

    }


    const action =
        brainState.action
        ||
        "IDLE";


    fly.className =
        action;


    if (action === "MOVE") {

        vx += (
            0.11
            *
            facing
        );


        vx =
            Math.max(
                -1.25,
                Math.min(
                    1.25,
                    vx
                )
            );

    }


    else if (action === "FLY") {

        vx += (
            0.12
            *
            facing
        );


        vx =
            Math.max(
                -1.8,
                Math.min(
                    1.8,
                    vx
                )
            );


        vy -= 0.22;


        vy =
            Math.max(
                -3.4,
                vy
            );

    }


    else if (action === "JUMP") {

        if (
            y
            >=
            ground - 2
        ) {

            vy = -5.0;

        }

    }


    else {

        vx *= 0.92;

    }


    /* gravity */

    vy += 0.12;


    vy =
        Math.min(
            4.0,
            vy
        );


    x += vx;

    y += vy;


    /* ground */

    if (y > ground) {

        y = ground;

        vy = 0;

    }


    /* ceiling */

    if (y < 10) {

        y = 10;

        vy = 0;

    }


    /* walls */

    const maxX =
        Math.max(
            20,
            width - 90
        );


    if (x < 10) {

        x = 10;

        facing = 1;

        vx =
            Math.abs(
                vx
            );

    }


    if (x > maxX) {

        x = maxX;

        facing = -1;

        vx =
            -Math.abs(
                vx
            );

    }


    const scaleX =
        facing;


    fly.style.transform =

        "translate("
        + x.toFixed(1)
        + "px,"
        + (y - 48).toFixed(1)
        + "px) "

        +

        "scaleX("
        + scaleX
        + ")";


    requestAnimationFrame(
        animate
    );

}


requestAnimationFrame(
    animate
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

        parsed = urlparse(
            self.path
        )


        path = parsed.path


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
        # MANUAL LOOMING
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
        # MANUAL INTERNAL BURST
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
        # OVERLAY
        # ====================================================

        if path == "/":

            data = HTML.encode(
                "utf-8"
            )


            self.send_bytes(

                data,

                "text/html; charset=utf-8"

            )

            return


        self.send_bytes(

            b"Not found",

            "text/plain; charset=utf-8",

            status=404

        )


# ============================================================
# START
# ============================================================

brain_thread = threading.Thread(

    target=brain_loop,

    daemon=True

)


brain_thread.start()


server = ThreadingHTTPServer(

    (
        HOST,
        PORT
    ),

    OverlayHandler

)


print()

print("=" * 80)
print("SERVER HAZIR")
print("=" * 80)

print()

print(
    f"OBS:"
)

print(
    f"http://{HOST}:{PORT}/"
)

print()

print(
    "DEBUG:"
)

print(
    f"http://{HOST}:{PORT}/?debug=1"
)

print()

print(
    "İlk automatic internal burst:"
    f" {FIRST_ENDOGENOUS_AT_MS / 1000:.1f} saniye"
)

print(
    "Sonraki interval:"
    f" {ENDOGENOUS_INTERVAL_MIN_MS / 1000:.0f}"
    " - "
    f"{ENDOGENOUS_INTERVAL_MAX_MS / 1000:.0f}"
    " saniye"
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