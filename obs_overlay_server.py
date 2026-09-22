import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session import ContinuousBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# SERVER
# ============================================================

HOST = "127.0.0.1"
PORT = 8765


# ============================================================
# BRAIN SETTINGS
# ============================================================

CHUNK_MS = 20.0

W_SYN = 0.110

LOOMING_RATE_HZ = 40.0
LOOMING_PULSE_MS = 60.0

RUNAWAY_SPIKES_PER_CHUNK = 10_000

SEED = 101


# ============================================================
# SHARED STATE
# ============================================================

state_lock = threading.Lock()


shared_state = {

    "brain_time_ms": 0.0,

    "input_rate": 0.0,

    "action": "IDLE",

    "total_spikes": 0,

    "descending_spikes": 0,

    "motor_spikes": 0,

    "flight_spikes": 0,

    "jump_spikes": 0,

    "other_spikes": 0,

    "compute_ms": 0.0,

    "runaway": False,

    "stimulus_remaining_ms": 0.0,

}


# ============================================================
# BRAIN
# ============================================================

print("=" * 70)
print("MALECNS OBS OVERLAY SERVER")
print("=" * 70)


brain = RealtimeMaleCNSLIF()

brain.w_syn = W_SYN


session = ContinuousBrainSession(

    brain,

    source_types=[
        "LPLC2",
        "LC4"
    ],

    seed=SEED

)


decoder = MotorDecoder()


# ============================================================
# WARMUP
# ============================================================

print("\nBrain warmup...")


session.step(

    rate_hz=0,

    chunk_ms=1.0

)


session.reset(

    seed=SEED

)


print("Brain hazır.")
print("W_SYN:", brain.w_syn)


# ============================================================
# TRIGGER
# ============================================================

def trigger_looming():

    with state_lock:

        shared_state[
            "stimulus_remaining_ms"
        ] = max(

            shared_state[
                "stimulus_remaining_ms"
            ],

            LOOMING_PULSE_MS

        )


    print(
        "\n>>> OBS LOOMING EVENT"
    )

    print(
        f">>> {LOOMING_RATE_HZ:.0f} Hz x "
        f"{LOOMING_PULSE_MS:.0f} ms brain time"
    )


# ============================================================
# BRAIN LOOP
# ============================================================

def brain_loop():

    next_tick = time.perf_counter()


    last_action = "IDLE"


    while True:

        # ====================================================
        # INPUT
        # ====================================================

        with state_lock:

            remaining = shared_state[
                "stimulus_remaining_ms"
            ]


        if remaining > 0:

            input_rate = LOOMING_RATE_HZ

        else:

            input_rate = 0.0


        # ====================================================
        # SIMULATION
        # ====================================================

        compute_start = time.perf_counter()


        result = session.step(

            rate_hz=input_rate,

            chunk_ms=CHUNK_MS

        )


        compute_ms = (

            time.perf_counter()

            -

            compute_start

        ) * 1000.0


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


        action = movement[
            "action"
        ]


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
        # UPDATE SHARED STATE
        # ====================================================

        with state_lock:

            if (

                shared_state[
                    "stimulus_remaining_ms"
                ]

                >

                0

            ):

                shared_state[
                    "stimulus_remaining_ms"
                ] -= CHUNK_MS


                shared_state[
                    "stimulus_remaining_ms"
                ] = max(

                    0.0,

                    shared_state[
                        "stimulus_remaining_ms"
                    ]

                )


            shared_state[
                "brain_time_ms"
            ] += CHUNK_MS


            shared_state[
                "input_rate"
            ] = input_rate


            shared_state[
                "action"
            ] = action


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
                "flight_spikes"
            ] = movement[
                "flight_spikes"
            ]


            shared_state[
                "jump_spikes"
            ] = movement[
                "jump_spikes"
            ]


            shared_state[
                "other_spikes"
            ] = movement[
                "other_spikes"
            ]


            shared_state[
                "compute_ms"
            ] = compute_ms


            shared_state[
                "runaway"
            ] = runaway


        # ====================================================
        # ACTION LOG
        # ====================================================

        if action != last_action:

            print(

                "BRAIN ACTION:",

                last_action,

                "->",

                action

            )


            last_action = action


        # ====================================================
        # SAFETY
        # ====================================================

        if runaway:

            print("\n" + "!" * 70)

            print(
                "RUNAWAY ALGILANDI!"
            )

            print(
                "Chunk spike:",
                result[
                    "total_spikes"
                ]
            )

            print(
                "Brain loop durduruluyor."
            )

            print(
                "!" * 70
            )

            return


        # ====================================================
        # REALTIME PACING
        # ====================================================

        next_tick += (

            CHUNK_MS

            /

            1000.0

        )


        sleep_time = (

            next_tick

            -

            time.perf_counter()

        )


        if sleep_time > 0:

            time.sleep(
                sleep_time
            )

        else:

            # Geride kalırsak schedule'ı duvara bindirmiyoruz.
            next_tick = time.perf_counter()


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>MaleCNS Fly Overlay</title>


<style>

html,
body {

    margin: 0;

    padding: 0;

    width: 100%;

    height: 100%;

    overflow: hidden;

    background: transparent;

    font-family: Arial, sans-serif;

}


#world {

    position: relative;

    width: 100vw;

    height: 100vh;

    overflow: hidden;

    background: transparent;

}


#ground {

    position: absolute;

    left: 0;

    right: 0;

    bottom: 60px;

    height: 2px;

    background: rgba(255,255,255,0.0);

}


.debug #ground {

    background: rgba(255,255,255,0.35);

}


/* ==========================================================
   FLY
   ========================================================== */

#fly {

    position: absolute;

    width: 70px;

    height: 55px;

    transform-origin: center center;

    pointer-events: none;

}


.body {

    position: absolute;

    left: 20px;

    top: 18px;

    width: 34px;

    height: 22px;

    border-radius: 50%;

    background: #333;

    border: 3px solid #111;

    box-sizing: border-box;

}


.stripe1,
.stripe2 {

    position: absolute;

    top: 21px;

    width: 3px;

    height: 16px;

    background: #d5a419;

}


.stripe1 {

    left: 31px;

}


.stripe2 {

    left: 40px;

}


.head {

    position: absolute;

    left: 50px;

    top: 18px;

    width: 22px;

    height: 22px;

    border-radius: 50%;

    background: #444;

    border: 3px solid #111;

    box-sizing: border-box;

}


.eye {

    position: absolute;

    left: 64px;

    top: 22px;

    width: 9px;

    height: 13px;

    border-radius: 50%;

    background: #e63946;

    border: 2px solid #780000;

    box-sizing: border-box;

}


.wing {

    position: absolute;

    left: 19px;

    width: 30px;

    height: 25px;

    border-radius: 50%;

    background: rgba(210, 245, 255, 0.82);

    border: 2px solid #8ecae6;

    transform-origin: right center;

}


#wingTop {

    top: 1px;

    transform: rotate(-25deg);

}


#wingBottom {

    top: 33px;

    transform: rotate(25deg);

}


.leg {

    position: absolute;

    width: 3px;

    height: 25px;

    background: #111;

    transform-origin: top center;

}


#leg1 {

    left: 28px;

    top: 34px;

    transform: rotate(35deg);

}


#leg2 {

    left: 38px;

    top: 35px;

    transform: rotate(5deg);

}


#leg3 {

    left: 48px;

    top: 33px;

    transform: rotate(-35deg);

}


/* ==========================================================
   SHADOW
   ========================================================== */

#shadow {

    position: absolute;

    width: 42px;

    height: 8px;

    border-radius: 50%;

    background: rgba(0,0,0,0.28);

    pointer-events: none;

}


/* ==========================================================
   DEBUG HUD
   ========================================================== */

#debugPanel {

    display: none;

    position: absolute;

    left: 15px;

    top: 15px;

    padding: 10px 12px;

    border-radius: 8px;

    color: white;

    background: rgba(0,0,0,0.65);

    font-family: Consolas, monospace;

    font-size: 14px;

    white-space: pre;

}


.debug #debugPanel {

    display: block;

}


#triggerButton {

    display: none;

    position: absolute;

    right: 20px;

    top: 20px;

    padding: 12px 18px;

    border: 0;

    border-radius: 8px;

    cursor: pointer;

    font-size: 15px;

    font-weight: bold;

}


.debug #triggerButton {

    display: block;

}

</style>

</head>


<body>

<div id="world">

    <div id="ground"></div>

    <div id="shadow"></div>


    <div id="fly">

        <div id="wingTop"
             class="wing"></div>

        <div id="wingBottom"
             class="wing"></div>


        <div id="leg1"
             class="leg"></div>

        <div id="leg2"
             class="leg"></div>

        <div id="leg3"
             class="leg"></div>


        <div class="body"></div>

        <div class="stripe1"></div>

        <div class="stripe2"></div>

        <div class="head"></div>

        <div class="eye"></div>

    </div>


    <div id="debugPanel"></div>


    <button id="triggerButton">

        TEST LOOMING

    </button>

</div>


<script>

// ============================================================
// DEBUG MODE
//
// Normal:
// http://127.0.0.1:8765
//
// Debug:
// http://127.0.0.1:8765/?debug=1
// ============================================================

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


// ============================================================
// ELEMENTS
// ============================================================

const fly =
    document.getElementById(
        "fly"
    );


const shadow =
    document.getElementById(
        "shadow"
    );


const wingTop =
    document.getElementById(
        "wingTop"
    );


const wingBottom =
    document.getElementById(
        "wingBottom"
    );


const debugPanel =
    document.getElementById(
        "debugPanel"
    );


const triggerButton =
    document.getElementById(
        "triggerButton"
    );


// ============================================================
// WORLD STATE
// ============================================================

let x = 220;

let y = 0;

let vx = 0;

let vy = 0;

let facing = 1;

let onGround = true;

let wingPhase = 0;

let brainState = {

    action: "IDLE",

    motor_spikes: 0,

    flight_spikes: 0,

    jump_spikes: 0,

    total_spikes: 0,

    brain_time_ms: 0,

    compute_ms: 0,

    input_rate: 0

};


function groundY() {

    return (
        window.innerHeight
        -
        120
    );

}


y = groundY();


// ============================================================
// TRIGGER BUTTON
// ============================================================

triggerButton.addEventListener(

    "click",

    async () => {

        await fetch(
            "/trigger"
        );

    }

);


// ============================================================
// BRAIN STATE FETCH
// ============================================================

async function fetchBrainState() {

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

    }

    catch (error) {

        console.error(
            error
        );

    }


    setTimeout(

        fetchBrainState,

        20

    );

}


fetchBrainState();


// ============================================================
// WORLD PHYSICS
//
// Brain decides:
// IDLE / MOVE / FLY / JUMP
//
// JavaScript only converts that motor state into screen physics.
// ============================================================

function updateWorld() {

    const action =
        brainState.action;


    // ========================================================
    // FLY
    // ========================================================

    if (action === "FLY") {

        if (onGround) {

            vy = -8.5;

            onGround = false;

        }


        vy -= 0.22;


        vx += (
            0.55
            *
            facing
        );


        vx =
            Math.max(
                -5,
                Math.min(
                    5,
                    vx
                )
            );


        wingPhase += 1.7;

    }


    // ========================================================
    // JUMP
    // ========================================================

    else if (action === "JUMP") {

        if (onGround) {

            vy = -7;

            vx += (
                2.5
                *
                facing
            );

            onGround = false;

        }


        wingPhase += 0.25;

    }


    // ========================================================
    // MOVE
    // ========================================================

    else if (action === "MOVE") {

        if (onGround) {

            vx += (
                0.35
                *
                facing
            );


            vx =
                Math.max(
                    -2.8,
                    Math.min(
                        2.8,
                        vx
                    )
                );

        }


        wingPhase += 0.15;

    }


    // ========================================================
    // IDLE
    // ========================================================

    else {

        if (onGround) {

            vx *= 0.82;

        }


        wingPhase += 0.04;

    }


    // ========================================================
    // GRAVITY
    // ========================================================

    if (!onGround) {

        vy += 0.50;


        vy =
            Math.min(
                8,
                vy
            );

    }


    // ========================================================
    // POSITION
    // ========================================================

    x += vx;

    y += vy;


    const gy =
        groundY();


    // ========================================================
    // FLOOR
    // ========================================================

    if (y >= gy) {

        y = gy;

        vy = 0;

        onGround = true;

        vx *= 0.92;

    }


    // ========================================================
    // WALLS
    // ========================================================

    if (x < 15) {

        x = 15;

        vx = Math.abs(
            vx
        );

        facing = 1;

    }


    if (
        x
        >
        window.innerWidth
        -
        90
    ) {

        x =
            window.innerWidth
            -
            90;

        vx =
            -Math.abs(
                vx
            );

        facing = -1;

    }


    // ========================================================
    // DRAW
    // ========================================================

    const scaleX =
        facing;


    fly.style.transform =
        `translate(${x}px, ${y}px) scaleX(${scaleX})`;


    const height =
        Math.max(
            0,
            gy - y
        );


    const shadowScale =
        Math.max(
            0.35,
            1 - height / 300
        );


    shadow.style.transform =
        `translate(${x + 15}px, ${gy + 58}px) scaleX(${shadowScale})`;


    // ========================================================
    // WINGS
    // ========================================================

    const wingMotion =
        Math.sin(
            wingPhase
        )
        *
        25;


    wingTop.style.transform =
        `rotate(${-25 - wingMotion}deg)`;


    wingBottom.style.transform =
        `rotate(${25 + wingMotion}deg)`;


    // ========================================================
    // DEBUG
    // ========================================================

    if (debug) {

        debugPanel.textContent =

            `ACTION     : ${brainState.action}\n`

            +

            `INPUT      : ${brainState.input_rate} Hz\n`

            +

            `MOTOR      : ${brainState.motor_spikes}\n`

            +

            `FLIGHT     : ${brainState.flight_spikes}\n`

            +

            `JUMP       : ${brainState.jump_spikes}\n`

            +

            `SPIKES     : ${brainState.total_spikes}\n`

            +

            `BRAIN TIME : ${(brainState.brain_time_ms / 1000).toFixed(2)} s\n`

            +

            `CALC       : ${brainState.compute_ms.toFixed(2)} ms`;

    }


    requestAnimationFrame(
        updateWorld
    );

}


requestAnimationFrame(
    updateWorld
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

        # Terminal spam olmasın.
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
            "Access-Control-Allow-Origin",
            "*"
        )


        self.end_headers()


        self.wfile.write(
            data
        )


    def do_GET(
        self
    ):

        path = urlparse(
            self.path
        ).path


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


        # ====================================================
        # STATE
        # ====================================================

        if path == "/state":

            with state_lock:

                payload = dict(
                    shared_state
                )


            self.send_bytes(

                json.dumps(
                    payload
                ).encode(
                    "utf-8"
                ),

                "application/json; charset=utf-8"

            )

            return


        # ====================================================
        # MANUAL TEST EVENT
        # ====================================================

        if path == "/trigger":

            trigger_looming()


            payload = {

                "ok": True,

                "pulse_ms":
                    LOOMING_PULSE_MS,

                "rate_hz":
                    LOOMING_RATE_HZ

            }


            self.send_bytes(

                json.dumps(
                    payload
                ).encode(
                    "utf-8"
                ),

                "application/json; charset=utf-8"

            )

            return


        # ====================================================
        # 404
        # ====================================================

        self.send_bytes(

            b"Not Found",

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
# START SERVER
# ============================================================

server = ThreadingHTTPServer(

    (
        HOST,
        PORT
    ),

    OverlayHandler

)


print("\n" + "=" * 70)
print("OBS OVERLAY HAZIR")
print("=" * 70)

print(
    "Normal overlay:"
)

print(
    f"http://{HOST}:{PORT}/"
)

print()

print(
    "Debug/test overlay:"
)

print(
    f"http://{HOST}:{PORT}/?debug=1"
)

print()

print(
    "Önce Chrome'da debug adresini aç."
)

print(
    "Kapatmak için PyCharm'da kırmızı STOP."
)


try:

    server.serve_forever()


except KeyboardInterrupt:

    pass


finally:

    server.server_close()