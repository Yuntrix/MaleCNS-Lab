import math
import time
import tkinter as tk

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session import ContinuousBrainSession
from motor_decoder import MotorDecoder


# ============================================================
# AYARLAR
# ============================================================

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

GROUND_Y = 445

CHUNK_MS = 20.0

LOOMING_RATE_HZ = 40.0
LOOMING_PULSE_MS = 60.0

W_SYN = 0.110

SEED = 101


# ============================================================
# BRAIN
# ============================================================

print("=" * 70)
print("MALECNS CARTOON FLY")
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
# TKINTER
# ============================================================

root = tk.Tk()

root.title(
    "MaleCNS Fly - ADIM 2"
)

root.geometry(
    f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"
)

root.resizable(
    False,
    False
)


canvas = tk.Canvas(

    root,

    width=WINDOW_WIDTH,

    height=WINDOW_HEIGHT,

    bg="#20242b",

    highlightthickness=0

)

canvas.pack()


# ============================================================
# WORLD
# ============================================================

canvas.create_rectangle(

    0,
    GROUND_Y + 18,

    WINDOW_WIDTH,
    WINDOW_HEIGHT,

    fill="#343a40",

    outline=""

)


canvas.create_line(

    0,
    GROUND_Y + 18,

    WINDOW_WIDTH,
    GROUND_Y + 18,

    fill="#777777",

    width=3

)


canvas.create_text(

    WINDOW_WIDTH / 2,
    35,

    text="MALECNS CARTOON FLY",

    fill="white",

    font=(
        "Arial",
        20,
        "bold"
    )

)


canvas.create_text(

    WINDOW_WIDTH / 2,
    65,

    text="SPACE = looming sensory event",

    fill="#bbbbbb",

    font=(
        "Arial",
        11
    )

)


# ============================================================
# HUD
# ============================================================

hud_id = canvas.create_text(

    15,
    15,

    anchor="nw",

    fill="#ffffff",

    font=(
        "Consolas",
        12
    ),

    text=""

)


# ============================================================
# FLY STATE
# ============================================================

state = {

    "x":
        200.0,

    "y":
        float(
            GROUND_Y
        ),

    "vx":
        0.0,

    "vy":
        0.0,

    "on_ground":
        True,

    "facing":
        1,

    "wing_phase":
        0.0,

    "action":
        "IDLE",

    "stimulus_remaining_ms":
        0.0,

    "brain_time_ms":
        0.0,

    "next_tick":
        0.0,

    "last_action":
        "IDLE",

}


# ============================================================
# SENSOR EVENT
# ============================================================

def trigger_looming(event=None):

    state[
        "stimulus_remaining_ms"
    ] = max(

        state[
            "stimulus_remaining_ms"
        ],

        LOOMING_PULSE_MS

    )


    print(
        "\n>>> LOOMING EVENT"
    )

    print(
        f">>> {LOOMING_RATE_HZ:.0f} Hz x "
        f"{LOOMING_PULSE_MS:.0f} ms brain time"
    )


# ============================================================
# WORLD PHYSICS
#
# Buradaki hareket mesafeleri biyoloji değildir.
#
# Beyin:
# IDLE / MOVE / FLY / JUMP
#
# kararı üretir.
#
# Bu bölüm o motor komutunu ekrandaki piksel hareketine çeviren
# actuator/world mechanics katmanıdır.
# ============================================================

def update_world():

    action = state[
        "action"
    ]


    # ========================================================
    # FLY
    # ========================================================

    if action == "FLY":

        if state[
            "on_ground"
        ]:

            state[
                "vy"
            ] = -8.5

            state[
                "on_ground"
            ] = False


        state[
            "vy"
        ] -= 0.25


        state[
            "vx"
        ] += (

            0.65

            *

            state[
                "facing"
            ]

        )


        state[
            "vx"
        ] = max(

            -5.0,

            min(
                5.0,
                state[
                    "vx"
                ]
            )

        )


        state[
            "wing_phase"
        ] += 1.8


    # ========================================================
    # JUMP
    # ========================================================

    elif action == "JUMP":

        if state[
            "on_ground"
        ]:

            state[
                "vy"
            ] = -7.0

            state[
                "vx"
            ] += (

                2.5

                *

                state[
                    "facing"
                ]

            )

            state[
                "on_ground"
            ] = False


        state[
            "wing_phase"
        ] += 0.3


    # ========================================================
    # MOVE
    # ========================================================

    elif action == "MOVE":

        if state[
            "on_ground"
        ]:

            state[
                "vx"
            ] += (

                0.45

                *

                state[
                    "facing"
                ]

            )


            state[
                "vx"
            ] = max(

                -3.0,

                min(
                    3.0,
                    state[
                        "vx"
                    ]
                )

            )


        state[
            "wing_phase"
        ] += 0.15


    # ========================================================
    # IDLE
    # ========================================================

    else:

        if state[
            "on_ground"
        ]:

            state[
                "vx"
            ] *= 0.80


        state[
            "wing_phase"
        ] += 0.05


    # ========================================================
    # GRAVITY
    # ========================================================

    if not state[
        "on_ground"
    ]:

        state[
            "vy"
        ] += 0.55


        state[
            "vy"
        ] = min(

            state[
                "vy"
            ],

            8.0

        )


    # ========================================================
    # POSITION
    # ========================================================

    state[
        "x"
    ] += state[
        "vx"
    ]


    state[
        "y"
    ] += state[
        "vy"
    ]


    # ========================================================
    # GROUND COLLISION
    # ========================================================

    if state[
        "y"
    ] >= GROUND_Y:

        state[
            "y"
        ] = float(
            GROUND_Y
        )

        state[
            "vy"
        ] = 0.0

        state[
            "on_ground"
        ] = True


        state[
            "vx"
        ] *= 0.92


    # ========================================================
    # LEFT WALL
    # ========================================================

    if state[
        "x"
    ] < 45:

        state[
            "x"
        ] = 45.0

        state[
            "vx"
        ] = abs(
            state[
                "vx"
            ]
        )

        state[
            "facing"
        ] = 1


    # ========================================================
    # RIGHT WALL
    # ========================================================

    if state[
        "x"
    ] > WINDOW_WIDTH - 45:

        state[
            "x"
        ] = float(
            WINDOW_WIDTH - 45
        )

        state[
            "vx"
        ] = -abs(
            state[
                "vx"
            ]
        )

        state[
            "facing"
        ] = -1


# ============================================================
# CARTOON FLY DRAW
# ============================================================

def draw_fly():

    canvas.delete(
        "fly"
    )


    x = state[
        "x"
    ]

    y = state[
        "y"
    ] - 18


    facing = state[
        "facing"
    ]


    wing_motion = (

        math.sin(

            state[
                "wing_phase"
            ]

        )

        *

        8.0

    )


    # ========================================================
    # SHADOW
    # ========================================================

    distance_from_ground = (

        GROUND_Y

        -

        state[
            "y"
        ]

    )


    shadow_width = max(

        12,

        32
        -
        distance_from_ground
        *
        0.04

    )


    canvas.create_oval(

        x - shadow_width,
        GROUND_Y + 10,

        x + shadow_width,
        GROUND_Y + 16,

        fill="#111111",

        outline="",

        tags="fly"

    )


    # ========================================================
    # WINGS
    # ========================================================

    canvas.create_oval(

        x - 15,
        y - 30 - wing_motion,

        x + 10,
        y - 3 - wing_motion,

        fill="#d8f3ff",

        outline="#8ecae6",

        width=2,

        tags="fly"

    )


    canvas.create_oval(

        x - 17,
        y + 2 + wing_motion,

        x + 8,
        y + 24 + wing_motion,

        fill="#d8f3ff",

        outline="#8ecae6",

        width=2,

        tags="fly"

    )


    # ========================================================
    # LEGS
    # ========================================================

    leg_color = "#1c1c1c"


    canvas.create_line(

        x - 10,
        y + 6,

        x - 22,
        y + 25,

        fill=leg_color,

        width=3,

        tags="fly"

    )


    canvas.create_line(

        x,
        y + 7,

        x - 3,
        y + 28,

        fill=leg_color,

        width=3,

        tags="fly"

    )


    canvas.create_line(

        x + 10,
        y + 5,

        x + 22,
        y + 24,

        fill=leg_color,

        width=3,

        tags="fly"

    )


    # ========================================================
    # BODY
    # ========================================================

    canvas.create_oval(

        x - 20,
        y - 11,

        x + 20,
        y + 11,

        fill="#353535",

        outline="#111111",

        width=3,

        tags="fly"

    )


    # Abdomen stripes

    canvas.create_line(

        x - 8,
        y - 9,

        x - 8,
        y + 9,

        fill="#d4a017",

        width=3,

        tags="fly"

    )


    canvas.create_line(

        x,
        y - 10,

        x,
        y + 10,

        fill="#d4a017",

        width=3,

        tags="fly"

    )


    # ========================================================
    # HEAD
    # ========================================================

    head_x = (

        x

        +

        facing
        *
        22

    )


    canvas.create_oval(

        head_x - 11,
        y - 11,

        head_x + 11,
        y + 11,

        fill="#444444",

        outline="#111111",

        width=3,

        tags="fly"

    )


    # ========================================================
    # EYES
    # ========================================================

    eye_x = (

        head_x

        +

        facing
        *
        5

    )


    canvas.create_oval(

        eye_x - 5,
        y - 7,

        eye_x + 5,
        y + 1,

        fill="#e63946",

        outline="#7f0000",

        width=2,

        tags="fly"

    )


    canvas.create_oval(

        eye_x - 5,
        y + 2,

        eye_x + 5,
        y + 9,

        fill="#e63946",

        outline="#7f0000",

        width=2,

        tags="fly"

    )


    # ========================================================
    # ANTENNA
    # ========================================================

    canvas.create_line(

        head_x
        +
        facing
        *
        5,

        y - 8,

        head_x
        +
        facing
        *
        16,

        y - 18,

        fill="#111111",

        width=2,

        tags="fly"

    )


# ============================================================
# HUD
# ============================================================

def update_hud(
    input_rate,
    result,
    movement,
    compute_ms
):

    text = (

        f"BRAIN TIME : "
        f"{state['brain_time_ms'] / 1000.0:6.2f} s\n"

        f"INPUT      : "
        f"{input_rate:5.0f} Hz\n"

        f"ACTION     : "
        f"{movement['action']}\n"

        f"MOTOR      : "
        f"{result['motor_spikes']}\n"

        f"FLIGHT     : "
        f"{movement['flight_spikes']}\n"

        f"JUMP       : "
        f"{movement['jump_spikes']}\n"

        f"CALC       : "
        f"{compute_ms:5.1f} ms"

    )


    canvas.itemconfig(

        hud_id,

        text=text

    )


# ============================================================
# BRAIN TICK
# ============================================================

def brain_tick():

    # ========================================================
    # INPUT
    # ========================================================

    if (

        state[
            "stimulus_remaining_ms"
        ]

        >

        0

    ):

        input_rate = (
            LOOMING_RATE_HZ
        )

    else:

        input_rate = 0.0


    # ========================================================
    # SIMULATION
    # ========================================================

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


    # ========================================================
    # STIMULUS TIMER
    # ========================================================

    if (

        state[
            "stimulus_remaining_ms"
        ]

        >

        0

    ):

        state[
            "stimulus_remaining_ms"
        ] -= CHUNK_MS


        state[
            "stimulus_remaining_ms"
        ] = max(

            0.0,

            state[
                "stimulus_remaining_ms"
            ]

        )


    # ========================================================
    # MOTOR DECODER
    # ========================================================

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


    state[
        "action"
    ] = movement[
        "action"
    ]


    # ========================================================
    # CONSOLE - SADECE ACTION DEĞİŞİRSE
    # ========================================================

    if (

        state[
            "action"
        ]

        !=

        state[
            "last_action"
        ]

    ):

        print(

            f"BRAIN ACTION: "
            f"{state['last_action']} "
            f"-> "
            f"{state['action']}"

        )


        state[
            "last_action"
        ] = state[
            "action"
        ]


    # ========================================================
    # WORLD
    # ========================================================

    update_world()

    draw_fly()


    update_hud(

        input_rate,

        result,

        movement,

        compute_ms

    )


    state[
        "brain_time_ms"
    ] += CHUNK_MS


    # ========================================================
    # REAL-TIME PACING
    # ========================================================

    state[
        "next_tick"
    ] += (

        CHUNK_MS

        /

        1000.0

    )


    delay_seconds = (

        state[
            "next_tick"
        ]

        -

        time.perf_counter()

    )


    delay_ms = max(

        1,

        int(
            delay_seconds
            *
            1000
        )

    )


    root.after(

        delay_ms,

        brain_tick

    )


# ============================================================
# KEYBOARD
# ============================================================

root.bind(

    "<space>",

    trigger_looming

)


root.bind(

    "<Escape>",

    lambda event:
        root.destroy()

)


# ============================================================
# FIRST DRAW
# ============================================================

draw_fly()


# ============================================================
# AUTO TEST
#
# Program açıldıktan 2 saniye sonra bir kez otomatik
# looming event gönderiyoruz.
# ============================================================

root.after(

    2000,

    trigger_looming

)


# ============================================================
# START
# ============================================================

state[
    "next_tick"
] = time.perf_counter()


print("\n" + "=" * 70)
print("ADIM 2 VISUAL TEST BAŞLADI")
print("=" * 70)

print(
    "2 saniye sonra otomatik looming gelecek."
)

print(
    "Tekrar test etmek için SPACE."
)

print(
    "Kapatmak için ESC."
)


brain_tick()

root.mainloop()