import json
import tkinter as tk

import mss


# ============================================================
# MALECNS VISION ROI SELECTOR
#
# OBS Fullscreen Projector açıkken
# sineğin görmesini istediğimiz alanı mouse ile seçer.
#
# Seçim vision_roi.json dosyasına kaydedilir.
# ============================================================


MONITOR_INDEX = 1

OUTPUT_FILE = "vision_roi.json"


with mss.MSS() as sct:

    print("=" * 80)
    print("MALECNS VISION ROI SELECTOR")
    print("=" * 80)

    print()
    print("Detected monitors:")

    for i, monitor in enumerate(
        sct.monitors
    ):

        print(
            f"[{i}] "
            f"{monitor['width']}x{monitor['height']} "
            f"left={monitor['left']} "
            f"top={monitor['top']}"
        )

    if MONITOR_INDEX >= len(
        sct.monitors
    ):

        raise RuntimeError(
            f"Monitor {MONITOR_INDEX} bulunamadı."
        )

    monitor = sct.monitors[
        MONITOR_INDEX
    ]


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


root = tk.Tk()

root.overrideredirect(
    True
)

root.attributes(
    "-topmost",
    True
)

root.attributes(
    "-alpha",
    0.30
)

root.geometry(
    f"{monitor_width}x{monitor_height}"
    f"+{monitor_left}+{monitor_top}"
)


canvas = tk.Canvas(
    root,
    width=monitor_width,
    height=monitor_height,
    bg="black",
    cursor="cross"
)

canvas.pack(
    fill="both",
    expand=True
)


instruction = canvas.create_text(
    monitor_width // 2,
    50,
    text=(
        "SİNEĞİN GÖRECEĞİ ALANI SEÇ\n"
        "Mouse ile kamera / world alanının etrafına kutu çiz\n"
        "ESC = iptal"
    ),
    fill="white",
    font=(
        "Arial",
        18,
        "bold"
    ),
    justify="center"
)


start_x = None
start_y = None

rectangle_id = None


def mouse_down(event):

    global start_x
    global start_y
    global rectangle_id

    start_x = event.x
    start_y = event.y

    if rectangle_id is not None:

        canvas.delete(
            rectangle_id
        )

    rectangle_id = canvas.create_rectangle(
        start_x,
        start_y,
        start_x,
        start_y,
        outline="red",
        width=4
    )


def mouse_move(event):

    if (
        start_x is None
        or
        start_y is None
        or
        rectangle_id is None
    ):

        return

    canvas.coords(
        rectangle_id,
        start_x,
        start_y,
        event.x,
        event.y
    )


def mouse_up(event):

    x1 = min(
        start_x,
        event.x
    )

    y1 = min(
        start_y,
        event.y
    )

    x2 = max(
        start_x,
        event.x
    )

    y2 = max(
        start_y,
        event.y
    )


    width = x2 - x1
    height = y2 - y1


    if (
        width < 100
        or
        height < 100
    ):

        print(
            "Seçim çok küçük. Tekrar çiz."
        )

        return


    roi = {

        "monitor_index":
            MONITOR_INDEX,

        "monitor_left":
            monitor_left,

        "monitor_top":
            monitor_top,

        "relative_left":
            x1,

        "relative_top":
            y1,

        "left":
            monitor_left + x1,

        "top":
            monitor_top + y1,

        "width":
            width,

        "height":
            height

    }


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            roi,
            file,
            indent=4
        )


    print()
    print("=" * 80)

    print(
        "ROI KAYDEDİLDİ"
    )

    print("=" * 80)

    print(
        f"Monitor: {MONITOR_INDEX}"
    )

    print(
        f"Relative x/y: {x1}, {y1}"
    )

    print(
        f"Boyut: {width} x {height}"
    )

    print(
        f"Dosya: {OUTPUT_FILE}"
    )

    print("=" * 80)


    root.destroy()


def cancel(event=None):

    print(
        "ROI seçimi iptal edildi."
    )

    root.destroy()


canvas.bind(
    "<ButtonPress-1>",
    mouse_down
)

canvas.bind(
    "<B1-Motion>",
    mouse_move
)

canvas.bind(
    "<ButtonRelease-1>",
    mouse_up
)

root.bind(
    "<Escape>",
    cancel
)


print()
print(
    "OBS Fullscreen Projector üzerinde"
)

print(
    "kameranın / world görüntüsünün etrafına mouse ile kutu çiz."
)

print()


root.mainloop()