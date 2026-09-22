from __future__ import annotations

import argparse
import os
import random
import time
import tkinter as tk
from typing import List, Optional, Tuple

from body.character_controller import CharacterController
from world.obs_habitat import ObsHabitat
from world.prop_placement import (
    PropFootprint,
    PropPlacement,
    PropPlacementEngine,
    VerticalPreference,
)


WIDTH = 960
HEIGHT = 540

# Temporary prototype chair footprint.
CHAIR_WIDTH = 0.10
CHAIR_HEIGHT = 0.12

RECENT_HISTORY_LIMIT = 3


class WatchingDemo:
    def __init__(
        self,
        root: tk.Tk,
        seed: Optional[int] = None,
    ) -> None:
        self.root = root
        self.root.title("MaleCNS WATCHING prototype")
        self.root.geometry(f"{WIDTH}x{HEIGHT}")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=WIDTH,
            height=HEIGHT,
            bg="#20242b",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.canvas.create_rectangle(
            0,
            HEIGHT * 0.92,
            WIDTH,
            HEIGHT,
            fill="#343a40",
            outline="",
        )

        self.canvas.create_text(
            WIDTH / 2,
            28,
            text="WATCHING STATE PROTOTYPE  |  W: watch  E: exit",
            fill="white",
            font=("Arial", 16, "bold"),
        )

        self.controller = CharacterController()

        self.habitat: Optional[ObsHabitat] = None
        self.chair_placement: Optional[PropPlacement] = None

        self.recent_placements: List[
            Tuple[str, float, float]
        ] = []

        self.random = random.Random(seed)

        self.started = time.monotonic()
        self.last = self.started

        root.bind("<KeyPress-w>", self.start_watching)
        root.bind("<KeyPress-W>", self.start_watching)
        root.bind("<KeyPress-e>", self.exit_watching)
        root.bind("<KeyPress-E>", self.exit_watching)
        root.bind("<Escape>", lambda _event: self.close())

        self.tick()

    def now(self) -> float:
        return time.monotonic() - self.started

    @staticmethod
    def _placement_key(
        placement: PropPlacement,
    ) -> Tuple[str, float, float]:
        return (
            placement.surface_name,
            round(placement.x, 6),
            round(placement.y, 6),
        )

    @staticmethod
    def _distance(
        first: PropPlacement,
        second: Tuple[str, float, float],
    ) -> float:
        _, other_x, other_y = second

        dx = first.x - other_x
        dy = first.y - other_y

        return (dx * dx + dy * dy) ** 0.5

    def _weight(
        self,
        placement: PropPlacement,
        candidates: List[PropPlacement],
    ) -> float:
        generic_scores = [
            candidate.score
            for candidate in candidates
        ]

        minimum_score = min(generic_scores)
        maximum_score = max(generic_scores)
        score_range = maximum_score - minimum_score

        if score_range <= 1e-9:
            generic_factor = 1.0
        else:
            generic_factor = (
                placement.score - minimum_score
            ) / score_range

        weight = 1.0 + generic_factor * 1.5

        # Lower surfaces are preferred, but never required.
        weight += placement.y * 2.0

        # SCREEN_FLOOR is natural for WATCHING.
        if placement.surface_type.upper() == "SCREEN_FLOOR":
            weight += 2.0

        # Continuous center penalty.
        center_distance = abs(placement.x - 0.5)
        center_factor = max(
            0.0,
            1.0 - center_distance / 0.5,
        )
        weight *= 1.0 - center_factor * 0.60

        # Strongly reduce recently used positions.
        for index, recent in enumerate(
            reversed(self.recent_placements)
        ):
            distance = self._distance(placement, recent)

            if distance < 0.05:
                weight *= 0.08
            elif distance < 0.12:
                weight *= 0.35
            elif distance < 0.22:
                weight *= 0.70

            if recent[0] == placement.surface_name:
                weight *= 0.82

            if index == 0 and distance < 0.03:
                weight *= 0.05

        return max(weight, 0.001)

    def _choose_weighted_candidate(
        self,
        candidates: List[PropPlacement],
    ) -> Tuple[Optional[PropPlacement], List[Tuple[PropPlacement, float]]]:
        if not candidates:
            return None, []

        weighted = [
            (candidate, self._weight(candidate, candidates))
            for candidate in candidates
        ]

        weighted.sort(
            key=lambda item: (
                -item[1],
                item[0].surface_name,
                round(item[0].x, 6),
                round(item[0].y, 6),
            )
        )

        total_weight = sum(
            weight
            for _, weight in weighted
        )

        if total_weight <= 0.0:
            return weighted[0][0], weighted

        selected_value = (
            self.random.random() * total_weight
        )

        cumulative = 0.0

        for candidate, weight in weighted:
            cumulative += weight

            if selected_value <= cumulative:
                return candidate, weighted

        return weighted[-1][0], weighted

    def start_watching(self, _event=None) -> None:
        if self.controller.current_state.value != "NORMAL":
            return

        print()
        print("WATCHING REQUEST")
        print(
            f"chair footprint: "
            f"{CHAIR_WIDTH:.3f} x {CHAIR_HEIGHT:.3f}"
        )

        habitat: Optional[ObsHabitat] = None

        try:
            habitat = ObsHabitat(
                host=os.getenv(
                    "OBS_WEBSOCKET_HOST",
                    "127.0.0.1",
                ),
                port=int(
                    os.getenv(
                        "OBS_WEBSOCKET_PORT",
                        "7654",
                    )
                ),
                password=os.getenv(
                    "OBS_WEBSOCKET_PASSWORD"
                ),
            )

            habitat.connect()

            engine = PropPlacementEngine(habitat)

            request = PropFootprint(
                name="WATCHING_CHAIR",
                width=CHAIR_WIDTH,
                height=CHAIR_HEIGHT,
                safety_margin=0.01,
                preferred_surface_types=(
                    "SCREEN_FLOOR",
                    "OVERLAY_TOP",
                ),
                vertical_preference=VerticalPreference.LOW,
                avoid_screen_center=True,
                center_avoidance_strength=2.0,
            )

            candidates = engine.find_candidates(request)

            print(
                f"valid candidates: "
                f"{len(candidates)}"
            )

            if not candidates:
                print(
                    "WATCHING placement failed: "
                    "no valid chair placement"
                )
                print("WATCHING aborted")
                habitat.disconnect()
                return

            placement, weighted = (
                self._choose_weighted_candidate(candidates)
            )

            if placement is None:
                print(
                    "WATCHING placement failed: "
                    "no valid chair placement"
                )
                print("WATCHING aborted")
                habitat.disconnect()
                return

            self.habitat = habitat
            self.chair_placement = placement

            key = self._placement_key(placement)
            self.recent_placements.append(key)

            if len(self.recent_placements) > RECENT_HISTORY_LIMIT:
                self.recent_placements.pop(0)

            selected_weight = next(
                weight
                for candidate, weight in weighted
                if self._placement_key(candidate) == key
            )

            print("selected:")
            print(f"surface: {placement.surface_name}")
            print(
                f"region: {self._region(placement.x)}"
            )
            print(
                f"center-x/bottom-y: "
                f"{placement.x:.4f}, "
                f"{placement.bottom:.4f}"
            )
            print(
                f"generic score: "
                f"{placement.score:.4f}"
            )
            print(
                f"watch weight: "
                f"{selected_weight:.4f}"
            )

            print("TOP WEIGHTED:")

            for index, (candidate, weight) in enumerate(
                weighted[:5],
                start=1,
            ):
                print(
                    f"{index}. "
                    f"surface={candidate.surface_name} "
                    f"region={self._region(candidate.x)} "
                    f"x={candidate.x:.4f} "
                    f"weight={weight:.4f}"
                )

            target_x = placement.x
            target_y = (
                placement.bottom
                - placement.height * 0.35
            )

            self.controller.watch_spot = (
                target_x,
                target_y,
            )

            print(
                "character watch target: "
                f"x={target_x:.4f}, "
                f"y={target_y:.4f}"
            )

            self.controller.enter_watching(
                self.now()
            )

        except Exception as exc:
            print(f"WATCHING placement failed: {exc}")
            print("WATCHING aborted")

            if habitat is not None:
                habitat.disconnect()

            self.habitat = None
            self.chair_placement = None

    @staticmethod
    def _region(center_x: float) -> str:
        if center_x < 0.40:
            return "LEFT"

        if center_x > 0.60:
            return "RIGHT"

        return "CENTER"

    def exit_watching(self, _event=None) -> None:
        self.controller.exit_watching(
            self.now()
        )

    def tick(self) -> None:
        now = self.now()
        dt = max(0.0, now - self.last)
        self.last = now

        self.controller.update(dt, now)
        self.draw()

        self.root.after(16, self.tick)

    def draw(self) -> None:
        self.canvas.delete("actor")
        self.canvas.delete("chair")
        self.canvas.delete("status")

        state = self.controller.get_render_state()
        placement = self.chair_placement

        if (
            placement is not None
            and state["chair_visible"]
        ):
            left = placement.left * WIDTH
            top = placement.top * HEIGHT
            right = placement.right * WIDTH
            bottom = placement.bottom * HEIGHT

            self.canvas.create_rectangle(
                left,
                top,
                right,
                bottom,
                fill="#8b5a2b",
                outline="",
                tags="chair",
            )

            self.canvas.create_rectangle(
                left + 8,
                top,
                right - 8,
                top + 22,
                fill="#a56a35",
                outline="",
                tags="chair",
            )

            self.canvas.create_line(
                left + 14,
                top + 22,
                left + 7,
                bottom,
                fill="#70451f",
                width=5,
                tags="chair",
            )

            self.canvas.create_line(
                right - 14,
                top + 22,
                right - 7,
                bottom,
                fill="#70451f",
                width=5,
                tags="chair",
            )

        x = float(state["x"]) * WIDTH
        y = float(state["y"]) * HEIGHT

        color = (
            "#e6b84d"
            if state["state"] == "WATCHING_IDLE"
            else "#f0d080"
        )

        self.canvas.create_oval(
            x - 25,
            y - 18,
            x + 25,
            y + 18,
            fill=color,
            outline="#111",
            width=2,
            tags="actor",
        )

        self.canvas.create_oval(
            x - 12,
            y - 28,
            x + 12,
            y - 8,
            fill="#202020",
            outline="#111",
            tags="actor",
        )

        self.canvas.create_oval(
            x - 7,
            y - 24,
            x - 2,
            y - 19,
            fill="white",
            outline="",
            tags="actor",
        )

        self.canvas.create_oval(
            x + 2,
            y - 24,
            x + 7,
            y - 19,
            fill="white",
            outline="",
            tags="actor",
        )

        self.canvas.create_text(
            16,
            HEIGHT - 18,
            anchor="w",
            text=(
                "CHARACTER STATE: "
                f"{state['state']}   "
                f"pose={state['pose']}   "
                f"reaction={state['reaction']}"
            ),
            fill="white",
            font=("Consolas", 12),
            tags="status",
        )

    def close(self) -> None:
        if self.habitat is not None:
            self.habitat.disconnect()
            self.habitat = None

        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    root = tk.Tk()
    WatchingDemo(root, seed=args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()