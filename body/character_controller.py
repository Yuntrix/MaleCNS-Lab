"""Small, deterministic character-state controller for the WATCHING prototype."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class CharacterState(str, Enum):
    NORMAL = "NORMAL"
    WATCHING_ENTER = "WATCHING_ENTER"
    WATCHING_SIT = "WATCHING_SIT"
    WATCHING_IDLE = "WATCHING_IDLE"
    WATCHING_EXIT = "WATCHING_EXIT"


@dataclass
class CharacterConfig:
    watch_spot: Tuple[float, float] = (0.50, 0.72)
    chair_width: float = 0.12
    chair_height: float = 0.16
    travel_speed: float = 0.42
    sit_seconds: float = 0.35
    exit_seconds: float = 0.55


class CharacterController:
    """Scripted scene goal with a future reaction-overlay seam."""

    def __init__(self, config: CharacterConfig | None = None):
        self.config = config or CharacterConfig()
        self.current_state = CharacterState.NORMAL
        self.previous_state = CharacterState.NORMAL
        self.watch_spot = self.config.watch_spot
        self.state_enter_time = 0.0
        self.x, self.y = 0.22, 0.92
        self.vx, self.vy = 0.0, 0.0
        self.facing = 1
        self.reaction = "NONE"

    def _set_state(self, state: CharacterState, now: float) -> None:
        if self.current_state == state:
            return
        self.previous_state = self.current_state
        self.current_state = state
        self.state_enter_time = now
        print(f"CHARACTER STATE: {state.value}")

    def enter_watching(self, now: float = 0.0) -> bool:
        if self.current_state != CharacterState.NORMAL:
            return False
        self._set_state(CharacterState.WATCHING_ENTER, now)
        self.vx = self.vy = 0.0
        return True

    def exit_watching(self, now: float = 0.0) -> bool:
        if self.current_state not in (CharacterState.WATCHING_SIT, CharacterState.WATCHING_IDLE):
            return False
        self._set_state(CharacterState.WATCHING_EXIT, now)
        return True

    def update(self, dt: float, now: float) -> None:
        if self.current_state == CharacterState.WATCHING_ENTER:
            tx, ty = self.watch_spot
            dx, dy = tx - self.x, ty - self.y
            distance = (dx * dx + dy * dy) ** 0.5
            if distance <= 0.012:
                self.x, self.y = tx, ty
                self._set_state(CharacterState.WATCHING_SIT, now)
            elif distance > 0:
                step = min(self.config.travel_speed * dt, distance)
                self.x += dx / distance * step
                self.y += dy / distance * step
                self.facing = 1 if dx >= 0 else -1
        elif self.current_state == CharacterState.WATCHING_SIT:
            self.x, self.y = self.watch_spot
            self.facing = 1
            if now - self.state_enter_time >= self.config.sit_seconds:
                self._set_state(CharacterState.WATCHING_IDLE, now)
        elif self.current_state == CharacterState.WATCHING_IDLE:
            self.x, self.y = self.watch_spot
            self.facing = 1
        elif self.current_state == CharacterState.WATCHING_EXIT:
            self.x, self.y = self.watch_spot
            self.facing = 1
            if now - self.state_enter_time >= self.config.exit_seconds:
                self._set_state(CharacterState.NORMAL, now)

    def get_render_state(self) -> Dict[str, object]:
        chair_visible = self.current_state in (CharacterState.WATCHING_SIT, CharacterState.WATCHING_IDLE, CharacterState.WATCHING_EXIT)
        pose = {
            CharacterState.NORMAL: "NORMAL_POSE",
            CharacterState.WATCHING_ENTER: "WATCH_FLY_TO_SPOT",
            CharacterState.WATCHING_SIT: "WATCH_SIT",
            CharacterState.WATCHING_IDLE: "WATCH_IDLE",
            CharacterState.WATCHING_EXIT: "WATCH_STAND",
        }[self.current_state]
        return {
            "state": self.current_state.value, "previous_state": self.previous_state.value,
            "x": self.x, "y": self.y, "facing": self.facing, "pose": pose,
            "chair_visible": chair_visible, "chair_x": self.watch_spot[0],
            "chair_y": self.watch_spot[1] + 0.10, "reaction": self.reaction,
        }
