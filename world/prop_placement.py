from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Tuple

try:
    from world.obs_habitat import HabitatCollider, HorizontalSupportSurface, ObsHabitat
except ModuleNotFoundError:
    from obs_habitat import HabitatCollider, HorizontalSupportSurface, ObsHabitat

EPSILON = 1e-6

class VerticalPreference(str, Enum):
    ANY = "ANY"
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"

@dataclass(frozen=True)
class PropFootprint:
    name: str
    width: float
    height: float
    safety_margin: float = 0.0
    preferred_surface_types: Tuple[str, ...] = ()
    vertical_preference: VerticalPreference = VerticalPreference.ANY
    avoid_screen_center: bool = False
    center_avoidance_strength: float = 2.0

@dataclass(frozen=True)
class PropPlacement:
    prop_name: str
    surface_name: str
    surface_type: str
    x: float
    y: float
    left: float
    top: float
    right: float
    bottom: float
    width: float
    height: float
    score: float
    pixel_left: float
    pixel_top: float
    pixel_right: float
    pixel_bottom: float

class PropPlacementEngine:
    def __init__(self, habitat: ObsHabitat) -> None:
        self.habitat = habitat

    def find_placement(self, prop: PropFootprint) -> Optional[PropPlacement]:
        candidates = self.find_candidates(prop)
        return candidates[0] if candidates else None

    def find_candidates(self, prop: PropFootprint) -> List[PropPlacement]:
        result: List[PropPlacement] = []
        # Use both physical colliders and visible non-support overlays.
        # TRANSPARENT_PENDING items (Mario, goals, combo widgets, etc.) are
        # not floors and never block the fly, but a prop must not be placed
        # on top of their visible pixels.
        colliders = self.habitat.get_colliders()
        preferred = {value.upper() for value in prop.preferred_surface_types}
        for surface in self.habitat.get_support_surfaces():
            if not surface.active or not surface.visible:
                continue
            if preferred and "ANY" not in preferred and surface.surface_type.upper() not in preferred:
                continue
            usable = surface.width - prop.safety_margin * 2
            if usable + EPSILON < prop.width:
                continue
            left = surface.x_start + prop.safety_margin + prop.width / 2
            right = surface.x_end - prop.safety_margin - prop.width / 2
            if right < left:
                continue
            for x in self._samples(left, right):
                placement = self._make(prop, surface, x)
                if placement is None or not self._inside(placement):
                    continue
                if self._blocked(placement, colliders, surface, prop.safety_margin):
                    continue
                score = self._score(prop, surface, x)
                result.append(PropPlacement(
                    placement.prop_name, placement.surface_name, placement.surface_type,
                    placement.x, placement.y, placement.left, placement.top,
                    placement.right, placement.bottom, placement.width,
                    placement.height, score, placement.pixel_left,
                    placement.pixel_top, placement.pixel_right, placement.pixel_bottom,
                ))
        result.sort(key=lambda item: (-item.score, item.surface_name, item.x, item.y))
        unique: List[PropPlacement] = []
        for item in result:
            if not any(item.surface_name == old.surface_name and abs(item.x - old.x) <= EPSILON and abs(item.y - old.y) <= EPSILON for old in unique):
                unique.append(item)
        return unique

    @staticmethod
    def _samples(left: float, right: float) -> List[float]:
        if abs(right - left) <= EPSILON:
            return [round(left, 8)]
        return [round(value, 8) for value in (left, left + (right-left)*0.25, (left+right)/2, left+(right-left)*0.75, right)]

    def _make(self, prop: PropFootprint, surface: HorizontalSupportSurface, x: float) -> Optional[PropPlacement]:
        left = x - prop.width / 2
        right = x + prop.width / 2
        bottom = surface.y
        top = bottom - prop.height
        if top < -EPSILON:
            return None
        return PropPlacement(prop.name, surface.name, surface.surface_type, x, bottom, left, top, right, bottom, prop.width, prop.height, 0.0, left*self.habitat.canvas_width, top*self.habitat.canvas_height, right*self.habitat.canvas_width, bottom*self.habitat.canvas_height)

    @staticmethod
    def _inside(p: PropPlacement) -> bool:
        return p.left >= -EPSILON and p.right <= 1+EPSILON and p.top >= -EPSILON and p.bottom <= 1+EPSILON

    @staticmethod
    def _supporting(p: PropPlacement, c: HabitatCollider, surface: HorizontalSupportSurface) -> bool:
        return c.scene_item_id == surface.source_scene_item_id and abs(p.bottom-c.y) <= 1e-5

    def _blocked(self, p: PropPlacement, colliders: Iterable[HabitatCollider], surface: HorizontalSupportSurface, margin: float) -> bool:
        left, top, right, bottom = p.left-margin, p.top-margin, p.right+margin, p.bottom+margin
        for c in colliders:
            if not c.visible or self._supporting(p, c, surface):
                continue
            is_physical = c.blocking
            is_visual_exclusion = (
                str(c.collision_mode).upper() == "TRANSPARENT_PENDING"
            )
            if not (is_physical or is_visual_exclusion):
                continue
            if left < c.x+c.width and right > c.x and top < c.y+c.height and bottom > c.y:
                return True
        return False

    @staticmethod
    def _score(prop: PropFootprint, surface: HorizontalSupportSurface, x: float) -> float:
        score = surface.y * 2.0
        if surface.surface_type.upper() == "SCREEN_FLOOR":
            score += 1.5
        if prop.preferred_surface_types and surface.surface_type.upper() in {v.upper() for v in prop.preferred_surface_types}:
            score += 1.0
        score -= abs(x - (surface.x_start + surface.x_end)/2) * 0.2
        if prop.avoid_screen_center:
            score -= max(0.0, prop.center_avoidance_strength) * max(0.0, 1.0 - abs(x-0.5)/0.5)
        if prop.vertical_preference == VerticalPreference.HIGH:
            score += 1.0 - surface.y
        elif prop.vertical_preference == VerticalPreference.MID:
            score += 1.0 - abs(surface.y - 0.5) * 2.0
        return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Generic deterministic prop placement engine")
    parser.add_argument("--host", default=os.getenv("OBS_WEBSOCKET_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OBS_WEBSOCKET_PORT", "7654")))
    parser.add_argument("--password", default=os.getenv("OBS_WEBSOCKET_PASSWORD"))
    args = parser.parse_args()
    habitat = ObsHabitat(args.host, args.port, args.password)
    try:
        habitat.connect()
        engine = PropPlacementEngine(habitat)
        print("GENERIC PROP PLACEMENT")
        print(f"Habitat Scene: {habitat.habitat_scene_name}")
        print(f"Support surfaces: {len(habitat.get_support_surfaces())}")
        for name, width, height in (("SMALL", .10, .12), ("MEDIUM", .20, .16), ("LARGE", .35, .20)):
            request = PropFootprint(name, width, height, .01, ("SCREEN_FLOOR", "OVERLAY_TOP"), VerticalPreference.LOW, True)
            candidates = engine.find_candidates(request)
            print(f"[{name}] valid candidates: {len(candidates)}")
            for index, candidate in enumerate(candidates[:5], 1):
                print(f"  {index}. {candidate.surface_name} x={candidate.x:.4f} y={candidate.y:.4f} score={candidate.score:.4f}")
    except Exception as exc:
        print(f"OBS connection failed: {exc}")
        return 2
    finally:
        habitat.disconnect()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
