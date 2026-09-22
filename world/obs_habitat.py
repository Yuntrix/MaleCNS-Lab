from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7654
DEFAULT_TIMEOUT = 5.0
DEFAULT_INTERVAL = 1.0
DEFAULT_FULLSCREEN_RATIO = 0.85
GEOMETRY_EPSILON = 1.0

# Source policy is intentionally independent from collision classification.
# A source can block movement without becoming a landing/activity surface.
SOURCE_IGNORE_NAMES = {
    "chatback",
    "chatback2",
    "kombo",
}
BLOCKING_NON_SUPPORT_NAMES = {
    "eventlist",
    "mario",
}

# Insets are measured in final OBS canvas pixels: left, top, right, bottom.
# They trim known transparent/padded regions without changing OBS transforms.
# EVENTLIST uses the visible bounds measured from its browser-source output.
COLLISION_INSETS_BY_SOURCE = {
    "eventlist": (104.0, 414.0, 11.0, 11.0),
}


class HabitatClassification(str, Enum):
    BACKGROUND = "BACKGROUND"
    SOLID_OVERLAY = "SOLID_OVERLAY"
    TRANSPARENT_OVERLAY = "TRANSPARENT_OVERLAY"
    IGNORE = "IGNORE"


class CollisionMode(str, Enum):
    NONE = "NONE"
    RECT = "RECT"
    TRANSPARENT_PENDING = "TRANSPARENT_PENDING"


@dataclass(frozen=True)
class HabitatItem:
    scene_item_id: int
    source_name: str
    source_kind: str
    source_type_id: str
    group_path: str
    visible: bool
    classification: str
    classification_reason: str
    collision_mode: str
    pixel_x: float
    pixel_y: float
    pixel_width: float
    pixel_height: float
    rotation: float
    raw_item: Dict[str, Any]

    @property
    def normalized_rect(self) -> Tuple[float, float, float, float]:
        return (
            self.pixel_x,
            self.pixel_y,
            self.pixel_width,
            self.pixel_height,
        )


@dataclass(frozen=True)
class HabitatCollider:
    name: str
    scene_item_id: int
    source_kind: str
    group_path: str
    x: float
    y: float
    width: float
    height: float
    pixel_x: float
    pixel_y: float
    pixel_width: float
    pixel_height: float
    classification: str
    collision_mode: str
    classification_reason: str
    blocking: bool
    visible: bool


@dataclass(frozen=True)
class HorizontalSupportSurface:
    name: str
    source_name: str
    source_scene_item_id: int
    surface_type: str
    x_start: float
    x_end: float
    y: float
    width: float
    pixel_x_start: float
    pixel_x_end: float
    pixel_y: float
    pixel_width: float
    active: bool
    visible: bool


class ObsProtocolError(RuntimeError):
    pass


class ObsWebSocket:
    def __init__(
        self,
        host: str,
        port: int,
        password: Optional[str],
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.timeout = timeout
        self.socket: Any = None

    def connect(self) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "websockets paketi eksik. Kurulum: "
                "python -m pip install websockets"
            ) from exc

        try:
            self.socket = connect(
                f"ws://{self.host}:{self.port}",
                open_timeout=self.timeout,
                close_timeout=self.timeout,
            )

            hello = self._receive()
            if hello.get("op") != 0:
                raise ObsProtocolError("OBS Hello mesajı alınamadı")

            hello_data = hello.get("d", {})
            identify: Dict[str, Any] = {
                "rpcVersion": int(hello_data.get("rpcVersion", 1))
            }

            authentication = hello_data.get("authentication")
            if authentication:
                if self.password is None:
                    raise ObsProtocolError(
                        "OBS password gerekli. "
                        "OBS_WEBSOCKET_PASSWORD ayarla."
                    )

                salt = authentication.get("salt", "")
                challenge = authentication.get("challenge", "")

                secret = base64.b64encode(
                    hashlib.sha256(
                        (self.password + salt).encode("utf-8")
                    ).digest()
                ).decode("utf-8")

                identify["authentication"] = base64.b64encode(
                    hashlib.sha256(
                        (secret + challenge).encode("utf-8")
                    ).digest()
                ).decode("utf-8")

            self._send({"op": 1, "d": identify})

            identified = self._receive()
            if identified.get("op") != 2:
                raise ObsProtocolError(
                    f"OBS Identify başarısız: {identified.get('d', {})}"
                )

        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def request(
        self,
        request_type: str,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.socket is None:
            raise ObsProtocolError("OBS bağlantısı açık değil")

        request_id = str(uuid.uuid4())

        self._send(
            {
                "op": 6,
                "d": {
                    "requestType": request_type,
                    "requestId": request_id,
                    "requestData": request_data or {},
                },
            }
        )

        deadline = time.monotonic() + self.timeout

        while time.monotonic() < deadline:
            message = self._receive()

            if message.get("op") != 7:
                continue

            data = message.get("d", {})

            if data.get("requestId") != request_id:
                continue

            status = data.get("requestStatus", {})

            if not status.get("result", False):
                raise ObsProtocolError(
                    f"{request_type} başarısız: "
                    f"{status.get('code', 'unknown')} "
                    f"{status.get('comment', '')}"
                )

            return data.get("responseData", {})

        raise TimeoutError(f"OBS request timeout: {request_type}")

    def _send(self, message: Dict[str, Any]) -> None:
        self.socket.send(json.dumps(message))

    def _receive(self) -> Dict[str, Any]:
        raw = self.socket.recv()

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        return json.loads(raw)


def normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()

    for character in (" ", "_", "-"):
        text = text.replace(character, "")

    return text


def source_allows_support(source_name: str) -> bool:
    """Return whether a blocking source may expose its top as a perch."""
    return normalize_name(source_name) not in BLOCKING_NON_SUPPORT_NAMES


def classify_source(
    source_name: str,
    source_kind: str,
    source_type_id: str,
    group_path: str,
    pixel_width: float,
    pixel_height: float,
    canvas_width: int,
    canvas_height: int,
) -> Tuple[str, str, str]:
    name = normalize_name(source_name)
    kind = str(source_kind or "").lower()
    type_id = str(source_type_id or "").lower()
    groups = normalize_name(group_path)

    if name in SOURCE_IGNORE_NAMES:
        return (
            HabitatClassification.IGNORE.value,
            "explicit habitat ignore source",
            CollisionMode.NONE.value,
        )

    if name in BLOCKING_NON_SUPPORT_NAMES:
        return (
            HabitatClassification.SOLID_OVERLAY.value,
            "blocking source; support disabled by policy",
            CollisionMode.RECT.value,
        )

    # These sources remain visible in OBS but are not physical habitat
    # surfaces: their content is sparse/irregular and must not support a chair.
    # Evaluate this by source name before the group-level meme rule so that
    # putting them inside a group never turns them into IGNORE items.
    if name in {
        "kickgoal",
        "kicksgoal",
    }:
        return (
            HabitatClassification.TRANSPARENT_OVERLAY.value,
            "explicit non-support visual source",
            CollisionMode.TRANSPARENT_PENDING.value,
        )

    if "meme" in groups:
        return (
            HabitatClassification.IGNORE.value,
            "ignored meme group",
            CollisionMode.NONE.value,
        )

    explicit_ignore = {
        "43",
        "browser2",
        "blerp2",
        "soonalerts",
        "host",
        "donate",
    }

    if name in explicit_ignore:
        reasons = {
            "43": "explicit ignore source",
            "browser2": "explicit moving character source",
            "blerp2": "explicit stream widget source",
            "soonalerts": "explicit alert source",
            "host": "explicit stream widget source",
            "donate": "explicit stream widget source",
        }

        return (
            HabitatClassification.IGNORE.value,
            reasons.get(name, "explicit ignore source"),
            CollisionMode.NONE.value,
        )

    if pixel_width <= GEOMETRY_EPSILON or pixel_height <= GEOMETRY_EPSILON:
        return (
            HabitatClassification.IGNORE.value,
            "zero-size geometry",
            CollisionMode.NONE.value,
        )

    area_ratio = 0.0

    if canvas_width > 0 and canvas_height > 0:
        area_ratio = (
            pixel_width * pixel_height
        ) / (canvas_width * canvas_height)

    if any(token in name for token in (
        "webcam",
        "camera",
        "facecam",
        "cam",
        "chat",
    )):
        return (
            HabitatClassification.SOLID_OVERLAY.value,
            "camera/chat name hint",
            CollisionMode.RECT.value,
        )

    if any(token in name for token in (
        "gamecapture",
        "gameplay",
        "displaycapture",
        "fullscreen",
        "background",
        "mainvideo",
    )):
        return (
            HabitatClassification.BACKGROUND.value,
            "background name hint",
            CollisionMode.NONE.value,
        )

    if any(token in name for token in (
        "alert",
        "alerts",
        "widget",
        "overlay",
        "notification",
        "goal",
        "progress",
        "label",
        "text",
    )):
        return (
            HabitatClassification.TRANSPARENT_OVERLAY.value,
            "foreground widget/text hint",
            CollisionMode.TRANSPARENT_PENDING.value,
        )

    if any(token in type_id for token in (
        "game_capture",
        "display_capture",
        "monitor_capture",
        "window_capture",
    )):
        return (
            HabitatClassification.BACKGROUND.value,
            "capture source type",
            CollisionMode.NONE.value,
        )

    if any(token in type_id for token in (
        "dshow",
        "camera",
        "av_capture",
    )):
        return (
            HabitatClassification.SOLID_OVERLAY.value,
            "camera source type",
            CollisionMode.RECT.value,
        )

    if "monitor" in kind or "display" in kind or "game" in kind:
        return (
            HabitatClassification.BACKGROUND.value,
            "display/game source kind",
            CollisionMode.NONE.value,
        )

    if "browser_source" in type_id:
        if area_ratio >= DEFAULT_FULLSCREEN_RATIO:
            return (
                HabitatClassification.TRANSPARENT_OVERLAY.value,
                "large browser source; alpha unknown",
                CollisionMode.TRANSPARENT_PENDING.value,
            )

        return (
            HabitatClassification.SOLID_OVERLAY.value,
            "foreground browser source",
            CollisionMode.RECT.value,
        )

    if area_ratio >= DEFAULT_FULLSCREEN_RATIO:
        return (
            HabitatClassification.BACKGROUND.value,
            "large canvas coverage",
            CollisionMode.NONE.value,
        )

    return (
        HabitatClassification.SOLID_OVERLAY.value,
        "small foreground fallback",
        CollisionMode.RECT.value,
    )


class ObsHabitat:
    # Habitat instances are recreated by the overlay poller. Cache only the
    # rendered alpha bounds; transforms are always taken from the current poll.
    _kick_chat_bounds_cache = {}

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        password: Optional[str] = None,
        scene_name: Optional[str] = None,
    ) -> None:
        self.client = ObsWebSocket(host, port, password)
        requested_scene = scene_name
        if requested_scene is None:
            requested_scene = os.getenv("OBS_HABITAT_SCENE", "").strip()
        self.requested_scene_name = requested_scene or None

        self.canvas_width = 0
        self.canvas_height = 0
        self.current_scene_name = ""
        self.habitat_scene_name = ""

        self.items: List[HabitatItem] = []
        self.colliders: List[HabitatCollider] = []
        self.support_surfaces: List[HorizontalSupportSurface] = []

    def connect(self) -> None:
        self.client.connect()
        self.refresh()

    def disconnect(self) -> None:
        self.client.disconnect()

    def refresh(self) -> bool:
        old_signature = self._signature()

        video = self.client.request("GetVideoSettings")

        self.canvas_width = int(
            video.get("baseWidth")
            or video.get("outputWidth")
            or 0
        )

        self.canvas_height = int(
            video.get("baseHeight")
            or video.get("outputHeight")
            or 0
        )

        program = self.client.request("GetCurrentProgramScene")
        self.current_scene_name = str(
            program.get("sceneName", "UNKNOWN")
        )

        scene_list = self.client.request("GetSceneList")

        available = {
            str(scene.get("sceneName", ""))
            for scene in scene_list.get("scenes", [])
        }

        if self.requested_scene_name:
            if self.requested_scene_name not in available:
                raise ObsProtocolError(
                    f"OBS scene bulunamadı: {self.requested_scene_name}"
                )
            self.habitat_scene_name = self.requested_scene_name
        elif self.current_scene_name in available:
            # The active program scene is the source of truth by default.
            # Set OBS_HABITAT_SCENE=WORLD when a fixed scene is required.
            self.habitat_scene_name = self.current_scene_name
        elif "WORLD" in available:
            self.habitat_scene_name = "WORLD"
        elif "FINAL" in available:
            self.habitat_scene_name = "FINAL"
        elif available:
            self.habitat_scene_name = sorted(available)[0]
        else:
            raise ObsProtocolError(
                "OBS scene listesi boş"
            )

        self.items = self._read_items(self.habitat_scene_name)
        self.colliders = self._build_colliders(self.items)
        self.support_surfaces = self._build_support_surfaces(
            self.colliders
        )

        return old_signature != self._signature()

    @staticmethod
    def _matrix_multiply(parent, child):
        pa, pb, pc, pd, pe, pf = parent
        ca, cb, cc, cd, ce, cf = child

        return (
            pa * ca + pc * cb,
            pb * ca + pd * cb,
            pa * cc + pc * cd,
            pb * cc + pd * cd,
            pa * ce + pc * cf + pe,
            pb * ce + pd * cf + pf,
        )

    @staticmethod
    def _matrix_apply(matrix, point):
        a, b, c, d, e, f = matrix
        x, y = point
        return a * x + c * y + e, b * x + d * y + f

    @staticmethod
    def _as_bool(value, default=True):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"false", "0", "no", "off", "hidden"}:
                return False
            if text in {"true", "1", "yes", "on", "visible"}:
                return True
        return bool(value)

    @staticmethod
    def _finite_number(value, default=0.0):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(number):
            return float(default)
        return number

    @staticmethod
    def _alignment_anchor(width, height, alignment):
        """Return the source point represented by OBS position.x/y."""
        anchor_x = width / 2.0
        anchor_y = height / 2.0

        if isinstance(alignment, str):
            text = alignment.lower()
            if "left" in text:
                anchor_x = 0.0
            elif "right" in text:
                anchor_x = width
            if "top" in text:
                anchor_y = 0.0
            elif "bottom" in text:
                anchor_y = height
            return anchor_x, anchor_y

        flags = int(ObsHabitat._finite_number(alignment, 0.0))
        # OBS alignment flags: LEFT=1, RIGHT=2, TOP=4, BOTTOM=8.
        if flags & 1:
            anchor_x = 0.0
        elif flags & 2:
            anchor_x = width
        if flags & 4:
            anchor_y = 0.0
        elif flags & 8:
            anchor_y = height
        return anchor_x, anchor_y

    @classmethod
    def _item_matrix(cls, transform):
        position = transform.get("position", {}) or {}
        scale_data = transform.get("scale", {}) or {}

        x = cls._finite_number(
            position.get("x", transform.get("positionX", 0.0)),
            0.0,
        )
        y = cls._finite_number(
            position.get("y", transform.get("positionY", 0.0)),
            0.0,
        )

        source_width = cls._finite_number(
            transform.get("sourceWidth", 0.0),
            0.0,
        )
        source_height = cls._finite_number(
            transform.get("sourceHeight", 0.0),
            0.0,
        )

        reported_width = cls._finite_number(
            transform.get("width", 0.0),
            0.0,
        )
        reported_height = cls._finite_number(
            transform.get("height", 0.0),
            0.0,
        )

        if source_width <= GEOMETRY_EPSILON:
            source_width = max(
                reported_width,
                cls._finite_number(transform.get("boundsWidth", 0.0), 0.0),
            )
        if source_height <= GEOMETRY_EPSILON:
            source_height = max(
                reported_height,
                cls._finite_number(transform.get("boundsHeight", 0.0), 0.0),
            )

        raw_scale_x = cls._finite_number(
            transform.get("scaleX", scale_data.get("x", 1.0)),
            1.0,
        )
        raw_scale_y = cls._finite_number(
            transform.get("scaleY", scale_data.get("y", 1.0)),
            1.0,
        )
        if abs(raw_scale_x) <= 1e-9:
            raw_scale_x = 1.0
        if abs(raw_scale_y) <= 1e-9:
            raw_scale_y = 1.0

        # OBS exposes width/height after bounds and scale.  Prefer those
        # values when present, while preserving negative scale for flips.
        if reported_width > GEOMETRY_EPSILON and source_width > GEOMETRY_EPSILON:
            raw_scale_x = math.copysign(
                reported_width / source_width,
                raw_scale_x,
            )
        if reported_height > GEOMETRY_EPSILON and source_height > GEOMETRY_EPSILON:
            raw_scale_y = math.copysign(
                reported_height / source_height,
                raw_scale_y,
            )

        crop_left = max(
            0.0,
            cls._finite_number(transform.get("cropLeft", 0.0), 0.0),
        )
        crop_top = max(
            0.0,
            cls._finite_number(transform.get("cropTop", 0.0), 0.0),
        )
        crop_right = max(
            0.0,
            cls._finite_number(transform.get("cropRight", 0.0), 0.0),
        )
        crop_bottom = max(
            0.0,
            cls._finite_number(transform.get("cropBottom", 0.0), 0.0),
        )
        crop_left = min(crop_left, source_width)
        crop_top = min(crop_top, source_height)
        crop_right = min(crop_right, max(0.0, source_width - crop_left))
        crop_bottom = min(crop_bottom, max(0.0, source_height - crop_top))
        visible_width = max(0.0, source_width - crop_left - crop_right)
        visible_height = max(0.0, source_height - crop_top - crop_bottom)

        rotation = cls._finite_number(transform.get("rotation", 0.0), 0.0)

        radians = math.radians(rotation)
        cos_value = math.cos(radians)
        sin_value = math.sin(radians)

        local = (
            cos_value * raw_scale_x,
            sin_value * raw_scale_x,
            -sin_value * raw_scale_y,
            cos_value * raw_scale_y,
            0.0,
            0.0,
        )

        alignment = transform.get("alignment", 0)
        anchor_x, anchor_y = cls._alignment_anchor(
            source_width,
            source_height,
            alignment,
        )
        transformed_anchor = cls._matrix_apply(
            local,
            (anchor_x, anchor_y),
        )

        matrix = (
            local[0],
            local[1],
            local[2],
            local[3],
            x - transformed_anchor[0],
            y - transformed_anchor[1],
        )

        return (
            matrix,
            visible_width,
            visible_height,
            rotation,
            crop_left,
            crop_top,
        )

    @classmethod
    def _rect_from_matrix(
        cls,
        matrix,
        width,
        height,
        origin=(0.0, 0.0),
    ):
        origin_x, origin_y = origin
        points = (
            cls._matrix_apply(matrix, (origin_x, origin_y)),
            cls._matrix_apply(matrix, (origin_x + width, origin_y)),
            cls._matrix_apply(matrix, (origin_x, origin_y + height)),
            cls._matrix_apply(
                matrix,
                (origin_x + width, origin_y + height),
            ),
        )

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        return (
            min(xs),
            min(ys),
            max(xs) - min(xs),
            max(ys) - min(ys),
        )

    def _kick_chat_rect(self, source_name, transform, parent_matrix):
        """Use the composited chat frame, not its uncropped browser child."""
        from PIL import Image

        key = (self.client.host, self.client.port, source_name)
        now = time.monotonic()
        cached = self._kick_chat_bounds_cache.get(key)
        if cached is None or now - cached[0] >= 1.0 or cached[1] != transform:
            try:
                response = self.client.request("GetSourceScreenshot", {
                    "sourceName": source_name, "imageFormat": "png",
                })
                data = base64.b64decode(response["imageData"].split(",", 1)[1])
                with Image.open(io.BytesIO(data)) as image:
                    alpha = image.convert("RGBA").getchannel("A")
                    left = int(transform.get("cropLeft", 0))
                    top = int(transform.get("cropTop", 0))
                    right = alpha.width - int(transform.get("cropRight", 0))
                    bottom = alpha.height - int(transform.get("cropBottom", 0))
                    bounds = None
                    if right > left and bottom > top:
                        bounds = alpha.crop((left, top, right, bottom)).getbbox()
                cached = (now, dict(transform), bounds)
                self._kick_chat_bounds_cache[key] = cached
            except Exception:
                # Never substitute transparent browser bounds on capture failure.
                if cached is None or cached[1] != transform:
                    return (0.0, 0.0, 0.0, 0.0)

        bounds = cached[2]
        if bounds is None:
            return (0.0, 0.0, 0.0, 0.0)

        # OBS positions/alignment refer to the cropped image. The screenshot
        # already includes the children's crop, alignment and group composition.
        matrix, width, height, _, _, _ = self._item_matrix(transform)
        old_anchor = self._alignment_anchor(
            float(transform["sourceWidth"]), float(transform["sourceHeight"]),
            transform.get("alignment", 0),
        )
        new_anchor = self._alignment_anchor(width, height, transform.get("alignment", 0))
        dx, dy = old_anchor[0] - new_anchor[0], old_anchor[1] - new_anchor[1]
        a, b, c, d, x, y = matrix
        matrix = (a, b, c, d, x + a * dx + c * dy, y + b * dx + d * dy)
        return self._rect_from_matrix(
            self._matrix_multiply(parent_matrix, matrix),
            bounds[2] - bounds[0], bounds[3] - bounds[1], bounds[:2],
        )

    def _read_items(
        self,
        scene_name: str,
        group_path: str = "",
        parent_matrix=None,
        parent_visible: bool = True,
        visited: Optional[Set[str]] = None,
    ) -> List[HabitatItem]:
        if parent_matrix is None:
            parent_matrix = (
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            )

        if visited is None:
            visited = set()

        if scene_name in visited:
            return []

        next_visited = set(visited)
        next_visited.add(scene_name)

        request_type = (
            "GetGroupSceneItemList"
            if group_path
            else "GetSceneItemList"
        )

        response = self.client.request(
            request_type,
            {"sceneName": scene_name},
        )

        result: List[HabitatItem] = []

        for item in response.get("sceneItems", []):
            source_name = str(item.get("sourceName", "Unnamed"))
            source_kind = str(item.get("sourceType", "unknown"))
            source_type_id = str(item.get("sourceTypeId", ""))

            transform = item.get("sceneItemTransform", {}) or {}
            own_visible = self._as_bool(
                item.get(
                    "sceneItemEnabled",
                    transform.get("visible", True),
                )
            )
            effective_visible = parent_visible and own_visible

            (
                local_matrix,
                width,
                height,
                rotation,
                crop_left,
                crop_top,
            ) = (
                self._item_matrix(transform)
            )

            global_matrix = self._matrix_multiply(
                parent_matrix,
                local_matrix,
            )

            rect = self._rect_from_matrix(
                global_matrix,
                width,
                height,
                (crop_left, crop_top),
            )

            is_group = (
                "group" in source_kind.lower()
                or "group" in source_type_id.lower()
                or str(item.get("isGroup", "")).lower() == "true"
            )

            if is_group:
                child_path = (
                    source_name
                    if not group_path
                    else f"{group_path} > {source_name}"
                )

                children = self._read_items(
                    scene_name=source_name,
                    group_path=child_path,
                    parent_matrix=global_matrix,
                    parent_visible=effective_visible,
                    visited=next_visited,
                )
                if normalize_name(source_name) == "kickchat" and effective_visible:
                    chat_rect = self._kick_chat_rect(source_name, transform, parent_matrix)
                    children = [
                        replace(child, pixel_x=chat_rect[0], pixel_y=chat_rect[1],
                                pixel_width=chat_rect[2], pixel_height=chat_rect[3])
                        if normalize_name(child.source_name) == "chatbotrix" else child
                        for child in children
                    ]
                result.extend(children)

                continue

            classification, reason, collision_mode = classify_source(
                source_name=source_name,
                source_kind=source_kind,
                source_type_id=source_type_id,
                group_path=group_path,
                pixel_width=rect[2],
                pixel_height=rect[3],
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
            )

            result.append(
                HabitatItem(
                    scene_item_id=int(item.get("sceneItemId", -1)),
                    source_name=source_name,
                    source_kind=source_kind,
                    source_type_id=source_type_id,
                    group_path=group_path,
                    visible=effective_visible,
                    classification=classification,
                    classification_reason=reason,
                    collision_mode=collision_mode,
                    pixel_x=rect[0],
                    pixel_y=rect[1],
                    pixel_width=rect[2],
                    pixel_height=rect[3],
                    rotation=rotation,
                    raw_item=item,
                )
            )

        return result

    def _build_colliders(
        self,
        items: Iterable[HabitatItem],
    ) -> List[HabitatCollider]:
        result: List[HabitatCollider] = []

        for item in items:
            if not item.visible:
                continue

            if item.classification in (
                HabitatClassification.BACKGROUND.value,
                HabitatClassification.IGNORE.value,
            ):
                continue

            if item.collision_mode == CollisionMode.NONE.value:
                continue

            if self.canvas_width <= 0 or self.canvas_height <= 0:
                continue

            # Collision geometry must describe pixels that are actually on
            # the OBS canvas.  A transformed source may extend outside the
            # canvas; keeping the off-canvas part creates apparent ghost
            # collisions when a source is moved or cropped.
            raw_left = float(item.pixel_x)
            raw_top = float(item.pixel_y)
            raw_right = raw_left + float(item.pixel_width)
            raw_bottom = raw_top + float(item.pixel_height)
            pixel_x = max(0.0, raw_left)
            pixel_y = max(0.0, raw_top)
            pixel_right = min(float(self.canvas_width), raw_right)
            pixel_bottom = min(float(self.canvas_height), raw_bottom)
            pixel_width = pixel_right - pixel_x
            pixel_height = pixel_bottom - pixel_y

            inset_left, inset_top, inset_right, inset_bottom = (
                COLLISION_INSETS_BY_SOURCE.get(
                    normalize_name(item.source_name),
                    (0.0, 0.0, 0.0, 0.0),
                )
            )
            pixel_x += max(0.0, float(inset_left))
            pixel_y += max(0.0, float(inset_top))
            pixel_right -= max(0.0, float(inset_right))
            pixel_bottom -= max(0.0, float(inset_bottom))
            pixel_width = pixel_right - pixel_x
            pixel_height = pixel_bottom - pixel_y

            if (
                pixel_width <= GEOMETRY_EPSILON
                or pixel_height <= GEOMETRY_EPSILON
            ):
                continue

            result.append(
                HabitatCollider(
                    name=item.source_name,
                    scene_item_id=item.scene_item_id,
                    source_kind=item.source_kind,
                    group_path=item.group_path,
                    x=pixel_x / self.canvas_width,
                    y=pixel_y / self.canvas_height,
                    width=pixel_width / self.canvas_width,
                    height=pixel_height / self.canvas_height,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                    pixel_width=pixel_width,
                    pixel_height=pixel_height,
                    classification=item.classification,
                    collision_mode=item.collision_mode,
                    classification_reason=item.classification_reason,
                    blocking=(
                        item.classification
                        == HabitatClassification.SOLID_OVERLAY.value
                        and item.collision_mode
                        == CollisionMode.RECT.value
                    ),
                    visible=item.visible,
                )
            )

        return result

    def _build_support_surfaces(
        self,
        colliders: Iterable[HabitatCollider],
    ) -> List[HorizontalSupportSurface]:
        surfaces = [
            HorizontalSupportSurface(
                name="SCREEN_FLOOR",
                source_name="SCREEN_FLOOR",
                source_scene_item_id=-1,
                surface_type="SCREEN_FLOOR",
                x_start=0.0,
                x_end=1.0,
                y=1.0,
                width=1.0,
                pixel_x_start=0.0,
                pixel_x_end=float(self.canvas_width),
                pixel_y=float(self.canvas_height),
                pixel_width=float(self.canvas_width),
                active=True,
                visible=True,
            )
        ]

        for collider in colliders:
            if not collider.visible:
                continue

            if not collider.blocking:
                continue

            if collider.classification != (
                HabitatClassification.SOLID_OVERLAY.value
            ):
                continue

            if not source_allows_support(collider.name):
                continue

            original_start = collider.x
            original_end = collider.x + collider.width
            x_start = max(original_start, 0.0)
            x_end = min(original_end, 1.0)
            y = collider.y

            if y < 0.0 or y > 1.0:
                continue

            clipped_width = x_end - x_start
            if clipped_width <= 1e-6:
                continue

            surfaces.append(
                HorizontalSupportSurface(
                    name=f"OVERLAY_TOP:{collider.name}",
                    source_name=collider.name,
                    source_scene_item_id=collider.scene_item_id,
                    surface_type="OVERLAY_TOP",
                    x_start=x_start,
                    x_end=x_end,
                    y=y,
                    width=clipped_width,
                    pixel_x_start=x_start * self.canvas_width,
                    pixel_x_end=x_end * self.canvas_width,
                    pixel_y=collider.pixel_y,
                    pixel_width=clipped_width * self.canvas_width,
                    active=True,
                    visible=True,
                )
            )

        return surfaces

    def get_items(self) -> List[HabitatItem]:
        return list(self.items)

    def get_all_items(self) -> List[HabitatItem]:
        return self.get_items()

    def get_colliders(self) -> List[HabitatCollider]:
        return list(self.colliders)

    def get_blocking_colliders(self) -> List[HabitatCollider]:
        return [
            collider
            for collider in self.colliders
            if collider.visible and collider.blocking
        ]

    def get_support_surfaces(
        self,
    ) -> List[HorizontalSupportSurface]:
        return list(self.support_surfaces)

    def is_point_blocked(self, x: float, y: float) -> bool:
        return any(
            collider.blocking
            and collider.visible
            and collider.x <= x <= collider.x + collider.width
            and collider.y <= y <= collider.y + collider.height
            for collider in self.colliders
        )

    def is_rect_blocked(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> bool:
        right = x + width
        bottom = y + height

        for collider in self.get_blocking_colliders():
            collider_right = collider.x + collider.width
            collider_bottom = collider.y + collider.height

            if (
                x < collider_right
                and right > collider.x
                and y < collider_bottom
                and bottom > collider.y
            ):
                return True

        return False

    def is_surface_clear(
        self,
        surface: HorizontalSupportSurface,
        object_width: float,
        safety_margin: float = 0.0,
    ) -> bool:
        required = object_width + safety_margin * 2.0
        return required <= surface.width

    def find_spot_on_surface(
        self,
        surface: HorizontalSupportSurface,
        object_width: float,
        safety_margin: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        if not self.is_surface_clear(
            surface,
            object_width,
            safety_margin,
        ):
            return None

        left = surface.x_start + safety_margin
        right = surface.x_end - safety_margin - object_width

        if right < left:
            return None

        return round((left + right) / 2.0, 6), surface.y

    def find_safe_watch_spot(
        self,
        actor_width: float = 0.06,
        actor_height: float = 0.06,
        chair_width: float = 0.14,
        chair_height: float = 0.18,
        safety_margin: float = 0.01,
        grid_step: float = 0.05,
    ) -> Optional[Tuple[float, float]]:
        width = max(actor_width, chair_width)
        height = max(actor_height, chair_height)

        width += safety_margin * 2.0
        height += safety_margin * 2.0

        half_width = width / 2.0
        half_height = height / 2.0

        candidates = []

        x = half_width
        while x <= 1.0 - half_width + 1e-9:
            y = half_height

            while y <= 1.0 - half_height + 1e-9:
                rect_x = x - half_width
                rect_y = y - half_height

                if not self.is_rect_blocked(
                    rect_x,
                    rect_y,
                    width,
                    height,
                ):
                    score = (
                        y * 2.0
                        - abs(x - 0.5) * 0.2
                        + min(
                            x,
                            1.0 - x,
                            y,
                            1.0 - y,
                        ) * 0.05
                    )

                    candidates.append((score, x, y))

                y += grid_step

            x += grid_step

        if not candidates:
            return None

        candidates.sort(
            key=lambda value: (
                -value[0],
                value[2],
                abs(value[1] - 0.5),
                value[1],
            )
        )

        _, safe_x, safe_y = candidates[0]
        return round(safe_x, 6), round(safe_y, 6)

    def _signature(self) -> Tuple[Any, ...]:
        return (
            self.canvas_width,
            self.canvas_height,
            self.current_scene_name,
            self.habitat_scene_name,
            tuple(
                (
                    item.scene_item_id,
                    item.source_name,
                    item.group_path,
                    item.visible,
                    item.classification,
                    round(item.pixel_x, 2),
                    round(item.pixel_y, 2),
                    round(item.pixel_width, 2),
                    round(item.pixel_height, 2),
                )
                for item in self.items
            ),
        )


def print_habitat(habitat: ObsHabitat) -> None:
    print()
    print("OBS HABITAT")
    print("===========")
    print("Connected: YES")
    print(
        f"Canvas: {habitat.canvas_width}x{habitat.canvas_height}"
    )
    print(f"Program Scene: {habitat.current_scene_name}")
    print(f"Habitat Scene: {habitat.habitat_scene_name}")
    print()
    print("Scene items:")

    for item in habitat.items:
        print(f"[{item.classification}] {item.source_name}")
        print(
            f"visible: {'YES' if item.visible else 'NO'}"
        )

        if item.group_path:
            print(
                f"group path: {habitat.habitat_scene_name} > "
                f"{item.group_path}"
            )

        print(f"reason: {item.classification_reason}")
        print(f"collision mode: {item.collision_mode}")
        print(
            "rect px: "
            f"x={item.pixel_x:.1f}, "
            f"y={item.pixel_y:.1f}, "
            f"w={item.pixel_width:.1f}, "
            f"h={item.pixel_height:.1f}"
        )

        if habitat.canvas_width and habitat.canvas_height:
            print(
                "normalized: "
                f"x={item.pixel_x / habitat.canvas_width:.4f}, "
                f"y={item.pixel_y / habitat.canvas_height:.4f}, "
                f"w={item.pixel_width / habitat.canvas_width:.4f}, "
                f"h={item.pixel_height / habitat.canvas_height:.4f}"
            )

    print()
    print(f"Habitat colliders: {len(habitat.colliders)}")
    print(
        "Blocking colliders: "
        f"{len(habitat.get_blocking_colliders())}"
    )

    safe_spot = habitat.find_safe_watch_spot()

    if safe_spot is None:
        print("Safe WATCHING spot: NONE")
    else:
        safe_x, safe_y = safe_spot
        print("Safe WATCHING spot:")
        print(
            f"normalized: x={safe_x:.4f}, y={safe_y:.4f}"
        )
        print(
            f"pixel: x={safe_x * habitat.canvas_width:.1f}, "
            f"y={safe_y * habitat.canvas_height:.1f}"
        )

    print()
    print(f"Support surfaces: {len(habitat.support_surfaces)}")

    for surface in habitat.support_surfaces:
        print(f"[{surface.surface_type}] {surface.name}")
        print(
            f"x={surface.x_start:.4f}..{surface.x_end:.4f}"
        )
        print(
            f"y={surface.y:.4f}, width={surface.width:.4f}"
        )
        print(
            f"active={surface.active} visible={surface.visible}"
        )


def run_sample_test() -> None:
    assert classify_source(
        "kombo",
        "OBS_SOURCE_TYPE_INPUT",
        "browser_source",
        "WORLD > kick web",
        800,
        76,
        2560,
        1440,
    ) == ("IGNORE", "explicit habitat ignore source", "NONE")

    for blocking_only_name in ("event list", "mario"):
        classification = classify_source(
            blocking_only_name,
            "OBS_SOURCE_TYPE_INPUT",
            "browser_source",
            "WORLD > kick web",
            300,
            200,
            2560,
            1440,
        )
        assert classification[0] == "SOLID_OVERLAY"
        assert classification[2] == "RECT"
        assert not source_allows_support(blocking_only_name)

    assert classify_source(
        "Browser 2",
        "OBS_SOURCE_TYPE_INPUT",
        "browser_source",
        "",
        1920,
        1080,
        2560,
        1440,
    )[0] == "IGNORE"

    assert classify_source(
        "chat botrix",
        "OBS_SOURCE_TYPE_INPUT",
        "browser_source",
        "WORLD > kick chat",
        600,
        400,
        2560,
        1440,
    )[0] == "SOLID_OVERLAY"

    assert classify_source(
        "future meme source",
        "OBS_SOURCE_TYPE_INPUT",
        "browser_source",
        "WORLD > meme",
        2560,
        1440,
        2560,
        1440,
    )[0] == "IGNORE"

    assert classify_source(
        "Game Capture",
        "OBS_SOURCE_TYPE_INPUT",
        "game_capture",
        "",
        1920,
        1080,
        2560,
        1440,
    )[0] == "BACKGROUND"

    print("Offline tests: OK")


def watch(habitat: ObsHabitat, interval: float) -> None:
    print("LIVE habitat watch started. Press CTRL+C to stop.")

    previous = None

    try:
        while True:
            changed = habitat.refresh()
            current = habitat._signature()

            if previous is None:
                print_habitat(habitat)
                previous = current
            elif changed:
                print()
                print("HABITAT UPDATE")
                print_habitat(habitat)
                previous = current

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nHabitat watch stopped.")

    finally:
        habitat.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="OBS WORLD habitat reader"
    )

    parser.add_argument(
        "--host",
        default=os.getenv("OBS_WEBSOCKET_HOST", DEFAULT_HOST),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(
            os.getenv(
                "OBS_WEBSOCKET_PORT",
                str(DEFAULT_PORT),
            )
        ),
    )

    parser.add_argument(
        "--password",
        default=os.getenv("OBS_WEBSOCKET_PASSWORD"),
    )

    parser.add_argument(
        "--scene",
        default=os.getenv("OBS_HABITAT_SCENE"),
        help=(
            "Scene to inspect. Default: current OBS program scene. "
            "Use WORLD for a fixed WORLD scene."
        ),
    )

    parser.add_argument("--watch", action="store_true")

    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
    )

    parser.add_argument(
        "--sample-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sample_test:
        run_sample_test()
        return 0

    if args.password:
        print("OBS password: configured")
    else:
        print("OBS password: not configured")

    habitat = ObsHabitat(
        host=args.host,
        port=args.port,
        password=args.password,
        scene_name=args.scene,
    )

    try:
        habitat.connect()
    except Exception as exc:
        print(f"OBS connection failed: {exc}")
        return 2

    if args.watch:
        watch(
            habitat,
            max(0.2, args.interval),
        )
    else:
        try:
            print_habitat(habitat)
        finally:
            habitat.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
