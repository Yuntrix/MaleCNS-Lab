"""Screen to Gemini to validated perception JSON.

Independent of MaleCNS, retina, OBS and motor code. Uses the installed
google-genai Interactions API without function calling or automatic tools.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_MONITOR = 2
DEFAULT_WIDTH = 768
DEFAULT_JPEG_QUALITY = 65
DEFAULT_INTERVAL = 1.8
DEFAULT_THRESHOLD = 0.025
DEFAULT_DURATION = 30.0
MAX_RETRIES = 2
REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "vision_reports"


class Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    label: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    position: Literal["left", "center", "right", "multiple", "unknown"]
    size: Literal["small", "medium", "large", "unknown"]
    moving: Literal["yes", "no", "unknown"]


class Perception(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene: Literal["desktop", "game", "indoor", "outdoor", "video", "image", "unknown"]
    summary: str = Field(max_length=300)
    entities: List[Entity] = Field(default_factory=list, max_length=12)
    salient_event: Literal["none", "appearance", "disappearance", "movement", "large_change", "unknown"]
    brightness: Literal["dark", "normal", "bright", "unknown"]
    motion_level: Literal["none", "low", "medium", "high", "unknown"]
    text_present: bool
    possible_food: bool
    possible_person: bool
    possible_animal: bool
    uncertainty: str = Field(max_length=200)


PROMPT = (
    "Analyze this screenshot as perception only. Return only the requested JSON. "
    "Report only clearly visible entities, use short labels, and use unknown when unsure. "
    "The position is approximate. moving must be the string yes, no, or unknown. "
    "Never output behavior or commands such as fly left, fly right, jump, escape, attack, eat, or go to food."
)


def _schema() -> Dict[str, Any]:
    # Keep the wire schema in the small JSON-schema subset accepted by the
    # Interactions endpoint; Pydantic remains the authoritative validator.
    entity = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "position": {"type": "string", "enum": ["left", "center", "right", "multiple", "unknown"]},
            "size": {"type": "string", "enum": ["small", "medium", "large", "unknown"]},
            "moving": {"type": "string", "enum": ["yes", "no", "unknown"]},
        },
        "required": ["label", "confidence", "position", "size", "moving"],
    }
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string", "enum": ["desktop", "game", "indoor", "outdoor", "video", "image", "unknown"]},
            "summary": {"type": "string"}, "entities": {"type": "array", "items": entity},
            "salient_event": {"type": "string", "enum": ["none", "appearance", "disappearance", "movement", "large_change", "unknown"]},
            "brightness": {"type": "string", "enum": ["dark", "normal", "bright", "unknown"]},
            "motion_level": {"type": "string", "enum": ["none", "low", "medium", "high", "unknown"]},
            "text_present": {"type": "boolean"}, "possible_food": {"type": "boolean"},
            "possible_person": {"type": "boolean"}, "possible_animal": {"type": "boolean"},
            "uncertainty": {"type": "string"},
        },
        "required": ["scene", "summary", "entities", "salient_event", "brightness", "motion_level",
                      "text_present", "possible_food", "possible_person", "possible_animal", "uncertainty"],
    }


def _safe_error(exc: Exception) -> str:
    text = str(exc).replace(os.getenv("GEMINI_API_KEY", ""), "[REDACTED]")
    low = text.lower()
    if "429" in low or "quota" in low or "rate" in low:
        return "Gemini quota or rate limit reached"
    if "404" in low or "not found" in low or ("model" in low and "available" in low):
        return "Gemini model unavailable"
    if "401" in low or "403" in low or "permission" in low or "api key" in low:
        return "Gemini authentication or permission failed"
    if "timeout" in low:
        return "Gemini network timeout"
    return f"Gemini request failed: {type(exc).__name__}"


def _import_dependencies():
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("Missing dependency: python -m pip install mss") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Missing dependency: python -m pip install pillow") from exc
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai sürümü güncellenmeli: python -m pip install -U google-genai") from exc
    return mss, Image, genai


def capture_frame(sct: Any, monitor_index: int, crop: Optional[Dict[str, int]] = None) -> np.ndarray:
    monitors = sct.monitors
    if monitor_index < 1 or monitor_index >= len(monitors):
        raise RuntimeError(f"Monitor {monitor_index} bulunamadı; mevcut monitor sayısı: {len(monitors) - 1}")
    region = dict(crop) if crop else dict(monitors[monitor_index])
    try:
        return np.asarray(sct.grab(region))[:, :, :3]
    except Exception as exc:
        raise RuntimeError(f"Screenshot alınamadı: {type(exc).__name__}") from exc


def encode_jpeg(frame: np.ndarray, max_width: int, quality: int, Image: Any) -> tuple[bytes, tuple[int, int]]:
    image = Image.fromarray(frame, mode="RGB")
    if image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=quality, optimize=True)
    return stream.getvalue(), image.size


def change_score(previous: Optional[np.ndarray], current: np.ndarray, Image: Any) -> float:
    if previous is None:
        return 1.0
    a = np.asarray(Image.fromarray(previous).convert("L").resize((64, 36)), dtype=np.float32)
    b = np.asarray(Image.fromarray(current).convert("L").resize((64, 36)), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


class GeminiPerception:
    def __init__(self, model: str):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY: MISSING")
        _, _, genai = _import_dependencies()
        self.client = genai.Client(api_key=key)
        if not hasattr(self.client, "interactions") or not hasattr(self.client.interactions, "create"):
            raise RuntimeError("google-genai sürümü güncellenmeli: python -m pip install -U google-genai")
        self.model = model

    def ask(self, jpeg: bytes) -> Perception:
        request_input = [
            {"type": "text", "text": PROMPT},
            {"type": "image", "data": base64.b64encode(jpeg).decode("ascii"), "mime_type": "image/jpeg"},
        ]
        last: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                interaction = self.client.interactions.create(
                    model=self.model,
                    input=request_input,
                    response_format={"type": "text", "mime_type": "application/json", "schema": _schema()},
                )
                text = getattr(interaction, "output_text", None)
                if not text:
                    raise ValueError("empty interaction.output_text")
                return Perception.model_validate_json(text)
            except (ValidationError, ValueError) as exc:
                raise RuntimeError(f"Structured perception validation failed: {type(exc).__name__}") from exc
            except Exception as exc:
                last = exc
                if attempt < MAX_RETRIES:
                    time.sleep(0.7 * (attempt + 1))
        raise RuntimeError(_safe_error(last or RuntimeError("unknown")))


def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_perception(result: Perception) -> None:
    print(f"scene: {result.scene}")
    print(f"summary: {result.summary}")
    print("entities:")
    for entity in result.entities:
        print(f"- {entity.label} | {entity.position} | {entity.size} | moving={entity.moving} | confidence={entity.confidence:.2f}")
    print(f"possible_food: {str(result.possible_food).lower()}")
    print(f"possible_person: {str(result.possible_person).lower()}")
    print(f"possible_animal: {str(result.possible_animal).lower()}")


def run(args: argparse.Namespace) -> int:
    print("==================================================\nGEMINI VISION\n==================================================")
    print("Frames sent to Gemini are external API data.")
    print(f"GEMINI_API_KEY: {'OK' if os.getenv('GEMINI_API_KEY') else 'MISSING'}")
    print(f"Gemini model: {args.model}\nmode: {args.mode}\nmonitor: {args.monitor}")
    if not os.getenv("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY before running the test.")
        return 2
    try:
        mss, Image, _ = _import_dependencies()
        vision = GeminiPerception(args.model)
    except Exception as exc:
        print(f"STARTUP ERROR: {exc}")
        return 2
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / f"gemini_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    started = time.monotonic()
    previous: Optional[np.ndarray] = None
    try:
        with mss.MSS() as sct:
            while True:
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    break
                captured = capture_frame(sct, args.monitor, None)
                score = change_score(previous, captured, Image)
                previous = captured
                if args.mode == "LIVE" and score < args.threshold:
                    print(f"SKIPPED | frame unchanged | change_score={score:.5f}")
                    time.sleep(args.interval)
                    continue
                jpeg, sent_size = encode_jpeg(captured, args.width, args.quality, Image)
                print(f"capture: {captured.shape[1]}x{captured.shape[0]} | sent image: {sent_size[0]}x{sent_size[1]} | jpeg: {len(jpeg) // 1024} KB")
                print("Calling Gemini...")
                t0 = time.monotonic()
                try:
                    result = vision.ask(jpeg)
                    latency = round((time.monotonic() - t0) * 1000.0, 2)
                    print(f"latency_ms: {latency}\n\nPERCEPTION")
                    print_perception(result)
                    write_jsonl(log_path, {"timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
                        "mode": args.mode, "monitor": args.monitor, "captured_width": int(captured.shape[1]),
                        "captured_height": int(captured.shape[0]), "sent_width": sent_size[0], "sent_height": sent_size[1],
                        "jpeg_bytes": len(jpeg), "latency_ms": latency, "frame_change_score": score,
                        "perception": result.model_dump()})
                except Exception as exc:
                    latency = round((time.monotonic() - t0) * 1000.0, 2)
                    print(f"API ERROR: {exc}")
                    write_jsonl(log_path, {"timestamp": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
                                          "sanitized_error": str(exc), "latency_ms": latency, "frame_change_score": score})
                if args.mode == "SINGLE_FRAME":
                    break
                time.sleep(args.interval)
    except Exception as exc:
        print(f"CAPTURE ERROR: {exc}")
        return 2
    print(f"JSONL: {log_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MaleCNS Gemini Vision perception-only bridge")
    parser.add_argument("--mode", choices=["SINGLE_FRAME", "LIVE"], default="SINGLE_FRAME")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--monitor", type=int, default=DEFAULT_MONITOR)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    if args.mode == "SINGLE_FRAME":
        args.duration = 1.0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
