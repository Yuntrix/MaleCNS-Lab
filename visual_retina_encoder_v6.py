"""Live screen -> V5 retina map bridge.

Default is DRY_RUN: no MaleCNS simulation and no motor/server control.  BRAIN
mode uses the existing ContinuousBrainSession API only for diagnostic network
activity.  The visual-to-spike conversion is an explicit engineering model,
not measured photoreceptor physiology.
"""
from pathlib import Path
import argparse
import csv
import sys
import time
import types
import numpy as np


# Easy-to-change capture/model settings.
MODE = "DRY_RUN"                 # DRY_RUN or BRAIN
MONITOR_INDEX = 2
CAPTURE_FPS = 8.0
PROCESS_WIDTH = 200
CROP = None                      # (left, top, width, height), or None
SHOW_DEBUG = False
BASE_RATE_HZ = 2.0
LUMINANCE_GAIN = 35.0
TEMPORAL_GAIN = 90.0
MAX_RATE_HZ = 80.0
UNKNOWN_CLASS_ENABLED = True
RNG_SEED = 101
REPORT_INTERVAL_SECONDS = 1.0


def block(title):
    print("\n" + "=" * 90 + "\n" + title + "\n" + "=" * 90, flush=True)


class RetinaLiveProjector:
    """Fast nearest-pixel sampler over the V5 normalized engineering map."""
    def __init__(self, map_path, swap_axes=False, flip_x=False, flip_y=False, rotation=0.0):
        self.map_path = Path(map_path)
        self.swap_axes, self.flip_x, self.flip_y, self.rotation = swap_axes, flip_x, flip_y, rotation
        with np.load(self.map_path, allow_pickle=False) as z:
            self.map = {k: z[k].copy() for k in z.files}
        required = {"coordinate_side", "hex1", "hex2", "neuron_idx", "normalized_x", "normalized_y"}
        missing = required - set(self.map)
        if missing:
            raise ValueError("Retina map missing: " + ", ".join(sorted(missing)))
        self.sides = {"LEFT": self.map["coordinate_side"] == -1, "RIGHT": self.map["coordinate_side"] == 1}
        self.coord_to_rows = {}
        for side, mask in self.sides.items():
            rows = np.flatnonzero(mask)
            for row in rows:
                self.coord_to_rows.setdefault((side, int(self.map["hex1"][row]), int(self.map["hex2"][row])), []).append(int(row))

    def project(self, gray):
        gray = np.asarray(gray, dtype=np.float32)
        if gray.ndim != 2 or gray.size == 0:
            raise ValueError("gray frame must be a non-empty 2D array")
        values = np.empty(len(self.map["neuron_idx"]), dtype=np.float32)
        for side, mask in self.sides.items():
            rows = np.flatnonzero(mask)
            x = np.clip(np.rint(self.map["normalized_x"][rows] * (gray.shape[1] - 1)).astype(int), 0, gray.shape[1] - 1)
            y = np.clip(np.rint(self.map["normalized_y"][rows] * (gray.shape[0] - 1)).astype(int), 0, gray.shape[0] - 1)
            values[rows] = gray[y, x] / 255.0
        return {"LEFT": values[self.sides["LEFT"]], "RIGHT": values[self.sides["RIGHT"]], "all": values}

    def hex_values(self, signals):
        result = {}
        for side, mask in self.sides.items():
            rows = np.flatnonzero(mask)
            for row in rows:
                key = (side, int(self.map["hex1"][row]), int(self.map["hex2"][row]))
                result[key] = float(signals["all"][row])
        return result


def gray_frame(screenshot, process_width):
    arr = np.asarray(screenshot, dtype=np.uint8)[..., :3]
    step = max(1, int(round(arr.shape[1] / process_width)))
    small = arr[::step, ::step]
    return ((77 * small[..., 2].astype(np.uint16) + 150 * small[..., 1].astype(np.uint16) + 29 * small[..., 0].astype(np.uint16)) >> 8).astype(np.uint8)


def rates_and_events(projector, signal, previous, rng, class_codes):
    luminance = signal["all"]
    contrast = np.abs(luminance - previous) if previous is not None else np.zeros_like(luminance)
    rates = np.clip(BASE_RATE_HZ + LUMINANCE_GAIN * luminance + TEMPORAL_GAIN * contrast, 0, MAX_RATE_HZ).astype(np.float32)
    events = rng.random(len(rates)) < (rates / CAPTURE_FPS)
    events &= rates > 0
    diagnostics = {}
    for side, mask in projector.sides.items():
        idx = np.flatnonzero(mask)
        diagnostics[side] = {"active_hex": int(np.count_nonzero(signal[side] > 0.02)), "mean_intensity": float(np.mean(signal[side])), "mean_contrast": float(np.mean(contrast[idx])), "events": int(events[idx].sum()), "neurons": len(idx), "rate_mean": float(np.mean(rates[idx]))}
    return rates, events, contrast, diagnostics


def brain_session_for_map(projector, seed):
    from brain_session import ContinuousBrainSession
    from lif_engine_realtime import RealtimeMaleCNSLIF
    brain = RealtimeMaleCNSLIF()
    session = ContinuousBrainSession(brain, source_types=[], seed=seed)
    source = projector.map["neuron_idx"].astype(np.int32)
    source = np.unique(source)
    session.source_indices = np.ascontiguousarray(source)
    session.source_mask = np.zeros(brain.N, dtype=bool)
    session.source_mask[source] = True
    session.per_source_rates = np.zeros(len(source), dtype=np.float32)

    def create_schedule(self, _rate_hz, chunk_steps):
        probability = np.clip(self.per_source_rates * self.brain.dt / 1000.0, 0.0, 1.0)
        out = np.zeros((chunk_steps, len(self.source_indices)), dtype=np.uint8)
        for step in range(chunk_steps):
            out[step] = self.rng.random(len(self.source_indices)) < probability
        return np.ascontiguousarray(out)

    session.create_schedule = types.MethodType(create_schedule, session)
    return brain, session


def run(args):
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    map_path = root / "data/processed/retina_spatial_map_v1.npz"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    projector = RetinaLiveProjector(map_path, args.swap_axes, args.flip_x, args.flip_y, args.rotation)
    block("RETINA MAP")
    print(map_path)
    with np.load(map_path, allow_pickle=False) as z:
        for key in z.files: print(key, z[key].shape, z[key].dtype)
    print("Rows:", len(projector.map["neuron_idx"]), "LEFT:", int(projector.sides["LEFT"].sum()), "RIGHT:", int(projector.sides["RIGHT"].sum()))
    if "cell_class_code" in projector.map:
        print("Cell-class codes:", dict(zip(*np.unique(projector.map["cell_class_code"], return_counts=True))))
    print("Transform: ENGINEERING / UNCALIBRATED", "swap_axes=", args.swap_axes, "flip_x=", args.flip_x, "flip_y=", args.flip_y, "rotation=", args.rotation)
    print("Cell classes use the same grayscale drive; no UV/color assumptions.")
    if args.mode not in ("DRY_RUN", "BRAIN"):
        raise ValueError("--mode must be DRY_RUN or BRAIN")
    brain = session = None
    if args.mode == "BRAIN":
        block("BRAIN MODE")
        brain, session = brain_session_for_map(projector, args.seed)
        print("Existing RealtimeMaleCNSLIF + ContinuousBrainSession loaded; motor output is observed only.")
    import mss
    with mss.MSS() as capture:
        print("Available monitors:")
        for i, monitor in enumerate(capture.monitors): print(i, monitor)
        if args.monitor >= len(capture.monitors):
            raise ValueError(f"Monitor {args.monitor} unavailable; choose one of 0..{len(capture.monitors)-1}")
        monitor = capture.monitors[args.monitor]
        if CROP:
            monitor = {"left": monitor["left"] + CROP[0], "top": monitor["top"] + CROP[1], "width": CROP[2], "height": CROP[3]}
        previous = None; rng = np.random.default_rng(args.seed); start = time.perf_counter(); next_frame = start; next_report = start + REPORT_INTERVAL_SECONDS
        frames = 0; total_events = 0; total_network = 0; total_desc = 0; active_left = []; active_right = []; report_rows = []
        frozen = None
        try:
            while not args.frames or frames < args.frames:
                frame_start = time.perf_counter()
                if args.state == "DARK":
                    gray = np.zeros((max(1, int(monitor["height"] / max(1, monitor["width"] / PROCESS_WIDTH))), PROCESS_WIDTH), dtype=np.uint8)
                    capture_ms = 0.0
                elif args.state == "STATIC" and frozen is not None:
                    gray = frozen; capture_ms = 0.0
                else:
                    cap_start = time.perf_counter(); screenshot = capture.grab(monitor); capture_ms = (time.perf_counter() - cap_start) * 1000.0; gray = gray_frame(screenshot, PROCESS_WIDTH)
                    if frozen is None: frozen = gray.copy()
                signal = projector.project(gray); project_start = time.perf_counter(); rates, events, contrast, diag = rates_and_events(projector, signal, previous, rng, projector.map.get("cell_class_code")); projection_ms = (time.perf_counter() - project_start) * 1000.0
                brain_result = {}; brain_ms = 0.0
                if session is not None:
                    session.per_source_rates = rates.astype(np.float32); b0 = time.perf_counter(); brain_result = session.step(float(np.mean(rates)), 1000.0 / args.fps); brain_ms = (time.perf_counter() - b0) * 1000.0; total_network += brain_result["total_spikes"]; total_desc += brain_result["descending_spikes"]
                total_events += int(events.sum()); active_left.append(diag["LEFT"]["active_hex"]); active_right.append(diag["RIGHT"]["active_hex"]); frames += 1
                now = time.perf_counter()
                if now >= next_report:
                    fps = frames / max(now - start, 1e-9); print(f"fps={fps:5.2f} cap={capture_ms:6.1f}ms proj={projection_ms:5.2f}ms brain={brain_ms:6.1f}ms | LEFT hex={diag['LEFT']['active_hex']:4d} int={diag['LEFT']['mean_intensity']:.3f} d={diag['LEFT']['mean_contrast']:.3f} events={diag['LEFT']['events']:4d} | RIGHT hex={diag['RIGHT']['active_hex']:4d} int={diag['RIGHT']['mean_intensity']:.3f} d={diag['RIGHT']['mean_contrast']:.3f} events={diag['RIGHT']['events']:4d} | total neurons={int(np.count_nonzero(events)):4d} source={int(events.sum()):4d} brain={brain_result.get('total_spikes',0):5d} desc={brain_result.get('descending_spikes',0):4d}")
                    report_rows.append(dict(frame=frames, fps=fps, capture_ms=capture_ms, projection_ms=projection_ms, brain_ms=brain_ms, left_active_hex=diag["LEFT"]["active_hex"], right_active_hex=diag["RIGHT"]["active_hex"], source_events=int(events.sum()), brain_spikes=brain_result.get("total_spikes", 0), descending_spikes=brain_result.get("descending_spikes", 0))); next_report = now + REPORT_INTERVAL_SECONDS
                previous = signal["all"].copy()
                next_frame += 1.0 / args.fps; wait = next_frame - time.perf_counter()
                if wait > 0: time.sleep(wait)
        except KeyboardInterrupt: print("\nStopping safely...")
    elapsed = time.perf_counter() - start
    block("SHUTDOWN TOTALS")
    print("mode", args.mode, "runtime_s", round(elapsed, 2), "frames", frames, "average_fps", round(frames / max(elapsed, 1e-9), 2), "mean_active_LEFT", round(float(np.mean(active_left)) if active_left else 0, 2), "mean_active_RIGHT", round(float(np.mean(active_right)) if active_right else 0, 2), "sensory_events_per_sec", round(total_events / max(elapsed, 1e-9), 2))
    if frames and np.mean(active_left) >= int(projector.sides["LEFT"].sum()) * 0.98 and np.mean(active_right) >= int(projector.sides["RIGHT"].sum()) * 0.98:
        print("WARNING: nearly all retina neurons are luminance-active; this is a diagnostic signal, not object detection.")
    if frames and total_events / max(frames * len(projector.map["neuron_idx"]), 1) > 0.95:
        print("WARNING: generated events are near the configured per-frame maximum; review BASE_RATE/LUMINANCE_GAIN/TEMPORAL_GAIN.")
    if args.mode == "BRAIN": print("total_source_spikes", total_events, "total_network_spikes", total_network, "total_descending_spikes", total_desc)
    reports = root / "data/processed/retina_reports"; reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"live_retina_v6_session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd = __import__("pandas"); pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding="utf-8-sig"); print("report", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--mode", choices=("DRY_RUN", "BRAIN"), default=MODE)
    parser.add_argument("--monitor", type=int, default=MONITOR_INDEX)
    parser.add_argument("--fps", type=float, default=CAPTURE_FPS)
    parser.add_argument("--process-width", type=int, default=PROCESS_WIDTH)
    parser.add_argument("--frames", type=int, default=0, help="0 means until Ctrl+C")
    parser.add_argument("--state", choices=("LIVE", "STATIC", "DARK"), default="LIVE")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--swap-axes", action="store_true")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--rotation", type=float, default=0.0)
    args = parser.parse_args()
    PROCESS_WIDTH = args.process_width
    try: run(args)
    except Exception as error: print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr); sys.exit(1)
