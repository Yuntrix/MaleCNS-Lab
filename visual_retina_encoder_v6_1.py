"""V6.1 live retina calibration bridge.

Default: 30-second DRY_RUN.  This file never controls the fly, OBS server,
motor decoder or Gemini.  Rates and thresholds are engineering calibration
parameters; they are not biological photoreceptor measurements.
"""
from pathlib import Path
import argparse
import csv
import sys
import time
import numpy as np

MODE = "DRY_RUN"
TEST_DURATION_SECONDS = 30.0   # None means run until Ctrl+C.
INPUT_MODE = "LIVE"             # LIVE, STATIC or DARK
MONITOR_INDEX = 2
CAPTURE_FPS = 8.0
PROCESS_WIDTH = 200
BASELINE_SECONDS = 5.0
LUMINANCE_THRESHOLD = 0.08
TEMPORAL_THRESHOLD = 0.02
BASE_RATE_HZ = 0.0
LUMINANCE_GAIN = 40.0
TEMPORAL_GAIN = 100.0
MAX_RATE_HZ = 40.0
RNG_SEED = 101
REPORT_INTERVAL_SECONDS = 1.0


def block(title):
    print("\n" + "=" * 92 + "\n" + title + "\n" + "=" * 92, flush=True)


class RetinaCalibrationProjector:
    """V5 map sampler; display geometry remains engineering/uncalibrated."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.map = {k: z[k].copy() for k in z.files}
        needed = {"coordinate_side", "hex1", "hex2", "neuron_idx", "normalized_x", "normalized_y"}
        missing = needed - set(self.map)
        if missing:
            raise ValueError("Map missing fields: " + ", ".join(sorted(missing)))
        self.neuron_count = len(self.map["neuron_idx"])
        raw_keys = [(int(s), int(a), int(b)) for s, a, b in zip(self.map["coordinate_side"], self.map["hex1"], self.map["hex2"])]
        key_to_index = {}
        unique_keys = []
        inverse = []
        for key in raw_keys:
            if key not in key_to_index:
                key_to_index[key] = len(unique_keys)
                unique_keys.append(key)
            inverse.append(key_to_index[key])
        self.coordinate_keys = raw_keys
        self.unique_keys = unique_keys
        self.inverse = np.asarray(inverse, dtype=np.int64)
        self.coordinate_side = np.array([int(k[0]) for k in self.unique_keys], dtype=np.int8)
        self.coordinate_hex1 = np.array([int(k[1]) for k in self.unique_keys], dtype=np.int64)
        self.coordinate_hex2 = np.array([int(k[2]) for k in self.unique_keys], dtype=np.int64)
        self.coordinate_x = np.zeros(len(self.unique_keys), dtype=float)
        self.coordinate_y = np.zeros(len(self.unique_keys), dtype=float)
        self.coordinate_x[:] = [np.mean(self.map["normalized_x"][self.inverse == i]) for i in range(len(self.unique_keys))]
        self.coordinate_y[:] = [np.mean(self.map["normalized_y"][self.inverse == i]) for i in range(len(self.unique_keys))]
        self.left_neurons = np.flatnonzero(self.map["coordinate_side"] == -1)
        self.right_neurons = np.flatnonzero(self.map["coordinate_side"] == 1)

    def project(self, gray):
        x = np.clip(np.rint(self.coordinate_x * (gray.shape[1] - 1)).astype(int), 0, gray.shape[1] - 1)
        y = np.clip(np.rint(self.coordinate_y * (gray.shape[0] - 1)).astype(int), 0, gray.shape[0] - 1)
        return gray[y, x].astype(np.float32) / 255.0


def to_gray(screenshot, width):
    frame = np.asarray(screenshot, dtype=np.uint8)[..., :3]
    step = max(1, int(round(frame.shape[1] / width)))
    small = frame[::step, ::step]
    return ((77 * small[..., 2].astype(np.uint16) + 150 * small[..., 1].astype(np.uint16) + 29 * small[..., 0].astype(np.uint16)) >> 8).astype(np.uint8)


def capture_frame(capture, monitor, mode, frozen, width):
    if mode == "DARK":
        h = max(1, int(round(monitor["height"] / max(1, monitor["width"] / width))))
        return np.zeros((h, width), dtype=np.uint8), frozen
    if mode == "STATIC" and frozen is not None:
        return frozen, frozen
    current = to_gray(capture.grab(monitor), width)
    return current, current.copy() if mode == "STATIC" and frozen is None else frozen


def main(args):
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    map_path = root / "data/processed/retina_spatial_map_v1.npz"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    projector = RetinaCalibrationProjector(map_path)
    block("INPUT / MAP")
    print("mode=DRY_RUN (BRAIN is intentionally disabled in V6.1)")
    print("input_mode=", args.input_mode, "duration_s=", args.duration)
    with np.load(map_path, allow_pickle=False) as z:
        for key in z.files:
            print(key, z[key].shape, z[key].dtype)
    print("sensory neurons:", projector.neuron_count, "coordinates:", len(projector.unique_keys), "LEFT coordinates:", int((projector.coordinate_side == -1).sum()), "RIGHT coordinates:", int((projector.coordinate_side == 1).sum()))
    print("Engineering calibration parameters: thresholds/gains/rates are not biological measurements.")
    import mss
    with mss.MSS() as capture:
        print("monitors:")
        for i, monitor in enumerate(capture.monitors):
            print(i, monitor)
        if args.monitor >= len(capture.monitors):
            raise ValueError(f"Monitor {args.monitor} unavailable")
        monitor = capture.monitors[args.monitor]
        rng = np.random.default_rng(args.seed)
        frozen = None
        previous = None
        baseline = None
        start = time.perf_counter()
        next_frame = start
        next_report = start + REPORT_INTERVAL_SECONDS
        frames = 0
        total_events = 0
        active_coordinate_samples = []
        active_neuron_samples = []
        max_rate_samples = []
        temporal_samples = []
        peak_temporal = 0.0
        rows = []
        saturation_seconds = 0.0
        weak_temporal_seconds = 0.0
        try:
            while args.duration is None or time.perf_counter() - start < args.duration:
                frame_start = time.perf_counter()
                gray, frozen = capture_frame(capture, monitor, args.input_mode, frozen, args.process_width)
                luminance = projector.project(gray)
                if baseline is None:
                    baseline = luminance.copy()
                temporal = np.abs(luminance - previous) if previous is not None else np.zeros_like(luminance)
                elapsed = frame_start - start
                acquiring = elapsed < args.baseline_seconds
                if acquiring:
                    baseline = 0.90 * baseline + 0.10 * luminance
                    active_coordinates = np.zeros(len(luminance), dtype=bool)
                    rates = np.full(projector.neuron_count, BASE_RATE_HZ, dtype=np.float32)
                else:
                    baseline_delta = np.abs(luminance - baseline)
                    active_coordinates = (baseline_delta >= args.luminance_threshold) | (temporal >= args.temporal_threshold)
                    drive = args.luminance_gain * np.maximum(0.0, baseline_delta - args.luminance_threshold) + args.temporal_gain * np.maximum(0.0, temporal - args.temporal_threshold)
                    rates_coord = np.clip(args.base_rate_hz + drive, 0.0, args.max_rate_hz)
                    rates = rates_coord[projector.inverse].astype(np.float32)
                    # Slowly follow static luminance; temporal changes remain visible.
                    baseline = 0.995 * baseline + 0.005 * luminance
                events = rng.random(projector.neuron_count) < np.clip(rates / args.fps, 0.0, 1.0)
                if acquiring:
                    events[:] = False
                active_neurons = np.bincount(projector.inverse, weights=active_coordinates[projector.inverse].astype(np.int8), minlength=len(active_coordinates)) > 0
                active_neuron_count = int(np.count_nonzero(active_coordinates[projector.inverse]))
                max_fraction = float(np.mean(rates >= args.max_rate_hz - 1e-6))
                active_coordinate_fraction = float(np.mean(active_coordinates))
                active_neuron_fraction = active_neuron_count / projector.neuron_count
                total_events += int(events.sum())
                frames += 1
                active_coordinate_samples.append(active_coordinate_fraction); active_neuron_samples.append(active_neuron_fraction); max_rate_samples.append(max_fraction); temporal_samples.append(float(np.mean(temporal))); peak_temporal = max(peak_temporal, float(np.max(temporal)))
                if not acquiring and active_neuron_fraction > 0.90: saturation_seconds += 1.0 / args.fps
                if not acquiring and active_coordinate_fraction < 0.001: weak_temporal_seconds += 1.0 / args.fps
                if frame_start >= next_report:
                    left = projector.coordinate_side == -1; right = ~left
                    print(f"fps={frames/max(time.perf_counter()-start,1e-9):5.2f} baseline={'ON' if acquiring else 'OFF'} | LEFT coordinates={int(np.count_nonzero(active_coordinates[left]))}/{int(left.sum())} neurons={int(np.count_nonzero(active_coordinates[projector.inverse][projector.map['coordinate_side']==-1]))}/{len(projector.left_neurons)} int={float(np.mean(luminance[left])):.4f} d={float(np.mean(temporal[left])):.5f} temporal_active={int(np.count_nonzero(temporal[left]>=args.temporal_threshold))} events={int(events[projector.left_neurons].sum())} | RIGHT coordinates={int(np.count_nonzero(active_coordinates[right]))}/{int(right.sum())} neurons={int(np.count_nonzero(active_coordinates[projector.inverse][projector.map['coordinate_side']==1]))}/{len(projector.right_neurons)} int={float(np.mean(luminance[right])):.4f} d={float(np.mean(temporal[right])):.5f} temporal_active={int(np.count_nonzero(temporal[right]>=args.temporal_threshold))} events={int(events[projector.right_neurons].sum())} | active_neuron_fraction={active_neuron_fraction:.3f} max_rate_fraction={max_fraction:.3f}")
                    rows.append(dict(frame=frames, elapsed_s=elapsed, active_coordinate_fraction=active_coordinate_fraction, active_neuron_fraction=active_neuron_fraction, max_rate_fraction=max_fraction, mean_temporal_contrast=float(np.mean(temporal)), peak_temporal_contrast=float(np.max(temporal)), source_events=int(events.sum())))
                    next_report = frame_start + REPORT_INTERVAL_SECONDS
                previous = luminance.copy()
                next_frame += 1.0 / args.fps
                delay = next_frame - time.perf_counter()
                if delay > 0: time.sleep(delay)
        except KeyboardInterrupt:
            print("\nSafe shutdown requested.")
    runtime = time.perf_counter() - start
    block("SHUTDOWN TOTALS")
    print("runtime_s", round(runtime, 3)); print("frames", frames); print("fps", round(frames / max(runtime, 1e-9), 3)); print("mean_active_coordinate_percent", round(100 * float(np.mean(active_coordinate_samples)) if rows else 0, 4)); print("mean_active_sensory_percent", round(100 * float(np.mean(active_neuron_samples)) if rows else 0, 4)); print("mean_events_per_sec", round(total_events / max(runtime, 1e-9), 3)); print("mean_events_per_neuron_sec", round(total_events / max(runtime * projector.neuron_count, 1), 6)); print("max_rate_fraction", round(float(np.mean(max_rate_samples)) if rows else 0, 6)); print("mean_temporal_contrast", f"{float(np.mean(temporal_samples)) if rows else 0:.8f}"); print("peak_temporal_contrast", f"{peak_temporal:.8f}")
    if saturation_seconds > 5.0: print("WARNING: RETINA SATURATION (>90% sensory neurons active for >5 s)")
    if weak_temporal_seconds > 5.0: print("WARNING: TEMPORAL SIGNAL TOO WEAK (<0.1% coordinates for >5 s)")
    report_dir = root / "data/processed/retina_reports"; report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"live_retina_v6_1_session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame"]); writer.writeheader(); writer.writerows(rows)
    print("report", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None); parser.add_argument("--mode", default="DRY_RUN", choices=("DRY_RUN",))
    parser.add_argument("--input-mode", default=INPUT_MODE, choices=("LIVE", "STATIC", "DARK")); parser.add_argument("--duration", type=float, default=TEST_DURATION_SECONDS)
    parser.add_argument("--baseline-seconds", type=float, default=BASELINE_SECONDS); parser.add_argument("--monitor", type=int, default=MONITOR_INDEX); parser.add_argument("--fps", type=float, default=CAPTURE_FPS); parser.add_argument("--process-width", type=int, default=PROCESS_WIDTH); parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--luminance-threshold", type=float, default=LUMINANCE_THRESHOLD); parser.add_argument("--temporal-threshold", type=float, default=TEMPORAL_THRESHOLD); parser.add_argument("--base-rate-hz", type=float, default=BASE_RATE_HZ); parser.add_argument("--luminance-gain", type=float, default=LUMINANCE_GAIN); parser.add_argument("--temporal-gain", type=float, default=TEMPORAL_GAIN); parser.add_argument("--max-rate-hz", type=float, default=MAX_RATE_HZ)
    args = parser.parse_args()
    try: main(args)
    except Exception as error: print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr); sys.exit(1)
