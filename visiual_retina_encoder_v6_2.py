"""V6.2 diagnostic bridge: live V5 retina map into the real MaleCNS LIF.

This is an experiment only.  It observes network activity and never calls OBS,
the motor decoder, a server, Gemini or any fly-control action.  The visual to
source-event conversion is an engineering assumption, not biological dynamics.
"""
from pathlib import Path
import argparse
import csv
import sys
import time
import types
import numpy as np

TEST_DURATION_SECONDS = 20.0
INPUT_MODE = "LIVE"                 # DARK, STATIC, LIVE
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
W_SYN = 0.110                      # current project live_brain/OBS calibration
CHUNK_MS = 125.0                   # one brain step per 8-FPS visual frame
RNG_SEED = 101


def block(title):
    print("\n" + "=" * 92 + "\n" + title + "\n" + "=" * 92, flush=True)


class RetinaMap:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as z:
            self.data = {k: z[k].copy() for k in z.files}
        required = {"coordinate_side", "hex1", "hex2", "neuron_idx", "normalized_x", "normalized_y"}
        missing = required - set(self.data)
        if missing:
            raise ValueError("retina_spatial_map_v1.npz missing: " + ", ".join(sorted(missing)))
        self.n = len(self.data["neuron_idx"])
        if len(np.unique(self.data["neuron_idx"])) != self.n:
            raise ValueError("retina map neuron_idx is not unique")
        self.left = np.flatnonzero(self.data["coordinate_side"] == -1)
        self.right = np.flatnonzero(self.data["coordinate_side"] == 1)
        keys = [(int(s), int(a), int(b)) for s, a, b in zip(self.data["coordinate_side"], self.data["hex1"], self.data["hex2"])]
        lookup = {}; unique = []; inverse = []
        for key in keys:
            if key not in lookup:
                lookup[key] = len(unique); unique.append(key)
            inverse.append(lookup[key])
        self.inverse = np.asarray(inverse, dtype=np.int64)
        self.unique = unique
        self.coordinate_side = np.asarray([x[0] for x in unique], dtype=np.int8)
        self.x = np.asarray([np.mean(self.data["normalized_x"][self.inverse == i]) for i in range(len(unique))])
        self.y = np.asarray([np.mean(self.data["normalized_y"][self.inverse == i]) for i in range(len(unique))])

    def project(self, gray):
        px = np.clip(np.rint(self.x * (gray.shape[1] - 1)).astype(int), 0, gray.shape[1] - 1)
        py = np.clip(np.rint(self.y * (gray.shape[0] - 1)).astype(int), 0, gray.shape[0] - 1)
        return gray[py, px].astype(np.float32) / 255.0


def gray_frame(screenshot, width):
    frame = np.asarray(screenshot, dtype=np.uint8)[..., :3]
    step = max(1, int(round(frame.shape[1] / width)))
    small = frame[::step, ::step]
    return ((77 * small[..., 2].astype(np.uint16) + 150 * small[..., 1].astype(np.uint16) + 29 * small[..., 0].astype(np.uint16)) >> 8).astype(np.uint8)


def configure_brain(retina, root, seed):
    # These are the actual current project APIs and the explicit W_SYN used by
    # live_brain.py / obs_overlay_server_v6.py.  No motor decoder is imported.
    sys.path.insert(0, str(root))
    from lif_engine_realtime import RealtimeMaleCNSLIF
    from brain_session import ContinuousBrainSession
    brain = RealtimeMaleCNSLIF()
    brain.w_syn = W_SYN
    session = ContinuousBrainSession(brain, source_types=[], seed=seed)
    source = retina.data["neuron_idx"].astype(np.int32)
    if ((source < 0) | (source >= brain.N)).any():
        raise ValueError("retina source index outside brain.N")
    session.source_indices = np.ascontiguousarray(source)
    session.source_mask = np.zeros(brain.N, dtype=bool)
    session.source_mask[source] = True
    session.per_source_rates = np.zeros(len(source), dtype=np.float32)

    def create_schedule(self, _rate_hz, chunk_steps):
        probability = np.clip(self.per_source_rates * self.brain.dt / 1000.0, 0.0, 1.0)
        schedule = np.zeros((chunk_steps, len(self.source_indices)), dtype=np.uint8)
        for step in range(chunk_steps):
            schedule[step] = self.rng.random(len(self.source_indices)) < probability
        return np.ascontiguousarray(schedule)

    session.create_schedule = types.MethodType(create_schedule, session)
    session.step(rate_hz=0.0, chunk_ms=1.0)
    session.reset(seed)
    masks = {name: brain.superclasses.eq(name).to_numpy() for name in ("ol_sensory", "ol_intrinsic", "visual_projection", "descending_neuron")}
    return brain, session, masks


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(args.root) if args.root else Path(__file__).resolve().parent
    map_path = root / "data/processed/retina_spatial_map_v1.npz"
    if not map_path.is_file():
        raise FileNotFoundError(map_path)
    retina = RetinaMap(map_path)
    block("INPUT / RETINA MAP")
    print(map_path)
    with np.load(map_path, allow_pickle=False) as z:
        for key in z.files: print(key, z[key].shape, z[key].dtype)
    print("neurons", retina.n, "coordinates", len(retina.unique), "LEFT neurons", len(retina.left), "RIGHT neurons", len(retina.right))
    print("W_SYN", W_SYN, "(explicit project live_brain/OBS calibration; unchanged)")
    brain, session, masks = configure_brain(retina, root, args.seed)
    print("MaleCNS brain N", brain.N, "source indices", len(session.source_indices), "W_SYN", brain.w_syn)
    print("visual population sizes", {name: int(mask.sum()) for name, mask in masks.items()})
    import mss
    with mss.MSS() as capture:
        for i, monitor in enumerate(capture.monitors): print("monitor", i, monitor)
        if args.monitor >= len(capture.monitors): raise ValueError(f"Monitor {args.monitor} unavailable")
        monitor = capture.monitors[args.monitor]
        rng = np.random.default_rng(args.seed); frozen = None; previous = None; baseline = None
        start = time.perf_counter(); next_frame = start; next_report = start + 1.0
        frames = 0; total_source = 0; total_network = 0; total_desc = 0; total_motor = 0; peak_total = 0
        report_rows = []; active_coords = []; active_neurons = []; max_rates = []; temporal_means = []; peak_temporal = 0.0
        try:
            while time.perf_counter() - start < args.duration:
                frame_start = time.perf_counter(); cap_start = time.perf_counter()
                if args.input_mode == "DARK":
                    h = max(1, int(round(monitor["height"] / max(1, monitor["width"] / args.process_width)))); gray = np.zeros((h, args.process_width), dtype=np.uint8); cap_ms = 0.0
                elif args.input_mode == "STATIC" and frozen is not None:
                    gray = frozen; cap_ms = 0.0
                else:
                    gray = gray_frame(capture.grab(monitor), args.process_width); cap_ms = (time.perf_counter() - cap_start) * 1000.0
                    if args.input_mode == "STATIC" and frozen is None: frozen = gray.copy()
                proj_start = time.perf_counter(); luminance = retina.project(gray); temporal = np.abs(luminance - previous) if previous is not None else np.zeros_like(luminance)
                if baseline is None: baseline = luminance.copy()
                warming = frame_start - start < args.baseline_seconds
                if warming:
                    baseline = 0.90 * baseline + 0.10 * luminance; active_coord = np.zeros(len(luminance), dtype=bool); rates_coord = np.zeros(len(retina.unique), dtype=np.float32)
                else:
                    delta = np.abs(luminance - baseline); active_coord = (delta >= args.luminance_threshold) | (temporal >= args.temporal_threshold); drive = args.luminance_gain * np.maximum(0.0, delta - args.luminance_threshold) + args.temporal_gain * np.maximum(0.0, temporal - args.temporal_threshold); rates_coord = np.clip(args.base_rate_hz + drive, 0.0, args.max_rate_hz).astype(np.float32); baseline = 0.995 * baseline + 0.005 * luminance
                rates = rates_coord[retina.inverse]
                session.per_source_rates = rates
                brain_start = time.perf_counter(); result = session.step(float(np.mean(rates)), CHUNK_MS); brain_ms = (time.perf_counter() - brain_start) * 1000.0
                # The session's actual stimulus_events is the injected source count.
                source_events = int(result["stimulus_events"]); total_source += source_events; total_network += result["total_spikes"]; total_desc += result["descending_spikes"]; total_motor += result["motor_spikes"]; peak_total = max(peak_total, result["total_spikes"])
                active_neuron = active_coord[retina.inverse]; frames += 1; active_coords.append(float(active_coord.mean())); active_neurons.append(float(active_neuron.mean())); max_rates.append(float(np.mean(rates >= args.max_rate_hz - 1e-6))); temporal_means.append(float(temporal.mean())); peak_temporal = max(peak_temporal, float(temporal.max()))
                if frame_start >= next_report:
                    elapsed = frame_start - start; fps = frames / max(elapsed, 1e-9); left_coord = retina.coordinate_side == -1; right_coord = ~left_coord; left_neuron = retina.data["coordinate_side"] == -1; right_neuron = ~left_neuron
                    print(f"fps={fps:5.2f} cap={cap_ms:6.1f}ms brain={brain_ms:7.1f}ms | RETINA L coord={int(active_coord[left_coord].sum())}/{int(left_coord.sum())} neuron={int(active_neuron[left_neuron].sum())}/{len(retina.left)} events={int(result['stimulus_events'])} | R coord={int(active_coord[right_coord].sum())}/{int(right_coord.sum())} neuron={int(active_neuron[right_neuron].sum())}/{len(retina.right)} | BRAIN total={result['total_spikes']:6d} desc={result['descending_spikes']:5d} motor(observed)={result['motor_spikes']:5d} hot={result['hot_neurons']:5d} peak_hot={result['peak_hot']:5d}")
                    report_rows.append(dict(frame=frames, elapsed_s=elapsed, fps=fps, capture_ms=cap_ms, brain_ms=brain_ms, active_coordinates=int(active_coord.sum()), active_sensory_neurons=int(active_neuron.sum()), source_events=source_events, network_spikes=result["total_spikes"], descending_spikes=result["descending_spikes"], motor_spikes=result["motor_spikes"], hot_neurons=result["hot_neurons"], peak_hot=result["peak_hot"], ol_sensory_spikes=int(result["spike_counts"][masks["ol_sensory"]].sum()), ol_intrinsic_spikes=int(result["spike_counts"][masks["ol_intrinsic"]].sum()), visual_projection_spikes=int(result["spike_counts"][masks["visual_projection"]].sum()), descending_population_spikes=int(result["spike_counts"][masks["descending_neuron"]].sum())))
                    next_report = frame_start + 1.0
                previous = luminance.copy(); next_frame += 1.0 / args.fps; delay = next_frame - time.perf_counter()
                if delay > 0: time.sleep(delay)
        except KeyboardInterrupt: print("\nSafe shutdown requested.")
    runtime = time.perf_counter() - start
    block("SESSION SUMMARY")
    print("input_mode", args.input_mode, "runtime_s", round(runtime, 3), "frames", frames, "average_fps", round(frames / max(runtime, 1e-9), 3)); print("retina source_events", total_source, "events_per_sec", round(total_source / max(runtime, 1e-9), 3), "mean_active_sensory_percent", round(100 * np.mean(active_neurons), 4)); print("brain network_spikes", total_network, "spikes_per_sec", round(total_network / max(runtime, 1e-9), 3), "descending", total_desc, "descending_per_sec", round(total_desc / max(runtime, 1e-9), 3), "motor_observed", total_motor, "peak_total_window", peak_total); print("mean_events_per_source_event", total_network / max(total_source, 1), "mean_temporal_contrast", f"{np.mean(temporal_means):.8f}", "peak_temporal_contrast", f"{peak_temporal:.8f}")
    if total_source and total_network == 0: print("WARNING: source events generated but network spikes stayed zero.")
    if total_network and total_desc == 0: print("WARNING: network ran but descending activity stayed zero.")
    if np.mean(max_rates) > 0.95: print("WARNING: sensory rate saturation is continuous.")
    report_dir = root / "data/processed/retina_reports"; report_dir.mkdir(parents=True, exist_ok=True); report = report_dir / f"live_retina_v6_2_brain_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]) if report_rows else ["frame"]); writer.writeheader(); writer.writerows(report_rows)
    print("report", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=None); parser.add_argument("--input-mode", choices=("DARK", "STATIC", "LIVE"), default=INPUT_MODE); parser.add_argument("--duration", type=float, default=TEST_DURATION_SECONDS); parser.add_argument("--baseline-seconds", type=float, default=BASELINE_SECONDS); parser.add_argument("--monitor", type=int, default=MONITOR_INDEX); parser.add_argument("--fps", type=float, default=CAPTURE_FPS); parser.add_argument("--process-width", type=int, default=PROCESS_WIDTH); parser.add_argument("--seed", type=int, default=RNG_SEED); parser.add_argument("--luminance-threshold", type=float, default=LUMINANCE_THRESHOLD); parser.add_argument("--temporal-threshold", type=float, default=TEMPORAL_THRESHOLD); parser.add_argument("--base-rate-hz", type=float, default=BASE_RATE_HZ); parser.add_argument("--luminance-gain", type=float, default=LUMINANCE_GAIN); parser.add_argument("--temporal-gain", type=float, default=TEMPORAL_GAIN); parser.add_argument("--max-rate-hz", type=float, default=MAX_RATE_HZ)
    args = parser.parse_args()
    try: main(args)
    except Exception as error: print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr); sys.exit(1)
