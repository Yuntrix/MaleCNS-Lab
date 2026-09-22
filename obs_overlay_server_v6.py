# Offline browser runtime:
# assets/vendor/fly-renderer.bundle.js
# Bu dosya projeye eklendi: Three.js 0.180.0 + GLTFLoader.
#
# GLB arama sırası:
# 1. assets/fly/fly_master.glb
# 2. fly_master.glb (mevcut proje kökündeki model)
#
# Mevcut fizik collider değerleri korunmuştur.
# CHARACTER_RENDER_SCALE yalnız görsel ölçeği değiştirir.

import json
import os
import random
import threading
import time

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from lif_engine_realtime import RealtimeMaleCNSLIF
from brain_session_multi import MultiInputBrainSession
from motor_decoder_v2 import MotorDecoder


HOST = '127.0.0.1'
PORT = 8765

W_SYN = 0.110
CHUNK_MS = 20.0
SEED = 404
RUNAWAY_SPIKES_PER_CHUNK = 10000

LOOMING_RATE_HZ = 40.0
LOOMING_DURATION_MS = 60.0

STEERING_DN_TYPE = 'DNa04'
STEERING_HOLD_MS = 700.0

ENDOGENOUS_RATE_HZ = 2.0
ENDOGENOUS_DURATION_MS = 200.0
ENDOGENOUS_NEURON_COUNT = 128
FIRST_ENDOGENOUS_AT_MS = 3000.0
ENDOGENOUS_INTERVAL_MIN_MS = 8000.0
ENDOGENOUS_INTERVAL_MAX_MS = 18000.0

MOVE_HOLD_MS = 450.0
FLIGHT_CONTROL_MS = 1800.0
FLIGHT_REFRESH_MS = 700.0

# Body x/y now represent the GLB LANDING_POINT. A grounded fly therefore
# uses the actual screen-floor coordinate instead of the old CSS anchor.
GROUND_Y = 1.0
FLIGHT_TARGET_Y = 0.69

LEFT_WALL = 0.03
RIGHT_WALL = 0.96
CEILING_Y = 0.05

MOVE_TARGET_SPEED = 0.075
MOVE_ACCELERATION = 0.30
TAKEOFF_VELOCITY = -0.52

AIR_HORIZONTAL_ACCELERATION = 0.055
MAX_AIR_HORIZONTAL_SPEED = 0.12

FLIGHT_VERTICAL_GAIN = 1.50
FLIGHT_VERTICAL_BLEND = 0.13
MAX_FLIGHT_UP_SPEED = -0.36
MAX_FLIGHT_DOWN_SPEED = 0.18

GRAVITY = 1.20
JUMP_VELOCITY = -0.58


PROJECT_DIR = Path(__file__).resolve().parent

CHARACTER_ASSET_PATHS = (
    PROJECT_DIR / 'assets' / 'fly' / 'fly_master.glb',
    PROJECT_DIR / 'fly_master.glb',
)
CHARACTER_RUNTIME_PATH = (
    PROJECT_DIR / 'assets' / 'vendor' / 'fly-renderer.bundle.js'
)


def _allowed_local_file(path):
    try:
        resolved = path.resolve(strict=True)
        return (
            resolved == path
            and resolved.is_relative_to(PROJECT_DIR)
            and resolved.is_file()
        )
    except (OSError, RuntimeError):
        return False


def _character_asset_path():
    return next(
        (
            path
            for path in CHARACTER_ASSET_PATHS
            if _allowed_local_file(path)
        ),
        None,
    )


if _character_asset_path() is None:
    print('[CHARACTER] GLB asset missing')
else:
    print('[CHARACTER] Local GLB asset found')

if not _allowed_local_file(CHARACTER_RUNTIME_PATH):
    print('[CHARACTER] Local renderer bundle missing')


CACHE_DIR = PROJECT_DIR / 'data' / 'processed'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ENDOGENOUS_CACHE = CACHE_DIR / 'endogenous_drive_indices.npy'
LOOMING_LEFT_CACHE = CACHE_DIR / 'looming_left_indices.npy'
LOOMING_RIGHT_CACHE = CACHE_DIR / 'looming_right_indices.npy'

ANNOTATION_FILE = (
    PROJECT_DIR
    / 'data'
    / 'raw'
    / 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
)


state_lock = threading.Lock()

shared_state = {
    'brain_time_ms': 0.0,
    'raw_action': 'IDLE',
    'total_spikes': 0,
    'descending_spikes': 0,
    'motor_spikes': 0,
    'flight': 0,
    'jump': 0,
    'other': 0,
    'visual_input': 'NONE',
    'looming_active': False,
    'looming_left_active': False,
    'looming_right_active': False,
    'looming_center_active': False,
    'steering_dn_type': STEERING_DN_TYPE,
    'steering_left_spikes': 0,
    'steering_right_spikes': 0,
    'steering_signal': 0.0,
    'steering_direction': 'NONE',
    'steering_hold_remaining_ms': 0.0,
    'endogenous_active': False,
    'endogenous_bursts': 0,
    'next_endogenous_ms': FIRST_ENDOGENOUS_AT_MS,
    'body_state': 'IDLE',
    'x': 0.22,
    'y': GROUND_Y,
    'vx': 0.0,
    'vy': 0.0,
    'facing': 1,
    'airborne': False,
    'flight_control_remaining_ms': 0.0,
    'compute_ms': 0.0,
    'max_compute_ms': 0.0,
    'backlog_ms': 0.0,
    'runaway': False,
    'behavior_id': 'NORMAL',
    'behavior_source': 'AUTONOMOUS',
    'behavior_priority': 100,
    'behavior_active': False,
    'behavior_phase': 'NORMAL',
    'behavior_cancel_requested': False,
    'cancel_requested': False,
    'chair_visible': False,
    'chair_x': 0.0,
    'chair_y': 0.0,
    'chair_width': 0.10,
    'chair_height': 0.12,
    'neural_body_x': 0.22,
    'neural_body_y': GROUND_Y,
    'neural_body_state': 'IDLE',
    'world_collision_ready': False,
    'world_collider_count': 0,
    'world_collision_snapshot_age_ms': None,
    'world_habitat_scene': '',
    'world_collision_error': '',
    'grounded_on_world_surface': False,
    'support_surface_name': '',
    'collision_contact': False,
    'collision_contact_side': '',
    'collision_surface_name': '',
    'collision_turn_active': False,
    'collision_turn_remaining_ms': 0.0,
    'collision_turn_direction': 'NONE',
    'fly_collider_left': 0.0,
    'fly_collider_right': 0.0,
    'fly_collider_top': 0.0,
    'fly_collider_bottom': 0.0,
}


# Presentation state only. Does not change neural/body/world decisions.
RENDER_LOOP_ANIMATIONS = {
    'IDLE': 'FLY_IDLE',
    'MOVE': 'FLY_WALK',
    'FLY': 'FLY_FLIGHT',
}
RENDER_TURN_ANIMATIONS = {
    -1: 'FLY_TURN_LEFT',
    1: 'FLY_TURN_RIGHT',
}


class RenderAnimationState:

    def __init__(self):
        self.session_id = str(time.time_ns())
        self.seq = 0
        self.previous = None
        self.loop = 'FLY_IDLE'
        self.animation = self.loop
        self.is_loop = True
        self.events = []

    def update(self, snapshot, now_ms):
        phase = str(snapshot.get('behavior_phase', 'NORMAL'))
        current = {
            'phase': phase,
            'airborne': bool(snapshot.get('airborne', False)),
            'facing': 1 if snapshot.get('facing', 1) >= 0 else -1,
            'x': float(snapshot.get('x', 0.22)),
            'y': float(snapshot.get('y', GROUND_Y)),
        }
        previous = self.previous
        mode = 'IDLE'

        if phase == 'NORMAL':
            if current['airborne']:
                mode = 'FLY'
            elif snapshot.get('body_state') == 'MOVE':
                mode = 'MOVE'
        elif phase == 'WATCHING_ENTER' and previous is not None:
            dx = abs(current['x'] - previous['x'])
            dy = abs(current['y'] - previous['y'])
            if dx > 1e-6 or dy > 1e-6:
                mode = 'FLY' if dy > 1e-6 else 'MOVE'

        loop = RENDER_LOOP_ANIMATIONS[mode]
        phase_changed = (
            previous is not None
            and previous['phase'] != phase
        )

        if loop != self.loop or phase_changed:
            self.seq += 1
            self.animation = loop
            self.is_loop = True

        self.loop = loop

        # Observe the final state after steering, movement and collisions.
        # WATCHING handoffs are not physical takeoff/landing events.
        if previous is not None and phase == previous['phase']:
            names = []

            if (
                phase == 'NORMAL'
                and current['airborne'] != previous['airborne']
            ):
                names.append(
                    'FLY_TAKEOFF'
                    if current['airborne']
                    else 'FLY_LAND'
                )

            if current['facing'] != previous['facing']:
                names.append(
                    RENDER_TURN_ANIMATIONS[current['facing']]
                )

            for name in names:
                self.seq += 1
                self.animation = name
                self.is_loop = False
                self.events.append({
                    'seq': self.seq,
                    'name': name,
                    'time_ms': now_ms,
                    'phase': phase,
                })

        # Retain recent events across multiple 20 ms brain updates so a
        # 40 ms browser poll can observe more than one transition.
        self.events = [
            event
            for event in self.events
            if 0 <= now_ms - event['time_ms'] <= 1500
        ][-32:]

        self.previous = current

        snapshot.update({
            'render_session': self.session_id,
            'render_clock_ms': round(now_ms, 3),
            'render_animation': self.animation,
            'render_animation_loop': self.is_loop,
            'render_loop_animation': self.loop,
            'render_animation_seq': self.seq,
            'render_animation_events': list(self.events),
        })


render_animation_state = RenderAnimationState()
render_animation_state.update(
    shared_state,
    time.monotonic() * 1000.0,
)

commands = {
    'looming_left_ms': 0.0,
    'looming_right_ms': 0.0,
    'looming_center_ms': 0.0,
    'endogenous_ms': 0.0,
}


print('=' * 80)
print('MALECNS AUTONOMOUS OBS SERVER V6')
print('ANATOMICAL VISUAL INPUT + DNa04 STEERING DECODER')
print('=' * 80)

brain = RealtimeMaleCNSLIF()
brain.w_syn = W_SYN
decoder = MotorDecoder()

print()
print('Nöron:', brain.N)
print('Bağlantı:', len(brain.indices))
print('W_SYN:', brain.w_syn)


def build_endogenous_population():
    if ENDOGENOUS_CACHE.exists():
        cached = np.load(ENDOGENOUS_CACHE).astype(np.int32)
        if len(cached) == ENDOGENOUS_NEURON_COUNT:
            print('Endogenous population cache yüklendi:')
            print(ENDOGENOUS_CACHE)
            return cached

    print('Endogenous population oluşturuluyor...')

    neurons = brain.neurons
    out_degree = np.diff(brain.indptr).astype(np.int32)
    superclass = neurons['superclass'].fillna('').astype(str).to_numpy()
    nt_role = neurons['nt_role'].fillna('').astype(str).to_numpy()

    base_mask = (
        (superclass == 'cb_intrinsic')
        & (nt_role == 'fast_excitatory')
        & ~brain.motor_mask
        & ~brain.descending_mask
    )
    base_indices = np.flatnonzero(base_mask)
    base_out = out_degree[base_indices]
    out_low = float(np.percentile(base_out, 25))
    out_high = float(np.percentile(base_out, 90))

    direct_dn_strength = np.zeros(brain.N, dtype=np.float64)
    direct_dn_targets = np.zeros(brain.N, dtype=np.int32)

    for neuron_idx in base_indices:
        degree = out_degree[neuron_idx]
        if degree < out_low or degree > out_high:
            continue

        start = brain.indptr[neuron_idx]
        end = brain.indptr[neuron_idx + 1]

        if end <= start:
            continue

        targets = brain.indices[start:end]
        weights = brain.weights[start:end]
        dn_mask = brain.descending_mask[targets] & (weights > 0)

        if not np.any(dn_mask):
            continue

        direct_dn_targets[neuron_idx] = int(
            np.count_nonzero(dn_mask)
        )
        direct_dn_strength[neuron_idx] = float(
            np.sum(weights[dn_mask])
        )

    candidate_mask = (
        base_mask
        & (direct_dn_targets > 0)
        & (out_degree >= out_low)
        & (out_degree <= out_high)
    )
    candidates = np.flatnonzero(candidate_mask)
    scores = np.zeros(brain.N, dtype=np.float64)
    scores[candidates] = (
        direct_dn_strength[candidates]
        / np.sqrt(np.maximum(out_degree[candidates], 1))
    )
    order = np.argsort(scores[candidates])[::-1]
    ranked = candidates[order]
    selected = ranked[:ENDOGENOUS_NEURON_COUNT].astype(np.int32)

    np.save(ENDOGENOUS_CACHE, selected)
    print('Endogenous cache kaydedildi:')
    print(ENDOGENOUS_CACHE)
    return selected


endogenous_indices = build_endogenous_population()


def load_visual_laterality_populations():
    if not LOOMING_LEFT_CACHE.exists():
        raise FileNotFoundError(
            'looming_left_indices.npy bulunamadı. '
            'Önce build_visual_laterality_indices.py çalıştır.'
        )

    if not LOOMING_RIGHT_CACHE.exists():
        raise FileNotFoundError(
            'looming_right_indices.npy bulunamadı. '
            'Önce build_visual_laterality_indices.py çalıştır.'
        )

    left_indices = np.load(LOOMING_LEFT_CACHE).astype(
        np.int32,
        copy=False,
    )
    right_indices = np.load(LOOMING_RIGHT_CACHE).astype(
        np.int32,
        copy=False,
    )
    overlap = np.intersect1d(left_indices, right_indices)

    if len(overlap) != 0:
        raise RuntimeError('Visual LEFT / RIGHT pools overlap.')

    types = brain.neurons['type'].fillna('').astype(str)
    exact_all = np.flatnonzero(
        types.isin(['LPLC2', 'LC4']).to_numpy()
    ).astype(np.int32)

    union = np.sort(
        np.concatenate([left_indices, right_indices])
    )

    if not np.array_equal(np.sort(exact_all), union):
        raise RuntimeError(
            'Saved visual laterality pools do not exactly match '
            'the current LPLC2 + LC4 population.'
        )

    return (
        np.ascontiguousarray(left_indices),
        np.ascontiguousarray(right_indices),
    )


looming_left_indices, looming_right_indices = (
    load_visual_laterality_populations()
)

print()
print('Endogenous neurons:', len(endogenous_indices))
print('Anatomical LEFT looming neurons:', len(looming_left_indices))
print('Anatomical RIGHT looming neurons:', len(looming_right_indices))
print(
    'Combined looming neurons:',
    len(looming_left_indices) + len(looming_right_indices),
)


def _instance_side(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value.endswith('_L'):
        return 'L'
    if value.endswith('_R'):
        return 'R'
    return None


def _soma_side(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {'L', 'R'}:
        return value
    return None


def build_steering_dn_masks():
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(ANNOTATION_FILE)

    annotations = pd.read_feather(
        ANNOTATION_FILE,
        columns=['bodyId', 'instance', 'somaSide'],
    )
    annotations = annotations.drop_duplicates(
        subset=['bodyId'],
        keep='first',
    )

    meta = brain.neurons[
        ['neuron_idx', 'bodyId', 'type']
    ].copy()

    meta = meta.merge(
        annotations,
        on='bodyId',
        how='left',
        validate='one_to_one',
    )
    meta['instance_side'] = meta['instance'].apply(_instance_side)
    meta['soma_side'] = meta['somaSide'].apply(_soma_side)
    meta['side'] = meta['instance_side']

    missing = meta['side'].isna()
    meta.loc[missing, 'side'] = meta.loc[missing, 'soma_side']

    type_clean = meta['type'].fillna('').astype(str).str.strip()

    left_indices = meta.loc[
        (type_clean == STEERING_DN_TYPE) & (meta['side'] == 'L'),
        'neuron_idx',
    ].to_numpy(dtype=np.int32)

    right_indices = meta.loc[
        (type_clean == STEERING_DN_TYPE) & (meta['side'] == 'R'),
        'neuron_idx',
    ].to_numpy(dtype=np.int32)

    if len(left_indices) != 1:
        raise RuntimeError(
            f'{STEERING_DN_TYPE} LEFT count expected 1, got '
            f'{len(left_indices)}'
        )

    if len(right_indices) != 1:
        raise RuntimeError(
            f'{STEERING_DN_TYPE} RIGHT count expected 1, got '
            f'{len(right_indices)}'
        )

    left_mask = np.zeros(brain.N, dtype=bool)
    right_mask = np.zeros(brain.N, dtype=bool)
    left_mask[left_indices] = True
    right_mask[right_indices] = True

    print()
    print(
        f'Steering DN {STEERING_DN_TYPE} LEFT index:',
        int(left_indices[0]),
    )
    print(
        f'Steering DN {STEERING_DN_TYPE} RIGHT index:',
        int(right_indices[0]),
    )

    return (
        np.ascontiguousarray(left_mask),
        np.ascontiguousarray(right_mask),
    )


steering_left_mask, steering_right_mask = build_steering_dn_masks()

session = MultiInputBrainSession(
    brain,
    input_groups={
        'endogenous': endogenous_indices,
        'looming_left': looming_left_indices,
        'looming_right': looming_right_indices,
    },
    seed=SEED,
)

print('Numba warmup...')
session.step(
    group_rates={
        'endogenous': 0.0,
        'looming_left': 0.0,
        'looming_right': 0.0,
    },
    chunk_ms=1.0,
)
session.reset(seed=SEED)
print('Warmup tamam.')


# Existing body/flight controller.
class BodyState:

    def __init__(self):
        self.x = 0.22
        self.y = GROUND_Y
        self.vx = 0.0
        self.vy = 0.0
        self.facing = 1
        self.state = 'IDLE'
        self.airborne = False
        self.move_until_ms = 0.0
        self.flight_control_until_ms = 0.0
        self.steering_until_ms = 0.0
        self.last_steering_direction = 0

    def receive_steering_signal(self, direction, brain_time_ms):
        if direction not in (-1, 1):
            return

        # DNa04 side -> same-side modeled motor bias.
        self.facing = int(direction)
        self.last_steering_direction = int(direction)
        self.steering_until_ms = max(
            self.steering_until_ms,
            brain_time_ms + STEERING_HOLD_MS,
        )

    def receive_flight_signal(self, brain_time_ms):
        if not self.airborne:
            self.airborne = True
            self.state = 'FLY'
            self.vy = TAKEOFF_VELOCITY
            self.vx += 0.025 * self.facing
            self.flight_control_until_ms = (
                brain_time_ms + FLIGHT_CONTROL_MS
            )
        else:
            self.flight_control_until_ms = max(
                self.flight_control_until_ms,
                brain_time_ms + FLIGHT_REFRESH_MS,
            )
            self.state = 'FLY'

    def update(self, raw_action, brain_time_ms, dt_seconds):
        if raw_action == 'FLY':
            self.receive_flight_signal(brain_time_ms)
        elif raw_action == 'JUMP':
            if not self.airborne:
                self.airborne = True
                self.state = 'JUMP'
                self.vy = JUMP_VELOCITY
        elif raw_action == 'MOVE':
            if not self.airborne:
                self.move_until_ms = brain_time_ms + MOVE_HOLD_MS

        flight_control_active = (
            self.airborne
            and brain_time_ms < self.flight_control_until_ms
        )

        if self.airborne:
            if flight_control_active:
                self.state = 'FLY'
                altitude_error = FLIGHT_TARGET_Y - self.y
                desired_vy = altitude_error * FLIGHT_VERTICAL_GAIN
                desired_vy = max(
                    MAX_FLIGHT_UP_SPEED,
                    min(MAX_FLIGHT_DOWN_SPEED, desired_vy),
                )
                self.vy += (
                    (desired_vy - self.vy) * FLIGHT_VERTICAL_BLEND
                )
            else:
                self.state = 'FLY'
                self.vy += GRAVITY * dt_seconds
        elif brain_time_ms < self.move_until_ms:
            self.state = 'MOVE'
        else:
            self.state = 'IDLE'

        if self.airborne:
            self.vx += (
                AIR_HORIZONTAL_ACCELERATION
                * self.facing
                * dt_seconds
            )
            self.vx = max(
                -MAX_AIR_HORIZONTAL_SPEED,
                min(MAX_AIR_HORIZONTAL_SPEED, self.vx),
            )
        elif self.state == 'MOVE':
            target_vx = MOVE_TARGET_SPEED * self.facing
            if self.vx < target_vx:
                self.vx += MOVE_ACCELERATION * dt_seconds
                self.vx = min(self.vx, target_vx)
            elif self.vx > target_vx:
                self.vx -= MOVE_ACCELERATION * dt_seconds
                self.vx = max(self.vx, target_vx)
        else:
            self.vx *= 0.88

        self.x += self.vx * dt_seconds
        self.y += self.vy * dt_seconds

        if self.x <= LEFT_WALL:
            self.x = LEFT_WALL
            self.facing = 1
            self.vx = abs(self.vx)
        elif self.x >= RIGHT_WALL:
            self.x = RIGHT_WALL
            self.facing = -1
            self.vx = -abs(self.vx)

        if self.y <= CEILING_Y:
            self.y = CEILING_Y
            if self.vy < 0:
                self.vy = 0.0

        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            if self.vy > 0:
                self.vy = 0.0

            if flight_control_active:
                self.airborne = True
                self.state = 'FLY'
                self.vy = -0.20
            else:
                if self.airborne:
                    self.airborne = False
                if brain_time_ms < self.move_until_ms:
                    self.state = 'MOVE'
                else:
                    self.state = 'IDLE'

    def steering_remaining(self, brain_time_ms):
        return max(
            0.0,
            self.steering_until_ms - brain_time_ms,
        )

    def flight_remaining(self, brain_time_ms):
        return max(
            0.0,
            self.flight_control_until_ms - brain_time_ms,
        )


BEHAVIOR_PRIORITY_KICKS = 300
BEHAVIOR_PRIORITY_SUBSCRIBER = 200
BEHAVIOR_PRIORITY_AUTONOMOUS = 100

WATCHING_MOVE_SPEED = 0.20
WATCHING_ENTER_MIN_SECONDS = 0.45
WATCHING_ENTER_MAX_SECONDS = 2.50
WATCHING_SIT_SECONDS = 0.5
WATCHING_HANDOFF_SECONDS = 0.45
WATCHING_EXIT_SECONDS = 0.55
WATCHING_MOVE_SPEED = 0.20

# WATCHING prop and seated-fly geometry. These values are independent from
# the physical fly collider and can be tuned after the final chair asset.
WATCHING_PROP_WIDTH = 0.065
WATCHING_PROP_HEIGHT = 0.085
WATCHING_PROP_SAFETY_MARGIN = 0.006
WATCHING_SEAT_HEIGHT_RATIO = 0.48

behavior_lock = threading.Lock()

watching_state = {
    'active': False,
    'phase': 'NORMAL',
    'started_at': 0.0,
    'phase_started_at': 0.0,
    'cancel_requested': False,
    'start_x': 0.22,
    'start_y': GROUND_Y,
    'target_x': 0.5,
    'target_y': GROUND_Y,
    'chair': None,
}

WATCHING_RECENT_HISTORY_LIMIT = 3
watching_random = random.Random()
placement_history = []


def _placement_key(placement):
    return (
        placement.surface_name,
        round(float(placement.x), 6),
        round(float(placement.y), 6),
    )


def _placement_distance(placement, previous_key):
    _, previous_x, previous_y = previous_key
    dx = float(placement.x) - previous_x
    dy = float(placement.y) - previous_y
    return (dx * dx + dy * dy) ** 0.5


def _watching_candidate_weight(placement, candidates):
    generic_scores = [
        float(candidate.score)
        for candidate in candidates
    ]
    minimum_score = min(generic_scores)
    maximum_score = max(generic_scores)
    score_range = maximum_score - minimum_score

    if score_range <= 1e-9:
        generic_factor = 1.0
    else:
        generic_factor = (
            float(placement.score) - minimum_score
        ) / score_range

    weight = 1.0 + generic_factor * 1.5
    weight += float(placement.y) * 2.0

    if placement.surface_type.upper() == 'SCREEN_FLOOR':
        weight += 2.0

    center_distance = abs(float(placement.x) - 0.5)
    center_factor = max(0.0, 1.0 - center_distance / 0.5)
    weight *= 1.0 - center_factor * 0.60

    for index, recent in enumerate(
        reversed(placement_history[-WATCHING_RECENT_HISTORY_LIMIT:])
    ):
        recent_key = _placement_key(recent)
        distance = _placement_distance(placement, recent_key)

        if distance < 0.05:
            weight *= 0.08
        elif distance < 0.12:
            weight *= 0.35
        elif distance < 0.22:
            weight *= 0.70

        if recent.surface_name == placement.surface_name:
            weight *= 0.82

        if index == 0 and distance < 0.03:
            weight *= 0.05

    return max(weight, 0.001)


def _choose_watching_candidate(candidates):
    if not candidates:
        return None

    weighted = [
        (
            candidate,
            _watching_candidate_weight(candidate, candidates),
        )
        for candidate in candidates
    ]
    weighted.sort(
        key=lambda item: (
            -item[1],
            item[0].surface_name,
            round(float(item[0].x), 6),
            round(float(item[0].y), 6),
        )
    )

    total_weight = sum(
        weight
        for _, weight in weighted
    )

    if total_weight <= 0.0:
        selected = weighted[0][0]
    else:
        selected_value = watching_random.random() * total_weight
        cumulative = 0.0
        selected = weighted[-1][0]

        for candidate, weight in weighted:
            cumulative += weight
            if selected_value <= cumulative:
                selected = candidate
                break

    placement_history.append(selected)
    if len(placement_history) > WATCHING_RECENT_HISTORY_LIMIT:
        del placement_history[:-WATCHING_RECENT_HISTORY_LIMIT]

    print(
        f'[WATCHING SELECTION] candidates={len(candidates)} '
        f'selected={selected.surface_name} '
        f'x={float(selected.x):.4f} '
        f'y={float(selected.y):.4f}'
    )
    return selected


def _lerp(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    t = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * t


def _watching_render_state(now):
    with behavior_lock:
        if not watching_state['active']:
            return None

        phase = watching_state['phase']
        phase_started = watching_state['phase_started_at']
        sx = watching_state['start_x']
        sy = watching_state['start_y']
        tx = watching_state['target_x']
        ty = watching_state['target_y']
        cancel = watching_state['cancel_requested']
        chair = watching_state['chair']
        elapsed = max(0.0, now - phase_started)

        if cancel and phase not in {'WATCHING_EXIT', 'DONE'}:
            watching_state['phase'] = 'WATCHING_EXIT'
            watching_state['phase_started_at'] = now
            phase = 'WATCHING_EXIT'
            phase_started = now
            elapsed = 0.0

        if phase == 'WATCHING_ENTER':
            duration = max(
                WATCHING_ENTER_MIN_SECONDS,
                min(
                    WATCHING_ENTER_MAX_SECONDS,
                    float(watching_state.get('enter_duration', 1.0)),
                ),
            )
            t = min(1.0, elapsed / duration)
            x, y = (
                _lerp(sx, tx, t),
                _lerp(sy, ty, t),
            )
            if t >= 1.0:
                watching_state['phase'] = 'WATCHING_SIT'
                watching_state['phase_started_at'] = now
            return x, y, 'WATCHING_ENTER', chair

        if phase == 'WATCHING_SIT':
            if elapsed >= WATCHING_SIT_SECONDS:
                watching_state['phase'] = 'WATCHING_IDLE'
            return tx, ty, 'WATCHING_SIT', chair

        if phase == 'WATCHING_IDLE':
            return tx, ty, 'WATCHING_IDLE', chair

        if phase == 'WATCHING_EXIT':
            if elapsed >= WATCHING_EXIT_SECONDS:
                watching_state['active'] = False
                watching_state['phase'] = 'DONE'
                watching_state['cancel_requested'] = False
                watching_state['chair'] = None
                return tx, ty, 'DONE', None
            return tx, ty, 'WATCHING_EXIT', chair

    return None


def _start_watching(placement):
    now = time.monotonic()

    target_y = max(
        0.05,
        min(
            GROUND_Y,
            float(placement.bottom)
            - float(placement.height) * WATCHING_SEAT_HEIGHT_RATIO,
        ),
    )

    with state_lock:
        sx = float(
            shared_state.get(
                'x',
                body.x if 'body' in globals() else 0.22,
            )
        )
        sy = float(
            shared_state.get(
                'y',
                body.y if 'body' in globals() else GROUND_Y,
            )
        )

    with behavior_lock:
        watching_state.update({
            'active': True,
            'phase': 'WATCHING_ENTER',
            'started_at': now,
            'phase_started_at': now,
            'cancel_requested': False,
            'start_x': sx,
            'start_y': sy,
            'target_x': float(placement.x),
            'target_y': target_y,
            'enter_duration': max(
                WATCHING_ENTER_MIN_SECONDS,
                min(
                    WATCHING_ENTER_MAX_SECONDS,
                    (
                        (float(placement.x) - sx) ** 2
                        + (
                            target_y - sy
                        ) ** 2
                    ) ** 0.5 / max(WATCHING_MOVE_SPEED, 1e-6),
                ),
            ),
            'chair': placement,
        })

    with state_lock:
        shared_state.update({
            'behavior_id': 'WATCHING',
            'behavior_source': 'AUTONOMOUS',
            'behavior_priority': BEHAVIOR_PRIORITY_AUTONOMOUS,
            'behavior_active': True,
            'behavior_phase': 'WATCHING_ENTER',
            'behavior_cancel_requested': False,
            'cancel_requested': False,
            'chair_visible': True,
            'chair_x': float(placement.x),
            'chair_y': float(placement.bottom),
            'chair_width': float(placement.width),
            'chair_height': float(placement.height),
        })


def _request_stop_watching():
    with behavior_lock:
        if not watching_state['active']:
            return False
        watching_state['cancel_requested'] = True
    return True


body = BodyState()


# Existing cached WORLD collision layer.
# Physical footprint relative to the GLB LANDING_POINT in final OBS pixels.
# Visual scale and collider footprint deliberately remain separate settings.
WORLD_COLLISION_REFRESH_SECONDS = 0.5
WORLD_COLLISION_RETRY_SECONDS = 2.0
WORLD_COLLISION_TURN_HOLD_MS = 260.0
COLLISION_EPSILON = 1e-5

FLY_COLLIDER_LEFT_PX = -26.0
FLY_COLLIDER_RIGHT_PX = 26.0
FLY_COLLIDER_TOP_PX = -38.0
FLY_COLLIDER_BOTTOM_PX = 0.0

COLLISION_DEFAULT_CANVAS_WIDTH = 1920.0
COLLISION_DEFAULT_CANVAS_HEIGHT = 1080.0

world_collision_lock = threading.Lock()

world_collision_snapshot = {
    'ready': False,
    'timestamp': 0.0,
    'colliders': [],
    'canvas_width': COLLISION_DEFAULT_CANVAS_WIDTH,
    'canvas_height': COLLISION_DEFAULT_CANVAS_HEIGHT,
    'scene_name': '',
    'signature': None,
    'error': '',
}
world_collision_runtime = {
    'grounded': False,
    'support_name': '',
    'contact': False,
    'contact_side': '',
    'surface_name': '',
    'turn_until_ms': 0.0,
    'turn_direction': 0,
    'contact_key': None,
}
world_collision_stop = threading.Event()


def _publish_world_collision_debug():
    now = time.monotonic()

    with world_collision_lock:
        snap = dict(world_collision_snapshot)
        runtime = dict(world_collision_runtime)

    age = None
    if snap['timestamp'] > 0.0:
        age = round(
            max(0.0, now - snap['timestamp']) * 1000.0,
            1,
        )

    with state_lock:
        shared_state['world_collision_ready'] = bool(snap['ready'])
        shared_state['world_collider_count'] = len(snap['colliders'])
        shared_state['world_collision_snapshot_age_ms'] = age
        shared_state['world_habitat_scene'] = str(
            snap.get('scene_name', '')
        )
        shared_state['world_collision_error'] = str(
            snap.get('error', '')
        )
        shared_state['grounded_on_world_surface'] = bool(
            runtime['grounded']
        )
        shared_state['support_surface_name'] = runtime['support_name']
        shared_state['collision_contact'] = bool(runtime['contact'])
        shared_state['collision_contact_side'] = runtime['contact_side']
        shared_state['collision_surface_name'] = runtime['surface_name']

        remaining = max(
            0.0,
            runtime['turn_until_ms']
            - shared_state.get('brain_time_ms', 0.0),
        )
        shared_state['collision_turn_active'] = remaining > 0.0
        shared_state['collision_turn_remaining_ms'] = round(remaining, 1)
        shared_state['collision_turn_direction'] = (
            'LEFT'
            if runtime['turn_direction'] < 0
            else 'RIGHT'
            if runtime['turn_direction'] > 0
            else 'NONE'
        )


def _refresh_world_collision_snapshot():
    password = os.getenv('OBS_WEBSOCKET_PASSWORD')

    if not password:
        with world_collision_lock:
            world_collision_snapshot['ready'] = False
            world_collision_snapshot['timestamp'] = time.monotonic()
            world_collision_snapshot['colliders'] = []
            world_collision_snapshot['canvas_width'] = (
                COLLISION_DEFAULT_CANVAS_WIDTH
            )
            world_collision_snapshot['canvas_height'] = (
                COLLISION_DEFAULT_CANVAS_HEIGHT
            )
            world_collision_snapshot['signature'] = None
            world_collision_snapshot['error'] = (
                'OBS_WEBSOCKET_PASSWORD_NOT_SET'
            )
        return

    try:
        from world.obs_habitat import ObsHabitat

        requested_scene = (
            os.getenv('OBS_HABITAT_SCENE', '').strip() or None
        )
        habitat = ObsHabitat(
            os.getenv('OBS_WEBSOCKET_HOST', '127.0.0.1'),
            int(os.getenv('OBS_WEBSOCKET_PORT', '7654')),
            password,
            scene_name=requested_scene,
        )
        habitat.connect()

        try:
            colliders = []
            for item in habitat.get_blocking_colliders():
                classification = str(item.classification).upper()
                collision_mode = str(item.collision_mode).upper()

                if not item.visible or not item.blocking:
                    continue

                if (
                    classification != 'SOLID_OVERLAY'
                    or collision_mode != 'RECT'
                ):
                    continue

                colliders.append({
                    'name': str(item.name),
                    'scene_item_id': int(item.scene_item_id),
                    'x': float(item.x),
                    'y': float(item.y),
                    'width': float(item.width),
                    'height': float(item.height),
                })
        finally:
            habitat.disconnect()

        signature = (
            int(habitat.canvas_width),
            int(habitat.canvas_height),
            habitat.current_scene_name,
            habitat.habitat_scene_name,
            tuple(
                (
                    item['scene_item_id'],
                    round(item['x'], 6),
                    round(item['y'], 6),
                    round(item['width'], 6),
                    round(item['height'], 6),
                )
                for item in colliders
            ),
        )

        with world_collision_lock:
            old_signature = world_collision_snapshot.get('signature')
            was_ready = bool(world_collision_snapshot['ready'])

            world_collision_snapshot['ready'] = True
            world_collision_snapshot['timestamp'] = time.monotonic()
            world_collision_snapshot['colliders'] = colliders
            world_collision_snapshot['canvas_width'] = float(
                habitat.canvas_width
            )
            world_collision_snapshot['canvas_height'] = float(
                habitat.canvas_height
            )
            world_collision_snapshot['scene_name'] = (
                habitat.habitat_scene_name
            )
            world_collision_snapshot['error'] = ''
            world_collision_snapshot['signature'] = signature

        if not was_ready or old_signature != signature:
            print(
                f'[WORLD] scene={habitat.habitat_scene_name} '
                f'colliders refreshed count={len(colliders)}'
            )

    except Exception as exc:
        error_name = type(exc).__name__

        with world_collision_lock:
            old_error = world_collision_snapshot.get('error', '')
            world_collision_snapshot['ready'] = False
            world_collision_snapshot['timestamp'] = time.monotonic()
            world_collision_snapshot['colliders'] = []
            world_collision_snapshot['canvas_width'] = (
                COLLISION_DEFAULT_CANVAS_WIDTH
            )
            world_collision_snapshot['canvas_height'] = (
                COLLISION_DEFAULT_CANVAS_HEIGHT
            )
            world_collision_snapshot['signature'] = None
            world_collision_snapshot['scene_name'] = ''
            world_collision_snapshot['error'] = error_name

        if old_error != error_name:
            print(f'[WORLD] collision refresh error={error_name}')


def _world_collision_refresh_loop():
    while not world_collision_stop.is_set():
        _refresh_world_collision_snapshot()
        _publish_world_collision_debug()
        if world_collision_stop.wait(WORLD_COLLISION_REFRESH_SECONDS):
            break


def _world_colliders():
    with world_collision_lock:
        return list(world_collision_snapshot['colliders'])


def get_fly_aabb(x, y, canvas_width=None, canvas_height=None):
    """Return the painted head/body AABB in normalized coordinates."""
    width = float(canvas_width or COLLISION_DEFAULT_CANVAS_WIDTH)
    height = float(canvas_height or COLLISION_DEFAULT_CANVAS_HEIGHT)

    anchor_x = float(x) * width
    anchor_y = float(y) * height

    left = (anchor_x + FLY_COLLIDER_LEFT_PX) / width
    right = (anchor_x + FLY_COLLIDER_RIGHT_PX) / width
    top = (anchor_y + FLY_COLLIDER_TOP_PX) / height
    bottom = (anchor_y + FLY_COLLIDER_BOTTOM_PX) / height

    return left, right, top, bottom


def _set_world_runtime(
    grounded=False,
    support_name='',
    contact=False,
    side='',
    surface_name='',
    turn_until_ms=0.0,
    turn_direction=0,
    contact_key=None,
):
    changed = False

    with world_collision_lock:
        old = (
            world_collision_runtime['grounded'],
            world_collision_runtime['support_name'],
            world_collision_runtime['contact'],
            world_collision_runtime['contact_side'],
            world_collision_runtime['surface_name'],
            world_collision_runtime['contact_key'],
        )
        new = (
            bool(grounded),
            str(support_name),
            bool(contact),
            str(side),
            str(surface_name),
            contact_key,
        )

        if (
            old != new
            or world_collision_runtime['turn_until_ms'] != turn_until_ms
        ):
            world_collision_runtime.update({
                'grounded': new[0],
                'support_name': new[1],
                'contact': new[2],
                'contact_side': new[3],
                'surface_name': new[4],
                'turn_until_ms': float(turn_until_ms),
                'turn_direction': int(turn_direction),
                'contact_key': contact_key,
            })
            changed = True

    if changed and contact and contact_key != old[5]:
        direction = 'LEFT' if turn_direction < 0 else 'RIGHT'
        print(
            f'[COLLISION] side hit surface={surface_name} '
            f'side={side} -> turn {direction}'
        )
    elif changed and old[1] and not grounded:
        print(f'[COLLISION] leave surface={old[1]}')

    _publish_world_collision_debug()


def resolve_world_collisions(previous_x, previous_y, brain_time_ms):
    """Resolve BodyState against cached normalized SOLID_OVERLAY rectangles."""
    with world_collision_lock:
        colliders = list(world_collision_snapshot['colliders'])
        canvas_width = world_collision_snapshot['canvas_width']
        canvas_height = world_collision_snapshot['canvas_height']

    if not colliders:
        _set_world_runtime(False, '', False, '', '', 0.0, 0, None)
        left, right, top, bottom = get_fly_aabb(
            body.x,
            body.y,
            canvas_width,
            canvas_height,
        )
        with state_lock:
            shared_state['fly_collider_left'] = round(left, 6)
            shared_state['fly_collider_right'] = round(right, 6)
            shared_state['fly_collider_top'] = round(top, 6)
            shared_state['fly_collider_bottom'] = round(bottom, 6)
        return

    (
        previous_left,
        previous_right,
        previous_top,
        previous_bottom,
    ) = get_fly_aabb(
        previous_x,
        previous_y,
        canvas_width,
        canvas_height,
    )

    left, right, top, bottom = get_fly_aabb(
        body.x,
        body.y,
        canvas_width,
        canvas_height,
    )

    dx = body.x - float(previous_x)
    dy = body.y - float(previous_y)

    support = None
    contact = False
    contact_side = ''
    contact_surface = ''
    contact_key = None
    turn_direction = 0
    turn_until = 0.0

    if not body.airborne:
        for collider in colliders:
            feet_on = abs(bottom - collider['y']) <= 0.002
            within = (
                right > collider['x'] + COLLISION_EPSILON
                and left < (
                    collider['x']
                    + collider['width']
                    - COLLISION_EPSILON
                )
            )
            if feet_on and within:
                support = collider
                body.y += collider['y'] - bottom
                body.vy = 0.0
                break

        if support is None and body.y < GROUND_Y - 0.002:
            body.airborne = True
            body.state = 'FLY'

    for collider in colliders:
        c_left = collider['x']
        c_top = collider['y']
        c_right = c_left + collider['width']
        c_bottom = c_top + collider['height']

        left, right, top, bottom = get_fly_aabb(
            body.x,
            body.y,
            canvas_width,
            canvas_height,
        )

        if not (
            right > c_left
            and left < c_right
            and bottom > c_top
            and top < c_bottom
        ):
            continue

        old_key = (
            world_collision_runtime.get('contact_key')
            if isinstance(world_collision_runtime, dict)
            else None
        )

        if dy >= 0.0 and previous_bottom <= c_top + COLLISION_EPSILON:
            body.y += c_top - bottom
            body.vy = 0.0
            body.airborne = False
            support = collider
            contact = True
            contact_side = 'TOP'
            contact_surface = collider['name']
            contact_key = (collider['scene_item_id'], 'TOP')

        elif dy < 0.0 and previous_top >= c_bottom - COLLISION_EPSILON:
            body.y += c_bottom - top
            body.vy = 0.0
            contact = True
            contact_side = 'BOTTOM'
            contact_surface = collider['name']
            contact_key = (collider['scene_item_id'], 'BOTTOM')

        elif dx > 0.0 and previous_right <= c_left + COLLISION_EPSILON:
            body.x += c_left - right
            body.vx = 0.0
            contact = True
            contact_side = 'LEFT'
            contact_surface = collider['name']
            contact_key = (collider['scene_item_id'], 'LEFT')
            turn_direction = -1

        elif dx < 0.0 and previous_left >= c_right - COLLISION_EPSILON:
            body.x += c_right - left
            body.vx = 0.0
            contact = True
            contact_side = 'RIGHT'
            contact_surface = collider['name']
            contact_key = (collider['scene_item_id'], 'RIGHT')
            turn_direction = 1

        if contact and contact_side in {'LEFT', 'RIGHT'}:
            same_contact = old_key == contact_key

            if not same_contact:
                body.facing = turn_direction
                turn_until = (
                    brain_time_ms + WORLD_COLLISION_TURN_HOLD_MS
                )
            else:
                with world_collision_lock:
                    turn_until = world_collision_runtime['turn_until_ms']
                    turn_direction = world_collision_runtime['turn_direction']
                if brain_time_ms < turn_until:
                    body.facing = turn_direction
            break

    if contact_side in {'LEFT', 'RIGHT'} and turn_until > 0.0:
        with world_collision_lock:
            prior_turn_until = world_collision_runtime['turn_until_ms']

        if (
            brain_time_ms < prior_turn_until
            and contact_key == world_collision_runtime.get('contact_key')
        ):
            body.facing = world_collision_runtime['turn_direction']
        elif contact_key != world_collision_runtime.get('contact_key'):
            body.facing = turn_direction
    else:
        with world_collision_lock:
            prior_turn_until = world_collision_runtime['turn_until_ms']
            prior_key = world_collision_runtime.get('contact_key')

        if prior_key is not None and contact_key is None:
            turn_until = 0.0
            turn_direction = 0

    if support is not None:
        _set_world_runtime(
            True,
            support['name'],
            contact,
            contact_side or 'TOP',
            contact_surface or support['name'],
            turn_until,
            turn_direction,
            contact_key,
        )
    else:
        _set_world_runtime(
            False,
            '',
            contact,
            contact_side,
            contact_surface,
            turn_until,
            turn_direction,
            contact_key,
        )

    left, right, top, bottom = get_fly_aabb(
        body.x,
        body.y,
        canvas_width,
        canvas_height,
    )

    with state_lock:
        shared_state['fly_collider_left'] = round(left, 6)
        shared_state['fly_collider_right'] = round(right, 6)
        shared_state['fly_collider_top'] = round(top, 6)
        shared_state['fly_collider_bottom'] = round(bottom, 6)


world_collision_thread = threading.Thread(
    target=_world_collision_refresh_loop,
    name='world-collision-refresh',
    daemon=True,
)
world_collision_thread.start()
_publish_world_collision_debug()


def brain_loop():
    rng = random.Random(20260918)
    brain_time_ms = 0.0
    next_endogenous_ms = FIRST_ENDOGENOUS_AT_MS
    automatic_endogenous_remaining = 0.0
    manual_endogenous_remaining = 0.0
    looming_left_remaining = 0.0
    looming_right_remaining = 0.0
    looming_center_remaining = 0.0
    endogenous_burst_count = 0
    max_compute_ms = 0.0
    dt_seconds = CHUNK_MS / 1000.0
    next_tick = time.perf_counter()
    last_raw = 'IDLE'
    last_body = 'IDLE'
    last_visual_input = 'NONE'

    print()
    print('Brain + flight controller başladı.')

    while True:
        with state_lock:
            if commands['looming_left_ms'] > 0:
                looming_left_remaining = max(
                    looming_left_remaining,
                    commands['looming_left_ms'],
                )
                commands['looming_left_ms'] = 0.0

            if commands['looming_right_ms'] > 0:
                looming_right_remaining = max(
                    looming_right_remaining,
                    commands['looming_right_ms'],
                )
                commands['looming_right_ms'] = 0.0

            if commands['looming_center_ms'] > 0:
                looming_center_remaining = max(
                    looming_center_remaining,
                    commands['looming_center_ms'],
                )
                commands['looming_center_ms'] = 0.0

            if commands['endogenous_ms'] > 0:
                manual_endogenous_remaining = max(
                    manual_endogenous_remaining,
                    commands['endogenous_ms'],
                )
                commands['endogenous_ms'] = 0.0

        if (
            brain_time_ms >= next_endogenous_ms
            and automatic_endogenous_remaining <= 0.0
        ):
            automatic_endogenous_remaining = ENDOGENOUS_DURATION_MS
            endogenous_burst_count += 1

            interval_ms = rng.uniform(
                ENDOGENOUS_INTERVAL_MIN_MS,
                ENDOGENOUS_INTERVAL_MAX_MS,
            )
            next_endogenous_ms = brain_time_ms + interval_ms

            print(
                f'[AUTO INTERNAL] brain={brain_time_ms:.0f} ms '
                f'| burst={endogenous_burst_count} '
                f'| next={next_endogenous_ms:.0f} ms'
            )

        endogenous_active = (
            automatic_endogenous_remaining > 0.0
            or manual_endogenous_remaining > 0.0
        )
        endogenous_rate = (
            ENDOGENOUS_RATE_HZ if endogenous_active else 0.0
        )

        center_active = looming_center_remaining > 0.0
        left_active = looming_left_remaining > 0.0
        right_active = looming_right_remaining > 0.0

        looming_left_rate = (
            LOOMING_RATE_HZ
            if left_active or center_active
            else 0.0
        )
        looming_right_rate = (
            LOOMING_RATE_HZ
            if right_active or center_active
            else 0.0
        )

        if center_active:
            visual_input = 'ANATOMICAL_CENTER'
        elif left_active and right_active:
            visual_input = 'ANATOMICAL_BOTH'
        elif left_active:
            visual_input = 'ANATOMICAL_LEFT'
        elif right_active:
            visual_input = 'ANATOMICAL_RIGHT'
        else:
            visual_input = 'NONE'

        if visual_input != last_visual_input:
            print(
                f'[VISUAL INPUT] {brain_time_ms:8.0f} ms '
                f'| {visual_input}'
            )
            last_visual_input = visual_input

        compute_start = time.perf_counter()
        result = session.step(
            group_rates={
                'endogenous': endogenous_rate,
                'looming_left': looming_left_rate,
                'looming_right': looming_right_rate,
            },
            chunk_ms=CHUNK_MS,
        )
        compute_ms = (
            time.perf_counter() - compute_start
        ) * 1000.0
        max_compute_ms = max(max_compute_ms, compute_ms)

        movement = decoder.decode(
            neurons=brain.neurons,
            spike_counts=result['spike_counts'],
            motor_mask=brain.motor_mask,
            steering_left_mask=steering_left_mask,
            steering_right_mask=steering_right_mask,
        )
        raw_action = movement['action']
        steering_direction = int(
            movement.get('steering_direction', 0)
        )
        steering_left_spikes = int(
            movement.get('steering_left_spikes', 0)
        )
        steering_right_spikes = int(
            movement.get('steering_right_spikes', 0)
        )
        steering_signal = float(
            movement.get('steering_signal', 0.0)
        )

        if steering_direction != 0:
            body.receive_steering_signal(
                direction=steering_direction,
                brain_time_ms=brain_time_ms,
            )
            steering_label = (
                'RIGHT' if steering_direction > 0 else 'LEFT'
            )
            print(
                f'[STEERING] {brain_time_ms:8.0f} ms '
                f'| {STEERING_DN_TYPE} '
                f'L/R={steering_left_spikes}/{steering_right_spikes} '
                f'| signal={steering_signal:+.2f} '
                f'| actuator={steering_label}'
            )

        previous_body_x = body.x
        previous_body_y = body.y

        body.update(
            raw_action=raw_action,
            brain_time_ms=brain_time_ms,
            dt_seconds=dt_seconds,
        )

        try:
            resolve_world_collisions(
                previous_body_x,
                previous_body_y,
                brain_time_ms,
            )
        except Exception as exc:
            _set_world_runtime(False, '', False, '')

        automatic_endogenous_remaining = max(
            0.0,
            automatic_endogenous_remaining - CHUNK_MS,
        )
        manual_endogenous_remaining = max(
            0.0,
            manual_endogenous_remaining - CHUNK_MS,
        )
        looming_left_remaining = max(
            0.0,
            looming_left_remaining - CHUNK_MS,
        )
        looming_right_remaining = max(
            0.0,
            looming_right_remaining - CHUNK_MS,
        )
        looming_center_remaining = max(
            0.0,
            looming_center_remaining - CHUNK_MS,
        )

        runaway = (
            result['total_spikes'] > RUNAWAY_SPIKES_PER_CHUNK
        )

        render_state = _watching_render_state(time.monotonic())

        if render_state is not None:
            (
                render_x,
                render_y,
                render_phase,
                render_chair,
            ) = render_state

            if render_phase == 'DONE':
                with state_lock:
                    body.x = float(render_x)
                    body.y = float(render_y)
                    body.vx = 0.0
                    body.vy = 0.0
                    body.airborne = False
                    body.state = 'IDLE'

                (
                    render_x,
                    render_y,
                    render_phase,
                    render_chair,
                ) = body.x, body.y, 'NORMAL', None
        else:
            (
                render_x,
                render_y,
                render_phase,
                render_chair,
            ) = body.x, body.y, 'NORMAL', None

        with state_lock:
            shared_state['brain_time_ms'] = round(brain_time_ms, 1)
            shared_state['raw_action'] = raw_action
            shared_state['total_spikes'] = result['total_spikes']
            shared_state['descending_spikes'] = result['descending_spikes']
            shared_state['motor_spikes'] = result['motor_spikes']
            shared_state['flight'] = int(
                movement.get('flight_spikes', 0)
            )
            shared_state['jump'] = int(
                movement.get('jump_spikes', 0)
            )
            shared_state['other'] = int(
                movement.get('other_spikes', 0)
            )
            shared_state['visual_input'] = visual_input
            shared_state['looming_active'] = bool(
                looming_left_rate > 0 or looming_right_rate > 0
            )
            shared_state['looming_left_active'] = bool(
                looming_left_rate > 0
            )
            shared_state['looming_right_active'] = bool(
                looming_right_rate > 0
            )
            shared_state['looming_center_active'] = bool(center_active)
            shared_state['steering_left_spikes'] = steering_left_spikes
            shared_state['steering_right_spikes'] = steering_right_spikes
            shared_state['steering_signal'] = round(steering_signal, 4)

            if steering_direction > 0:
                steering_label = 'RIGHT'
            elif steering_direction < 0:
                steering_label = 'LEFT'
            else:
                steering_label = 'NONE'

            shared_state['steering_direction'] = steering_label
            shared_state['steering_hold_remaining_ms'] = round(
                body.steering_remaining(brain_time_ms),
                1,
            )
            shared_state['endogenous_active'] = bool(
                endogenous_rate > 0
            )
            shared_state['endogenous_bursts'] = endogenous_burst_count
            shared_state['next_endogenous_ms'] = round(
                next_endogenous_ms,
                1,
            )
            shared_state['neural_body_x'] = round(body.x, 6)
            shared_state['neural_body_y'] = round(body.y, 6)
            shared_state['neural_body_state'] = body.state
            shared_state['body_state'] = (
                render_phase
                if render_phase != 'NORMAL'
                else body.state
            )
            shared_state['x'] = round(render_x, 6)
            shared_state['y'] = round(render_y, 6)
            shared_state['behavior_active'] = render_phase != 'NORMAL'
            shared_state['behavior_phase'] = render_phase
            shared_state['behavior_cancel_requested'] = bool(
                watching_state.get('cancel_requested', False)
            )
            shared_state['behavior_id'] = (
                'WATCHING'
                if render_phase != 'NORMAL'
                else 'NORMAL'
            )
            shared_state['behavior_source'] = 'AUTONOMOUS'
            shared_state['behavior_priority'] = (
                BEHAVIOR_PRIORITY_AUTONOMOUS
            )
            shared_state['cancel_requested'] = bool(
                watching_state.get('cancel_requested', False)
            )

            if render_chair is None:
                shared_state['chair_visible'] = False
                shared_state['behavior_active'] = False
                shared_state['behavior_phase'] = 'NORMAL'

            shared_state['vx'] = round(body.vx, 6)
            shared_state['vy'] = round(body.vy, 6)
            shared_state['facing'] = body.facing
            shared_state['airborne'] = body.airborne
            shared_state['flight_control_remaining_ms'] = round(
                body.flight_remaining(brain_time_ms),
                1,
            )
            shared_state['compute_ms'] = round(compute_ms, 3)
            shared_state['max_compute_ms'] = round(max_compute_ms, 3)
            shared_state['runaway'] = bool(runaway)

            render_animation_state.update(
                shared_state,
                time.monotonic() * 1000.0,
            )

        if raw_action != last_raw or body.state != last_body:
            print(
                f"[STATE] {brain_time_ms:8.0f} ms"
                f" | visual={visual_input:18s}"
                f" | raw={raw_action:5s}"
                f" | body={body.state:5s}"
                f" | spike={result['total_spikes']:4d}"
                f" | DN={result['descending_spikes']:3d}"
                f" | motor={result['motor_spikes']:3d}"
                f" | y={body.y:.3f}"
                f" | flightRemain="
                f"{body.flight_remaining(brain_time_ms):.0f} ms"
                f" | calc={compute_ms:.2f} ms"
            )
            last_raw = raw_action
            last_body = body.state

        brain_time_ms += CHUNK_MS
        next_tick += dt_seconds
        sleep_seconds = next_tick - time.perf_counter()

        if sleep_seconds > 0:
            with state_lock:
                shared_state['backlog_ms'] = 0.0
            time.sleep(sleep_seconds)
        else:
            backlog_ms = -sleep_seconds * 1000.0
            with state_lock:
                shared_state['backlog_ms'] = round(backlog_ms, 3)
            if backlog_ms > 100.0:
                next_tick = time.perf_counter()


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
}

#world {
    position: fixed;
    inset: 0;
    overflow: hidden;
    background: transparent;
}

#chair {
    display: none;
    position: absolute;
    z-index: 2;
    transform: translate(-50%, -100%);
    width: 90px;
    height: 95px;
}
.chair-back {
    position: absolute;
    left: 18%;
    top: 0;
    width: 64%;
    height: 58%;
    border: 3px solid #2b180c;
    border-radius: 9px 9px 4px 4px;
    background: #87552e;
}
.chair-seat {
    position: absolute;
    left: 8%;
    top: 52%;
    width: 84%;
    height: 18%;
    border: 3px solid #2b180c;
    border-radius: 5px;
    background: #a66a38;
}
.chair-leg {
    position: absolute;
    top: 68%;
    width: 5px;
    height: 32%;
    background: #2b180c;
}
.chair-leg.left { left: 20%; }
.chair-leg.right { right: 20%; }

#characterCanvas {
    position: absolute;
    inset: 0;
    z-index: 3;
    pointer-events: none;
}

#fly {
    z-index: 3;
    position: absolute;
    width: 76px;
    height: 58px;
    transform-origin: center center;
    will-change: transform;
}

.fly-body {
    position: absolute;
    width: 40px;
    height: 28px;
    left: 20px;
    top: 17px;
    border-radius: 50% 58% 58% 50%;
    border: 2px solid #111;
    background: repeating-linear-gradient(
        90deg,
        #252525 0px,
        #252525 7px,
        #d89e25 7px,
        #d89e25 12px
    );
}

.head {
    position: absolute;
    width: 24px;
    height: 24px;
    left: 7px;
    top: 18px;
    border-radius: 50%;
    border: 2px solid #111;
    background: #343434;
}

.eye {
    position: absolute;
    width: 9px;
    height: 13px;
    border-radius: 50%;
    background: radial-gradient(
        circle at 35% 25%,
        #ffc4c4 0%,
        #e02730 30%,
        #74080c 100%
    );
}

.eye.left {
    left: 3px;
    top: 4px;
}

.eye.right {
    left: 12px;
    top: 5px;
}

.wing {
    position: absolute;
    width: 37px;
    height: 20px;
    left: 31px;
    border-radius: 70% 30% 70% 30%;
    border: 1px solid rgba(90, 150, 180, 0.75);
    background: rgba(190, 235, 255, 0.58);
    transform-origin: 4px 10px;
}

.wing.top {
    top: 7px;
    transform: rotate(-24deg);
}

.wing.bottom {
    top: 30px;
    transform: rotate(24deg);
}

.leg {
    position: absolute;
    width: 30px;
    height: 2px;
    left: 28px;
    background: #171717;
    transform-origin: 2px 1px;
}

.leg.one {
    top: 37px;
    transform: rotate(35deg);
}

.leg.two {
    top: 41px;
    transform: rotate(65deg);
}

.leg.three {
    top: 42px;
    transform: rotate(100deg);
}

.antenna {
    position: absolute;
    width: 15px;
    height: 2px;
    background: #161616;
}

.antenna.one {
    left: -2px;
    top: 20px;
    transform: rotate(25deg);
}

.antenna.two {
    left: -1px;
    top: 29px;
    transform: rotate(-20deg);
}

#fly.MOVE {
    animation: walkBob 0.15s ease-in-out infinite alternate;
}

#fly.FLY .wing.top {
    animation: wingTop 0.055s linear infinite;
}

#fly.FLY .wing.bottom {
    animation: wingBottom 0.055s linear infinite;
}

#fly.JUMP {
    animation: jumpBody 0.12s ease-in-out infinite alternate;
}

@keyframes walkBob {
    from { margin-top: 0px; }
    to { margin-top: -3px; }
}

@keyframes wingTop {
    0% { transform: rotate(-38deg); }
    50% { transform: rotate(17deg); }
    100% { transform: rotate(-38deg); }
}

@keyframes wingBottom {
    0% { transform: rotate(38deg); }
    50% { transform: rotate(-17deg); }
    100% { transform: rotate(38deg); }
}

@keyframes jumpBody {
    from { margin-top: 0px; }
    to { margin-top: -4px; }
}

.debug-ui {
    display: none;
}

body.debug .debug-ui {
    display: block;
}

#debugPanel {
    max-height: calc(100vh - 60px);
    overflow-y: auto;
    position: fixed;
    left: 15px;
    top: 15px;
    width: 410px;
    padding: 14px;
    border-radius: 10px;
    background: rgba(0, 0, 0, 0.84);
    color: white;
    font-family: Consolas, monospace;
    font-size: 13px;
    line-height: 1.45;
    white-space: pre-wrap;
    z-index: 1000;
}

#buttons {
    position: fixed;
    right: 18px;
    top: 18px;
    z-index: 1000;
}

button {
    display: block;
    width: 190px;
    margin-bottom: 10px;
    padding: 11px;
    cursor: pointer;
    font-size: 14px;
}
</style>
</head>
<body>

<div id="world">
    <div id="chair" class="chair" aria-hidden="true"><div class="chair-back"></div><div class="chair-seat"></div><div class="chair-leg left"></div><div class="chair-leg right"></div></div>
    <div id="fly" class="IDLE">
        <div class="wing top"></div>
        <div class="wing bottom"></div>

        <div class="leg one"></div>
        <div class="leg two"></div>
        <div class="leg three"></div>

        <div class="fly-body"></div>

        <div class="head">
            <div class="eye left"></div>
            <div class="eye right"></div>
        </div>

        <div class="antenna one"></div>
        <div class="antenna two"></div>
    </div>
</div>

<div id="debugPanel" class="debug-ui">Loading...</div>

<div id="buttons" class="debug-ui">
    <button onclick="triggerVisual('/trigger_left')">
        TEST ANATOMICAL LEFT
    </button>

    <button onclick="triggerVisual('/trigger_center')">
        TEST ANATOMICAL CENTER
    </button>

    <button onclick="triggerVisual('/trigger_right')">
        TEST ANATOMICAL RIGHT
    </button>

    <button onclick="fetch('/watching/start')">START WATCHING</button>
    <button onclick="fetch('/watching/stop')">STOP WATCHING</button>

    <button onclick="triggerInternal()">
        TEST INTERNAL
    </button>
</div>

<script src="/assets/vendor/fly-renderer.bundle.js"></script>
<script>
const params = new URLSearchParams(
    window.location.search
);

const debug = params.get("debug") === "1";

if (debug) {
    document.body.classList.add("debug");
}

const fly = document.getElementById("fly");
const panel = document.getElementById("debugPanel");
const chair = document.getElementById("chair");

let state = {
    body_state: "IDLE",
    x: 0.22,
    y: 0.92,
    facing: 1
};

// Display settings only; the existing Python collider is unchanged.
const CHARACTER_RENDER_SCALE = 16; // CSS pixels per glTF unit; provisional.
const CHARACTER_SCREEN_OFFSET_X = 0;
// Moves the model up so animated feet meet the LANDING_POINT precisely.
const CHARACTER_SCREEN_OFFSET_Y = 3;

// Inspected model: Y up, head toward -Z. Rotate -Z toward screen +X.
const CHARACTER_MODEL_YAW = -Math.PI / 2;

// The inspected export includes these non-character meshes.
const CHARACTER_HIDDEN_NODES = new Set(['Cube', 'BODY_COLLIDER']);

const CHARACTER_CLIPS = {
    FLY_IDLE: true,
    FLY_WALK: true,
    FLY_FLIGHT: true,
    FLY_TAKEOFF: false,
    FLY_LAND: false,
    FLY_TURN_LEFT: false,
    FLY_TURN_RIGHT: false
};

const characterStatus = {
    loaded: false,
    ready: false,
    animation: '',
    loop: true,
    seq: 0,
    error: '',
    warning: ''
};

let character = null;
let lastFrameTime = null;

function characterFailed(message) {
    characterStatus.ready = false;
    characterStatus.error = message;
    fly.style.display = 'block';

    const canvas = document.getElementById('characterCanvas');
    if (canvas) canvas.style.display = 'none';

    console.warn('[CHARACTER] ' + message);
}

function resizeCharacter() {
    if (!character) return;

    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);

    character.renderer.setPixelRatio(
        Math.min(window.devicePixelRatio || 1, 2)
    );
    character.renderer.setSize(w, h);

    character.camera.left = -w / 2;
    character.camera.right = w / 2;
    character.camera.top = h / 2;
    character.camera.bottom = -h / 2;
    character.camera.updateProjectionMatrix();
}

function playCharacterAction(name, loop, restart = false) {
    const c = character;
    if (!c || !c.mixer) return;

    let action = c.actions.get(name);

    if (!action) {
        characterStatus.warning = 'Missing animation: ' + name;
        name = 'FLY_IDLE';
        loop = true;
        action = c.actions.get(name);
    }

    if (!action) return;

    if (
        c.current === action
        && !restart
        && characterStatus.loop === loop
    ) {
        return;
    }

    // Reuse actions. Stop the previous action so one-shots
    // cannot retain blending weight after finishing.
    if (c.current) c.current.stop();

    action.reset();
    action.enabled = true;
    action.clampWhenFinished = !loop;
    action.setLoop(
        loop ? FlyRuntime.LoopRepeat : FlyRuntime.LoopOnce,
        loop ? Infinity : 1
    );
    action.setEffectiveTimeScale(1);
    action.setEffectiveWeight(1);
    action.play();

    c.current = action;
    c.oneShot = !loop;
    characterStatus.animation = name;
    characterStatus.loop = loop;
}

function consumeCharacterState() {
    const c = character;
    if (!c || !characterStatus.ready) return;

    const seq = Number(state.render_animation_seq || 0);
    const session = String(state.render_session || '');
    const behavior = String(state.behavior_phase || 'NORMAL');

    c.loopName = String(state.render_loop_animation || 'FLY_IDLE');

    const reset = c.session !== session || seq < c.lastSeq;

    if (reset) {
        c.session = session;
        c.lastSeq = seq;
        c.pending = [];
        c.oneShot = false;

        // A new browser starts with the current loop, not old events.
        playCharacterAction(c.loopName, true, true);
    } else {
        if (behavior !== c.behavior) {
            c.pending = [];
            c.oneShot = false;
            playCharacterAction(c.loopName, true, true);
        }

        const clockMs = Number(state.render_clock_ms || 0);

        for (const event of state.render_animation_events || []) {
            const eventSeq = Number(event.seq);
            const age = clockMs - Number(event.time_ms);

            if (
                eventSeq > c.lastSeq
                && eventSeq <= seq
                && age >= 0
                && age <= 1500
                && event.phase === behavior
                && CHARACTER_CLIPS[event.name] === false
            ) {
                c.pending.push({
                    name: event.name,
                    deadline: performance.now() + 1500 - age
                });
            }
        }

        c.pending = c.pending.slice(-16);
        c.lastSeq = Math.max(c.lastSeq, seq);
    }

    c.behavior = behavior;
    characterStatus.seq = seq;

    if (!c.oneShot) advanceCharacterAction();
}

function advanceCharacterAction() {
    const c = character;
    if (!c) return;

    while (
        c.pending.length
        && c.pending[0].deadline < performance.now()
    ) {
        c.pending.shift();
    }

    const next = c.pending.shift();

    if (next) {
        playCharacterAction(next.name, false, true);
    } else {
        playCharacterAction(c.loopName, true);
    }
}

async function initCharacter() {
    if (typeof FlyRuntime === 'undefined') {
        characterFailed('Local renderer bundle missing or invalid');
        return;
    }

    const T = FlyRuntime;
    let phase = 'WebGL renderer initialization failed';
    let timeout;

    try {
        const renderer = new T.WebGLRenderer({
            alpha: true,
            antialias: true
        });

        renderer.domElement.id = 'characterCanvas';
        renderer.setClearColor(0x000000, 0);
        renderer.outputColorSpace = T.SRGBColorSpace;
        renderer.toneMapping = T.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;

        document.getElementById('world').appendChild(
            renderer.domElement
        );

        renderer.domElement.addEventListener(
            'webglcontextlost',
            event => {
                event.preventDefault();
                characterFailed(
                    'WebGL context lost; refresh Browser Source'
                );
            }
        );

        const scene = new T.Scene();
        const camera = new T.OrthographicCamera(
            -1, 1, 1, -1, 0.1, 5000
        );
        camera.position.set(0, 0, 1000);

        scene.add(new T.HemisphereLight(
            0xffffff,
            0x647080,
            2
        ));

        const keyLight = new T.DirectionalLight(0xffffff, 3);
        keyLight.position.set(-3, 5, 6);
        scene.add(keyLight);

        const placement = new T.Group();
        const facing = new T.Group();
        const modelOffset = new T.Group();

        placement.add(facing);
        facing.add(modelOffset);
        scene.add(placement);

        facing.scale.setScalar(CHARACTER_RENDER_SCALE);

        character = {
            renderer,
            scene,
            camera,
            placement,
            facing,
            modelOffset,
            mixer: null,
            actions: new Map(),
            current: null,
            oneShot: false,
            pending: [],
            loopName: 'FLY_IDLE',
            session: null,
            lastSeq: -1,
            behavior: 'NORMAL'
        };

        resizeCharacter();
        window.addEventListener('resize', resizeCharacter);

        phase = 'GLB load failed';

        const controller = new AbortController();
        timeout = setTimeout(
            () => controller.abort(),
            15000
        );

        const response = await fetch(
            '/assets/fly/fly_master.glb',
            {
                cache: 'no-store',
                signal: controller.signal
            }
        );

        if (!response.ok) {
            phase = response.status === 404
                ? 'GLB asset missing'
                : 'GLB asset unavailable';
            throw new Error(phase);
        }

        const bytes = await response.arrayBuffer();
        clearTimeout(timeout);

        phase = 'GLB parse failed or unsupported asset dependency';

        const manager = new T.LoadingManager();

        // Accept self-contained GLB files; do not fetch external resources.
        manager.setURLModifier(url => {
            if (/^(data:|blob:)/.test(url)) return url;
            throw new Error('External GLB resource blocked');
        });

        const gltf = await new T.GLTFLoader(manager).parseAsync(
            bytes,
            ''
        );

        characterStatus.loaded = true;

        const model = gltf.scene;

        model.traverse(node => {
            if (CHARACTER_HIDDEN_NODES.has(node.name)) {
                node.visible = false;
            }
            if (node.isMesh) {
                node.frustumCulled = false;
            }
        });

        modelOffset.add(model);
        model.updateMatrixWorld(true);

        const anchor = (
            model.getObjectByName('LANDING_POINT')
            || model.getObjectByName('ROOT_ANCHOR')
        );

        if (anchor) {
            const point = anchor.getWorldPosition(new T.Vector3());
            model.worldToLocal(point);
            modelOffset.position.copy(point).multiplyScalar(-1);
        } else {
            characterStatus.warning = (
                'Landing anchor missing; using model origin'
            );
        }

        character.mixer = new T.AnimationMixer(model);

        for (const clip of gltf.animations) {
            character.actions.set(
                clip.name,
                character.mixer.clipAction(clip)
            );
        }

        const missing = Object.keys(CHARACTER_CLIPS).filter(
            name => !character.actions.has(name)
        );

        if (missing.length) {
            characterStatus.warning = (
                'Missing animation: ' + missing.join(', ')
            );
        }

        phase = 'Required animation FLY_IDLE missing';

        if (!character.actions.has('FLY_IDLE')) {
            throw new Error(phase);
        }

        character.mixer.addEventListener('finished', event => {
            if (event.action !== character.current) return;
            character.oneShot = false;
        });

        characterStatus.ready = true;
        characterStatus.error = '';

        consumeCharacterState();
        renderCharacter(0);

        if (characterStatus.ready) {
            fly.style.display = 'none';
        }
    } catch (error) {
        characterFailed(phase);
    } finally {
        clearTimeout(timeout);
    }
}

function renderCharacter(delta) {
    const c = character;
    if (!c || !characterStatus.ready) return;

    try {
        if (!c.oneShot) advanceCharacterAction();

        c.mixer.update(
            Math.min(Math.max(delta, 0), 0.1)
        );

        c.placement.position.set(
            Number(state.x) * window.innerWidth
                - window.innerWidth / 2
                + CHARACTER_SCREEN_OFFSET_X,
            window.innerHeight / 2
                - Number(state.y) * window.innerHeight
                + CHARACTER_SCREEN_OFFSET_Y,
            0
        );

        c.facing.rotation.y = (
            CHARACTER_MODEL_YAW
            + (Number(state.facing) < 0 ? Math.PI : 0)
        );

        c.renderer.render(c.scene, c.camera);
    } catch (error) {
        characterFailed(
            'Character render failed; refresh Browser Source'
        );
    }
}

initCharacter();

async function triggerVisual(path) {
    try {
        await fetch(
            path,
            { cache: "no-store" }
        );
    }
    catch (error) {
    }
}

async function triggerInternal() {
    try {
        await fetch(
            "/internal",
            { cache: "no-store" }
        );
    }
    catch (error) {
    }
}

async function updateState() {
    try {
        const response = await fetch(
            "/state",
            { cache: "no-store" }
        );

        if (!response.ok) throw new Error("State unavailable");

        state = await response.json();
        consumeCharacterState();

        if (debug) {
            panel.textContent =
                "MALECNS AUTONOMOUS OBS V6\n\n"
                +
                "brain: "
                + state.brain_time_ms
                + " ms\n"
                +
                "raw action: "
                + state.raw_action
                + "\n"
                +
                "BODY STATE: "
                + state.body_state
                + "\n"
                +
                "airborne: "
                + state.airborne
                + "\n"
                +
                "flight control: "
                + state.flight_control_remaining_ms
                + " ms\n\n"
                +
                "VISUAL INPUT: "
                + state.visual_input
                + "\n"
                +
                "anat left active: "
                + state.looming_left_active
                + "\n"
                +
                "anat right active: "
                + state.looming_right_active
                + "\n"
                +
                "center active: "
                + state.looming_center_active
                + "\n\n"
                +
                "STEERING DN: "
                + state.steering_dn_type
                + "\n"
                +
                "DNa04 L/R spikes: "
                + state.steering_left_spikes
                + "/"
                + state.steering_right_spikes
                + "\n"
                +
                "steering signal: "
                + Number(state.steering_signal).toFixed(3)
                + "\n"
                +
                "steering direction: "
                + state.steering_direction
                + "\n"
                +
                "steering hold: "
                + state.steering_hold_remaining_ms
                + " ms\n\n"
                +
                "x: "
                + Number(state.x).toFixed(3)
                + "\n"
                +
                "y: "
                + Number(state.y).toFixed(3)
                + "\n"
                +
                "vx: "
                + Number(state.vx).toFixed(4)
                + "\n"
                +
                "vy: "
                + Number(state.vy).toFixed(4)
                + "\n\n"
                +
                "spikes: "
                + state.total_spikes
                + "\n"
                +
                "DN: "
                + state.descending_spikes
                + "\n"
                +
                "motor: "
                + state.motor_spikes
                + "\n"
                +
                "flight motor: "
                + state.flight
                + "\n"
                +
                "jump: "
                + state.jump
                + "\n"
                +
                "other: "
                + state.other
                + "\n\n"
                +
                "internal active: "
                + state.endogenous_active
                + "\n"
                +
                "internal bursts: "
                + state.endogenous_bursts
                + "\n"
                +
                "next internal: "
                + state.next_endogenous_ms
                + " ms\n\n"
                +
                "calc: "
                + state.compute_ms
                + " ms\n"
                +
                "max calc: "
                + state.max_compute_ms
                + " ms\n"
                +
                "backlog: "
                + state.backlog_ms
                + " ms\n"
                +
                "runaway: "
                + state.runaway
                + "\n\n"
                + "behavior: "
                + String(state.behavior_active)
                + " "
                + String(state.behavior_id)
                + " / "
                + String(state.behavior_phase)
                + "\n"
                + "render: "
                + Number(state.x).toFixed(3)
                + ","
                + Number(state.y).toFixed(3)
                + "\n"
                + "neural: "
                + Number(state.neural_body_x).toFixed(3)
                + ","
                + Number(state.neural_body_y).toFixed(3)
                + " "
                + String(state.neural_body_state || "")
                + "\n"
                + "chair: "
                + String(state.chair_visible)
                + "\n"
                + "world: "
                + String(state.world_collision_ready)
                + " colliders="
                + String(state.world_collider_count)
                + "\n"
                + "support: "
                + String(state.support_surface_name || "")
                + " contact="
                + String(state.collision_contact_side || "")
                + "\n"
                + "turn: "
                + String(state.collision_turn_direction || "NONE")
                + " "
                + String(state.collision_turn_remaining_ms || 0)
                + "ms"
                + "\n"
                + "hitbox: "
                + Number(state.fly_collider_left || 0).toFixed(3)
                + ".."
                + Number(state.fly_collider_right || 0).toFixed(3);

            panel.textContent +=
                "\n\ncharacter asset loaded: " + characterStatus.loaded
                + "\ncharacter renderer ready: " + characterStatus.ready
                + "\nrender animation: " + characterStatus.animation
                + "\nanimation loop: " + characterStatus.loop
                + "\ncurrent loop: "
                + String(state.render_loop_animation || 'FLY_IDLE')
                + "\nanimation seq: "
                + String(state.render_animation_seq || 0)
                + "\ncharacter render scale: " + CHARACTER_RENDER_SCALE
                + "\ncharacter error: " + characterStatus.error
                + "\ncharacter warning: " + characterStatus.warning;
        }
    }
    catch (error) {
    }
    finally {
        // One request at a time prevents out-of-order event snapshots.
        setTimeout(updateState, 40);
    }
}

updateState();

function render(timestamp) {
    const delta = lastFrameTime === null
        ? 0
        : (timestamp - lastFrameTime) / 1000;

    lastFrameTime = timestamp;
    renderCharacter(delta);

    const width = window.innerWidth;
    const height = window.innerHeight;

    const x = Number(state.x) * width;
    const y = Number(state.y) * height;

    if (chair) {
        chair.style.display = state.chair_visible ? "block" : "none";
        chair.style.left = (Number(state.chair_x) * width).toFixed(1) + "px";
        chair.style.top = (Number(state.chair_y) * height).toFixed(1) + "px";
        chair.style.width = (Number(state.chair_width) * width).toFixed(1) + "px";
        chair.style.height = (Number(state.chair_height) * height).toFixed(1) + "px";
    }

    const facing = Number(state.facing) >= 0 ? 1 : -1;
    const bodyState = state.body_state || "IDLE";

    fly.className = bodyState;

    fly.style.transform =
        "translate("
        +
        (x - 38).toFixed(1)
        +
        "px,"
        +
        (y - 55).toFixed(1)
        +
        "px) "
        +
        "scaleX("
        +
        facing
        +
        ")";

    requestAnimationFrame(render);
}

requestAnimationFrame(render);
</script>

</body>
</html>
"""


class OverlayHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def send_bytes(self, data, content_type, status=200):
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
        ):
            return

    def send_json(self, obj):
        data = json.dumps(obj).encode('utf-8')
        self.send_bytes(
            data,
            'application/json; charset=utf-8',
        )

    def set_visual_command(self, command_name, event_name):
        with state_lock:
            commands[command_name] = max(
                commands[command_name],
                LOOMING_DURATION_MS,
            )

        self.send_json({
            'ok': True,
            'event': event_name,
            'duration_ms': LOOMING_DURATION_MS,
            'rate_hz': LOOMING_RATE_HZ,
        })

    def _watching_start_response(self):
        if not os.getenv('OBS_WEBSOCKET_PASSWORD'):
            self.send_json({
                'ok': False,
                'error': 'OBS_WEBSOCKET_PASSWORD_NOT_SET',
            })
            return

        with behavior_lock:
            if watching_state['active']:
                self.send_json({
                    'ok': False,
                    'reason': 'behavior_already_active',
                })
                return

        try:
            from world.obs_habitat import ObsHabitat
            from world.prop_placement import (
                PropFootprint,
                PropPlacementEngine,
                VerticalPreference,
            )

            habitat = ObsHabitat(
                os.getenv('OBS_WEBSOCKET_HOST', '127.0.0.1'),
                int(os.getenv('OBS_WEBSOCKET_PORT', '7654')),
                os.getenv('OBS_WEBSOCKET_PASSWORD'),
            )
            habitat.connect()

            try:
                prop = PropFootprint(
                    'WATCHING_CHAIR',
                    WATCHING_PROP_WIDTH,
                    WATCHING_PROP_HEIGHT,
                    WATCHING_PROP_SAFETY_MARGIN,
                    ('SCREEN_FLOOR', 'OVERLAY_TOP'),
                    VerticalPreference.LOW,
                    True,
                    2.0,
                )
                candidates = (
                    PropPlacementEngine(habitat).find_candidates(prop)
                )
            finally:
                habitat.disconnect()

            if not candidates:
                self.send_json({
                    'ok': False,
                    'error': 'NO_VALID_WATCHING_PLACEMENT',
                })
                return

            placement = _choose_watching_candidate(candidates)
            _start_watching(placement)

            self.send_json({
                'ok': True,
                'behavior': 'WATCHING',
                'source': 'AUTONOMOUS',
                'priority': BEHAVIOR_PRIORITY_AUTONOMOUS,
                'phase': 'WATCHING_ENTER',
                'surface': placement.surface_name,
                'surface_type': placement.surface_type,
                'chair_x': placement.x,
                'chair_y': placement.bottom,
                'target_x': placement.x,
                'target_y': max(
                    0.05,
                    min(
                        GROUND_Y,
                        placement.bottom
                        - placement.height * WATCHING_SEAT_HEIGHT_RATIO,
                    ),
                ),
            })

        except Exception as exc:
            # Do not send exception details or credentials to the browser.
            self.send_json({
                'ok': False,
                'error': type(exc).__name__,
            })

    def _watching_stop_response(self):
        accepted = _request_stop_watching()
        self.send_json({
            'ok': accepted,
            'phase': 'WATCHING_EXIT' if accepted else 'NORMAL',
        })

    def do_GET(self):
        path = urlparse(self.path).path

        if path in {
            '/assets/fly/fly_master.glb',
            '/assets/vendor/fly-renderer.bundle.js',
        }:
            is_glb = path == '/assets/fly/fly_master.glb'
            asset = (
                _character_asset_path()
                if is_glb
                else CHARACTER_RUNTIME_PATH
            )

            if asset is None or not _allowed_local_file(asset):
                message = (
                    b'GLB asset missing'
                    if is_glb
                    else b'Local renderer bundle missing'
                )
                self.send_bytes(
                    message,
                    'text/plain; charset=utf-8',
                    status=404,
                )
                return

            try:
                data = asset.read_bytes()
            except OSError:
                self.send_bytes(
                    b'Character asset unavailable',
                    'text/plain; charset=utf-8',
                    status=503,
                )
                return

            mime = (
                'model/gltf-binary'
                if is_glb
                else 'application/javascript; charset=utf-8'
            )
            self.send_bytes(data, mime)
            return

        if path == '/watching/start':
            self._watching_start_response()
            return

        if path == '/watching/stop':
            self._watching_stop_response()
            return

        if path == '/state':
            with state_lock:
                snapshot = dict(shared_state)
            self.send_json(snapshot)
            return

        if path == '/trigger_left':
            self.set_visual_command(
                'looming_left_ms',
                'looming_anatomical_left',
            )
            return

        if path == '/trigger_right':
            self.set_visual_command(
                'looming_right_ms',
                'looming_anatomical_right',
            )
            return

        if path == '/trigger_center':
            self.set_visual_command(
                'looming_center_ms',
                'looming_anatomical_center',
            )
            return

        if path == '/trigger':
            self.set_visual_command(
                'looming_center_ms',
                'looming_anatomical_center_legacy',
            )
            return

        if path == '/internal':
            with state_lock:
                commands['endogenous_ms'] = max(
                    commands['endogenous_ms'],
                    ENDOGENOUS_DURATION_MS,
                )

            self.send_json({
                'ok': True,
                'event': 'endogenous',
            })
            return

        if path == '/':
            self.send_bytes(
                HTML.encode('utf-8'),
                'text/html; charset=utf-8',
            )
            return

        self.send_bytes(
            b'Not found',
            'text/plain; charset=utf-8',
            status=404,
        )


brain_thread = threading.Thread(
    target=brain_loop,
    daemon=True,
)
brain_thread.start()

server = ThreadingHTTPServer(
    (HOST, PORT),
    OverlayHandler,
)

print()
print('=' * 80)
print('SERVER V6 HAZIR')
print('=' * 80)
print()

print('OBS:')
print(f'http://{HOST}:{PORT}/')
print()

print('DEBUG:')
print(f'http://{HOST}:{PORT}/?debug=1')
print()

print('ANATOMICAL VISUAL TEST ROUTES:')
print(f'http://{HOST}:{PORT}/trigger_left')
print(f'http://{HOST}:{PORT}/trigger_center')
print(f'http://{HOST}:{PORT}/trigger_right')
print()

print('Legacy /trigger -> anatomical CENTER/bilateral')
print()

print('Flight controller:', FLIGHT_CONTROL_MS, 'ms')
print('Flight target Y:', FLIGHT_TARGET_Y)
print()

print('V4 body/flight physics preserved.')
print('Screen LEFT/RIGHT is NOT mapped to anatomy yet.')
print('=' * 80)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print('\nServer kapatılıyor...')
finally:
    world_collision_stop.set()
    server.server_close()
