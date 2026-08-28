"""
predefined_opponent.py
======================
Simplified opponent client for RCSSServerMJ evaluation experiments.

Two behaviour modes (--mode flag):
  passive    : robots beam to starting position and stand still
  aggressive : all robots chase the ball directly (no role assignment)

Uses the same locomotion neural network as nn_client.py for realistic
movement quality. Only the high-level goal velocity decision differs.

Usage (single robot):
  python predefined_opponent.py --team BlueTeam --robot T1 --player_no 1 --mode passive
  python predefined_opponent.py --team BlueTeam --robot T1 --player_no 1 --mode aggressive

Use start_opponent.py to launch a full team.
"""

import argparse
import csv
import logging
import os
import re
import signal
import socket
import struct
import threading
import time
from datetime import datetime
from typing import ClassVar, Mapping

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from torch_policy import load_policy_from_files

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Field constants
# ---------------------------------------------------------------------------

# Opponent starts on the LEFT side (negative x) — mirror of RedTeam positions.
OPPONENT_BEAM_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
    1:  (-27.5,   0.0, 180),
    2:  (-22.0,  12.0, 180),
    3:  (-22.0,   4.0, 180),
    4:  (-22.0,  -4.0, 180),
    5:  (-22.0, -12.0, 180),
    6:  (-15.0,   0.0, 180),
    7:  ( -4.0,  16.0, 180),
    8:  (-11.0,   6.0, 180),
    9:  (-11.0,  -6.0, 180),
    10: ( -4.0, -16.0, 180),
    11: ( -7.0,   0.0, 180),
}

ROBOT_MOTORS: ClassVar[Mapping[str, tuple[str, ...]]] = {
    'ant': ('l4e1', 'l4e2', 'l1e1', 'l1e2', 'l2e1', 'l2e2', 'l3e1', 'l3e2'),
    'T1':  ('he1', 'he2',
            'lae1', 'lae2', 'lae3', 'lae4',
            'rae1', 'rae2', 'rae3', 'rae4',
            'te1',
            'lle1', 'lle2', 'lle3', 'lle4', 'lle5', 'lle6',
            'rle1', 'rle2', 'rle3', 'rle4', 'rle5', 'rle6'),
}

# Nominal (rest) joint positions for T1 in RADIANS — must match nn_client.py exactly
T1_NOMINAL_RAD = np.array([
    0.0, 0.0,                          # head:      he1, he2
    0.0, -1.4, 0.0, -0.4,             # left arm:  lae1-4
    0.0,  1.4, 0.0,  0.4,             # right arm: rae1-4
    0.0,                               # torso:     te1
   -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,   # left leg:  lle1-6
   -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,   # right leg: rle1-6
], dtype=np.float32)

NN_SCALING_FACTOR = 0.5  # matches nn_client.py scaling_factor


# ---------------------------------------------------------------------------

class OpponentClient:
    """Minimalist opponent agent — locomotion NN + simple decision logic."""

    def __init__(self, host: str, port: int, team: str, player_no: int,
                 robot: str, mode: str, csv_dir: str = 'CSV'):
        self.host = host
        self.port = port
        self.team = team
        self.player_no = player_no
        self.robot = robot
        self.mode = mode          # 'passive' or 'aggressive'
        self.csv_dir = csv_dir

        # Network
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # NN settings
        self._policy_checkpoint = 'locomotion_nn.pth'
        self._policy_meta = 'locomotion_nn_meta.json'
        self._gait_period = 1.0
        self._policy_dt = 0.02      # 50 Hz

        # Runtime state
        self._previous_action = None
        self._policy_hidden = None
        self._gait_phase = None
        self._wait_cycles = 0
        self._formation_reached = False
        self._play_started = False

        # Shutdown flag
        self._running = True
        signal.signal(signal.SIGINT, self._handle_sigint)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self):
        try:
            self._sock.connect((self.host, self.port))
            log.info(f'Connected to {self.host}:{self.port} | team={self.team} '
                     f'player={self.player_no} mode={self.mode}')
        except ConnectionRefusedError:
            log.error('Connection refused — is the server running?')
            return
        t = threading.Thread(target=self._action_loop, daemon=True)
        t.start()
        t.join()
        self._sock.close()

    def _handle_sigint(self, *_):
        self._running = False

    # ------------------------------------------------------------------
    # Policy runtime
    # ------------------------------------------------------------------

    def _reset_policy_state(self, n_joints: int):
        self._previous_action = np.zeros(n_joints, dtype=np.float32)
        # Initialise GRU carry as a proper zero tensor (not None)
        if hasattr(self, '_policy') and self._policy is not None:
            device = getattr(self, '_device', torch.device('cpu'))
            self._policy_hidden = self._policy.initialize_carry(1, device)
        else:
            self._policy_hidden = None
        self._gait_phase = np.array([0.0, -np.pi], dtype=np.float32)
        self._gait_phase_dt = 2.0 * np.pi * self._policy_dt / self._gait_period
        self._wait_cycles = 50

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _step_gait(self):
        self._gait_phase_dt = 2.0 * np.pi * self._policy_dt / self._gait_period
        self._gait_phase = self._wrap_to_pi(
            self._gait_phase + self._gait_phase_dt).astype(np.float32)

    def _gait_features(self) -> np.ndarray:
        # Use next-step phase (matches nn_client.py _get_gait_phase_features)
        phase_tp1 = self._wrap_to_pi(self._gait_phase + self._gait_phase_dt)
        return np.concatenate([np.sin(phase_tp1),
                               np.cos(phase_tp1)]).astype(np.float32)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _action_loop(self):
        motors = ROBOT_MOTORS[self.robot]
        n_joints = len(motors)
        nominal_rad = T1_NOMINAL_RAD[:n_joints] if self.robot == 'T1' else np.zeros(n_joints)

        device = torch.device('cpu')
        self._device = device
        policy, meta = load_policy_from_files(self._policy_checkpoint,
                                              self._policy_meta, device)
        self._policy = policy
        log.info('Locomotion policy loaded.')

        # Init message
        init_msg = f'(init {self.robot} {self.team} {self.player_no})'
        self._send_message(init_msg)

        # First perception → beam
        raw = self._receive_message()
        bp = OPPONENT_BEAM_POSES.get(self.player_no, (-10.0, 0.0, 180))
        beam_msg = f'(beam {bp[0]} {bp[1]} {bp[2]})'
        self._send_message(beam_msg)
        self._reset_policy_state(n_joints)
        # Ensure carry is a proper tensor now that policy is available
        self._policy_hidden = policy.initialize_carry(1, device)

        # CSV setup
        os.makedirs(self.csv_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(
            self.csv_dir,
            f'opponent_{self.mode}_{self.team}_p{self.player_no}_{ts}.csv'
        )
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow([
            'game_time', 'play_mode', 'player_no', 'team', 'mode',
            'robot_world_x', 'robot_world_y',
            'ball_visible', 'ball_rel_x', 'ball_rel_y',
            'goal_vel_x', 'goal_vel_yaw', 'event'
        ])

        game_time = 0.0
        log.info(f'Starting loop | mode={self.mode}')

        while self._running:
            try:
                raw = self._receive_message()
            except (ConnectionResetError, OSError):
                log.warning('Server disconnected.')
                break

            sensors = parse_sensor_string(raw)

            # ── Parse joints ──────────────────────────────────────────
            hj_list = sensors.get('HJ', [])
            if not isinstance(hj_list, list):
                hj_list = [hj_list]
            joint_pos = np.array([h.get('ax', 0.0) for h in hj_list],
                                 dtype=np.float32)[:n_joints]
            joint_vel = np.array([h.get('vx', 0.0) for h in hj_list],
                                 dtype=np.float32)[:n_joints]
            joint_pos_rad = np.deg2rad(joint_pos)
            joint_vel_rad = np.deg2rad(joint_vel)

            # ── Parse IMU / orientation ───────────────────────────────
            gyr = sensors.get('GYR', {})
            ang_vel_raw = gyr.get('rt', [0.0, 0.0, 0.0])
            ang_vel = np.array(ang_vel_raw[:3], dtype=np.float32)
            ang_vel_rad = np.deg2rad(ang_vel)

            quat_raw = sensors.get('quat', {})
            q_vals = quat_raw.get('q', [1.0, 0.0, 0.0, 0.0])
            if len(q_vals) >= 4:
                q = np.array(q_vals[:4], dtype=np.float32)
                rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                projected_gravity = rot.inv().apply(
                    np.array([0.0, 0.0, -1.0])).astype(np.float32)
            else:
                projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

            # ── Parse game state ─────────────────────────────────────
            game_state = sensors.get('GS', {})
            play_mode = game_state.get('pm', 'Unknown')
            game_time_raw = game_state.get('t', game_time)
            try:
                game_time = float(game_time_raw)
            except (ValueError, TypeError):
                pass

            if 'KickOff' in play_mode or 'PlayOn' in play_mode:
                self._play_started = True

            # ── Parse ball ───────────────────────────────────────────
            vision = sensors.get('See', {})
            ball_data = vision.get('B', {})
            ball_pol = ball_data.get('pol', []) if isinstance(ball_data, dict) else []
            ball_visible = len(ball_pol) >= 2
            ball_dist = float(ball_pol[0]) if ball_visible else 0.0
            ball_az_deg = float(ball_pol[1]) if ball_visible else 0.0
            ball_az_rad = np.deg2rad(ball_az_deg)
            ball_rel_x = ball_dist * np.cos(ball_az_rad)
            ball_rel_y = ball_dist * np.sin(ball_az_rad)

            # ── Parse robot world position ───────────────────────────
            pos_data = sensors.get('pos', {})
            robot_x = float(pos_data.get('x', bp[0]))
            robot_y = float(pos_data.get('y', bp[1]))

            # ── Motor command decision ────────────────────────────────
            if self.mode == 'passive':
                # Passive: hold nominal joint positions — no NN needed.
                # The walking NN cannot stand still; direct position hold is stable.
                goal_vel = np.zeros(3, dtype=np.float32)
                target_deg = np.rad2deg(nominal_rad)
                event = 'passive_hold' if self._play_started else 'passive_wait'

            else:
                # Aggressive: use the locomotion NN with a goal velocity.
                if self._wait_cycles > 0:
                    goal_vel = np.zeros(3, dtype=np.float32)
                    self._wait_cycles -= 1
                    event = 'stabilise'
                elif not self._play_started:
                    goal_vel = np.zeros(3, dtype=np.float32)
                    event = 'wait_kickoff'
                elif ball_visible:
                    yaw_vel = float(np.clip(ball_az_deg / 45.0, -1.0, 1.0))
                    goal_vel = np.array([1.0, 0.0, yaw_vel], dtype=np.float32)
                    event = 'chase_ball'
                else:
                    goal_vel = np.array([0.0, 0.0, 0.3], dtype=np.float32)
                    event = 'search_ball'

                # Build observation vector (order must match nn_client.py exactly)
                self._step_gait()
                scaled_jp = (joint_pos_rad - nominal_rad) / np.pi
                scaled_jv = joint_vel_rad / 100.0
                scaled_pa = self._previous_action / 10.0
                scaled_av = np.clip(ang_vel_rad / 50.0, -1.0, 1.0)
                gait_f = self._gait_features()
                obs = np.concatenate([
                    scaled_jp, scaled_jv, scaled_pa,
                    scaled_av, goal_vel, gait_f, projected_gravity
                ]).astype(np.float32)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                obs = np.clip(obs, -10.0, 10.0)

                obs_t = torch.from_numpy(obs).unsqueeze(0)
                with torch.no_grad():
                    act_t, self._policy_hidden = policy(obs_t, self._policy_hidden)
                action = act_t.squeeze(0).numpy().astype(np.float32)
                self._previous_action = action
                target_rad = nominal_rad + NN_SCALING_FACTOR * action
                target_deg = np.rad2deg(target_rad)
            cmd_parts = []
            for motor, t_deg in zip(motors, target_deg):
                cmd_parts.append(f'({motor} {t_deg:.4f} 0.0 25.00 0.60 0.0)')
            self._send_message(''.join(cmd_parts))

            # ── CSV log (sample every 5 cycles) ──────────────────────
            if int(game_time * 50) % 5 == 0 or event not in ('passive_hold', 'chase_ball'):
                writer.writerow([
                    f'{game_time:.2f}', play_mode, self.player_no, self.team,
                    self.mode, f'{robot_x:.3f}', f'{robot_y:.3f}',
                    int(ball_visible), f'{ball_rel_x:.3f}', f'{ball_rel_y:.3f}',
                    f'{goal_vel[0]:.2f}', f'{goal_vel[2]:.2f}', event
                ])

        csv_file.close()
        log.info(f'CSV saved → {csv_path}')

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    def _send_message(self, msg: str):
        data = msg.encode('utf-8')
        header = struct.pack('>I', len(data))
        self._sock.sendall(header + data)

    def _receive_message(self) -> str:
        header = self._recv_exact(4)
        length = struct.unpack('>I', header)[0]
        buf = bytearray(max(length, 4096))
        mv = memoryview(buf)
        received = 0
        while received < length:
            chunk = self._sock.recv_into(mv[received:], length - received)
            if chunk == 0:
                raise ConnectionResetError('Server closed connection.')
            received += chunk
        return buf[:length].decode('utf-8', errors='replace')

    def _recv_exact(self, n: int) -> bytes:
        data = b''
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionResetError('Server closed connection.')
            data += chunk
        return data


# ---------------------------------------------------------------------------
# S-expression parser (same as nn_client.py)
# ---------------------------------------------------------------------------

def parse_sensor_string(msg: str) -> dict:
    result = {}
    top_pattern = re.compile(r'\((\w+)\s+(.*?)\)(?=\s*\(|\s*$)', re.DOTALL)
    kv_pattern = re.compile(r'\((\w+)\s+([^()]+)\)')

    for top_match in top_pattern.finditer(msg):
        tag = top_match.group(1)
        body = top_match.group(2).strip()
        entry = {}
        for kv in kv_pattern.finditer(body):
            key = kv.group(1)
            vals = kv.group(2).split()
            parsed = []
            for v in vals:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
            entry[key] = parsed[0] if len(parsed) == 1 else parsed

        if tag in result:
            existing = result[tag]
            if not isinstance(existing, list):
                result[tag] = [existing]
            result[tag].append(entry)
        else:
            result[tag] = entry

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Predefined opponent client')
    parser.add_argument('-s', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=60000)
    parser.add_argument('-t', '--team', default='BlueTeam')
    parser.add_argument('-n', '--player_no', type=int, default=1)
    parser.add_argument('-r', '--robot', default='T1',
                        choices=['ant', 'T1'])
    parser.add_argument('--mode', default='passive',
                        choices=['passive', 'aggressive'],
                        help='passive = stand still; aggressive = all chase ball')
    parser.add_argument('--csv-dir', default='CSV')
    args = parser.parse_args()

    client = OpponentClient(
        host=args.host,
        port=args.port,
        team=args.team,
        player_no=args.player_no,
        robot=args.robot,
        mode=args.mode,
        csv_dir=args.csv_dir,
    )
    client.run()


if __name__ == '__main__':
    main()
