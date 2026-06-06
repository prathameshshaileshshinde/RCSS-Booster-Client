import argparse
import csv
import datetime
import logging
import os
import signal
import socket
import threading
from collections.abc import Mapping
from types import FrameType
from typing import ClassVar

import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import torch

from torch_policy import load_policy_from_files

# ============================================================================
#  CONFIG  -  beginner-friendly knobs. Change these, re-run, watch what happens.
# ============================================================================

# Vision / head tracking ------------------------------------------------------
ENABLE_HEAD_TRACKING   = True    # keep the head pointed at the ball
HEAD_YAW_KP            = 0.6    # how hard the head turns left/right toward the ball
HEAD_PITCH_KP          = 0.6    # how hard the head tilts up/down toward the ball
HEAD_YAW_SIGN          = +1     # if the head turns AWAY from the ball, change to -1
HEAD_PITCH_SIGN        = -1     # if the head tilts the WRONG way, change to +1
HEAD_YAW_LIMIT_DEG     = (-90.0, 90.0)   # he1 joint limits
HEAD_PITCH_LIMIT_DEG   = (-20.0, 70.0)   # he2 joint limits
# Head sweep speed while searching (degrees per cycle):
HEAD_SWEEP_DEG_PER_CYCLE = 4.0

# Ball following (body movement) ----------------------------------------------
ENABLE_BALL_FOLLOWING  = True    # walk toward the ball
FOLLOW_FORWARD_SPEED   = 1.0    # forward goal velocity when chasing the ball
# STEER_KP converts degrees of body-relative angle → yaw goal velocity.
# body_relative_angle = head_yaw_deg + ball_azimuth_deg
# e.g. 45° off → yaw_vel = 45 * (1/45) = 1.0 (full spin)
STEER_KP               = 1.0 / 45.0

# Ball persistence ------------------------------------------------------------
# The vision perceptor fires every 2 cycles (25 Hz). These settings let the
# robot keep walking toward the ball even when vision temporarily loses it.
BALL_CLOSE_DIST_M      = 2.0    # metres: closer than this → start the timer
BALL_CLOSE_TIMER_CYCLES = 150   # cycles (~3 s) to remember the ball under foot
BALL_SMOOTH_ALPHA      = 0.25   # exponential smoothing weight (0=frozen, 1=raw)
# Ball-reset detection: if the ball jumps more than this, clear all state.
BALL_RESET_NEW_DIST_M  = 5.0
BALL_RESET_OLD_DIST_M  = 3.0

# Search behaviour (when the ball is completely lost) -------------------------
ENABLE_SEARCH      = True    # spin the whole body to find the ball
LOST_BALL_CYCLES   = 30     # how many cycles before declaring ball "lost"
SEARCH_YAW_SPEED   = 0.6   # body spin speed while searching

# CSV logging -----------------------------------------------------------------
ENABLE_CSV_LOGGING  = True   # write position data to a CSV file
CSV_EVERY_N_CYCLES  = 5     # one row every N cycles (keeps the file small)
CSV_DIR             = "CSV"  # folder where CSV files are saved (auto-created)

# ============================================================================


# ---------- LOGGING CONFIG ----------
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
ch.setLevel(logging.INFO)
logging.basicConfig(handlers=[ch], level=logging.DEBUG)
# ---------- LOGGING CONFIG ----------

logger = logging.getLogger(__name__)


class Client:
    """Neural-network locomotion client with robust ball tracking.

    Ball tracking works in three layers (checked in order each cycle):
      1. World model   — converts ball position to world coords when visible;
                         projects back when the camera loses the ball.
      2. Close timer   — remembers the last (x,y,z) for BALL_CLOSE_TIMER_CYCLES
                         cycles after the ball is within BALL_CLOSE_DIST_M.
      3. Search        — if all else fails, spin to sweep the camera over the field.
    """

    BEAM_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
        1: (27.5, 0.0, 0),
        2: (22.0, 12.0, 0),
        3: (22.0, 4.0, 0),
        4: (22.0, -4.0, 0),
        5: (22.0, -12.0, 0),
        6: (15.0, 0.0, 0),
        7: (4.0, 16.0, 0),
        8: (11.0, 6.0, 0),
        9: (11.0, -6.0, 0),
        10: (4.0, -16.0, 0),
        11: (7.0, 0.0, 0),
    }

    ROBOT_MOTORS: ClassVar[Mapping[str, tuple[str, ...]]] = {
        'ant': ('l4e1', 'l4e2', 'l1e1', 'l1e2', 'l2e1', 'l2e2', 'l3e1', 'l3e2'),
        # For T1: index 0 = he1 (head yaw), index 1 = he2 (head pitch).
        'T1': ('he1', 'he2', 'lae1', 'lae2', 'lae3', 'lae4', 'rae1', 'rae2', 'rae3', 'rae4',
               'te1', 'lle1', 'lle2', 'lle3', 'lle4', 'lle5', 'lle6',
               'rle1', 'rle2', 'rle3', 'rle4', 'rle5', 'rle6'),
    }

    HEAD_YAW_IDX   = 0  # he1 → AAHead_yaw
    HEAD_PITCH_IDX = 1  # he2 → Head_pitch

    def __init__(self, host: str, port: int, team: str, player_no: int, model_name: str | None = None):
        self._host: str = host
        self._port: int = port
        self._model_name: str = 'ant' if model_name is None else model_name
        self._team: str = team
        self._player_no: int = player_no

        self._policy_checkpoint = "locomotion_nn.pth"
        self._policy_meta       = "locomotion_nn_meta.json"
        self._gait_period       = 1.0
        self._policy_dt         = 0.02

        self._rcv_buffer_size = 1024
        self._rcv_buffer      = bytearray(self._rcv_buffer_size)
        self._sock            = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._has_beamed: bool = False

        # ── head tracking state ───────────────────────────────────────────────
        self._head_yaw_target   = 0.0    # commanded he1 angle (degrees)
        self._head_pitch_target = 0.0   # commanded he2 angle (degrees)
        self._head_scan_angle   = 0.0   # sweep position while searching (degrees)
        self._search_dir        = 1.0   # +1 = sweep left, −1 = sweep right

        # ── ball tracking state (Cartesian metres, robot frame) ───────────────
        self._last_ball_x: float | None      = None  # most recent seen position
        self._last_ball_y: float | None      = None
        self._last_ball_z: float | None      = None
        self._smooth_ball_x: float | None    = None  # exponentially smoothed
        self._smooth_ball_y: float | None    = None
        self._ball_world_pos: np.ndarray | None = None  # ball in world coordinates
        self._ball_close_timer: int          = 0    # counts down from BALL_CLOSE_TIMER_CYCLES
        self._cycles_since_ball: int         = 999  # for LOST_BALL_CYCLES threshold

        # ── goal position (world coords, set once it's seen) ──────────────────
        self._goal_world_pos: np.ndarray | None = None

        # ── misc ──────────────────────────────────────────────────────────────
        self._cycle      = 0
        self._log_cycle  = 0
        self._csv_writer = None
        self._csv_file   = None

        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # ═══════════════════════════════════════════════ connection / shutdown
    def run(self):
        logger.info('Connecting to server at %s:%d...', self._host, self._port)
        try:
            self._sock.connect((self._host, self._port))
        except ConnectionRefusedError:
            logger.error('Connection refused. Is the server running?')  # noqa: TRY400
            return

        client_thread = threading.Thread(target=self._action_loop)
        client_thread.start()
        client_thread.join()

        if self._csv_file is not None:
            self._csv_file.close()
        self._sock.close()

    def shutdown(self) -> None:
        self._sock.shutdown(socket.SHUT_RDWR)

    # ═══════════════════════════════════════════════ policy / gait helpers
    def _init_policy_runtime_state(self):
        self.previous_action  = np.zeros(self.nr_joints, dtype=np.float32)
        self.policy_hidden    = self.policy.initialize_carry(batch_size=1, device=self.device)
        self.wait_until_walking = 50

        self.gait_phase_offset = np.array([0.0, -np.pi], dtype=np.float32)
        self.gait_phase        = self.gait_phase_offset.copy()
        self.gait_mean_freq    = 1.0 / self._gait_period
        self.gait_freq         = self.gait_mean_freq
        self.gait_phase_dt     = (2.0 * np.pi * self._policy_dt) * self.gait_freq

        # Reset all ball tracking state on (re)spawn so stale data never carries over.
        self._last_ball_x      = None
        self._last_ball_y      = None
        self._last_ball_z      = None
        self._smooth_ball_x    = None
        self._smooth_ball_y    = None
        self._ball_world_pos   = None
        self._ball_close_timer = 0
        self._cycles_since_ball = 999
        self._head_scan_angle  = 0.0
        self._goal_world_pos   = None

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _get_gait_phase_features(self) -> np.ndarray:
        phase_tp1 = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt)
        return np.concatenate([np.sin(phase_tp1), np.cos(phase_tp1)], axis=-1).astype(np.float32)

    def _step_gait_manager(self):
        self.gait_phase = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt).astype(np.float32)

    # ═══════════════════════════════════════════════ vision / parsing helpers
    @staticmethod
    def _polar_to_cartesian(dist: float, az_deg: float, el_deg: float) -> tuple[float, float, float]:
        """Convert polar (dist, azimuth°, elevation°) → Cartesian (x, y, z) metres.

        x = forward, y = sideways (+ = left), z = height (− = below camera).
        All trig is done in radians internally.
        """
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        x = dist * np.cos(el) * np.cos(az)
        y = dist * np.cos(el) * np.sin(az)
        z = dist * np.sin(el)
        return float(x), float(y), float(z)

    @staticmethod
    def _extract_balanced(s: str, start: int) -> str:
        """Return the substring from '(' at index `start` to its matching ')'."""
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return s[start:]

    def _parse_ball(self, msg: str):
        """Return (dist, az_deg, el_deg) for the ball, or None if not visible.

        Server format:  (B (pol <distance> <azimuth> <elevation>))
        Angles are relative to the camera/head, in degrees.
        """
        m = re.search(r'\(B\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)\)', msg)
        if not m:
            return None
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    def _parse_goal(self, msg: str):
        """Return (goal_x, goal_y) metres (robot-relative) for the opponent goal,
        averaging the two goalposts if both are visible. Returns (None, None) otherwise."""
        m1 = re.search(r'\(G1R\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\)\)', msg)
        m2 = re.search(r'\(G2R\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\)\)', msg)
        xs, ys = [], []
        for m in (m1, m2):
            if m:
                d  = float(m.group(1))
                az = np.deg2rad(float(m.group(2)))
                xs.append(d * np.cos(az))
                ys.append(d * np.sin(az))
        if not xs:
            return None, None
        return float(np.mean(xs)), float(np.mean(ys))

    def _parse_players(self, msg: str):
        """Return a list of dicts describing every visible player.

        Each entry: {team, id, distance, azimuth, elevation}
        Polar values are the mean over the player's visible body markers.
        """
        players = []
        for m in re.finditer(r'\(P\s+\(team\s+(\S+)\)\s+\(id\s+(\d+)\)', msg):
            team  = m.group(1)
            pid   = int(m.group(2))
            block = self._extract_balanced(msg, m.start())
            pols  = re.findall(r'\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)', block)
            if not pols:
                continue
            arr = np.array(pols, dtype=np.float32)
            d, az, el = arr.mean(axis=0)
            players.append({'team': team, 'id': pid,
                            'distance': float(d), 'azimuth': float(az), 'elevation': float(el)})
        return players

    # ═══════════════════════════════════════════════ head tracking
    def _track_head(self, ball, cur_yaw_deg: float, cur_pitch_deg: float):
        """Update head yaw/pitch targets to keep the camera on the ball.

        When the ball IS visible: proportional controller — nudge toward it.
        When the ball is NOT visible: sweep the head left/right (search pattern).

        Args:
            ball: (dist, az_deg, el_deg) from _parse_ball(), or None.
            cur_yaw_deg:   current he1 angle in degrees.
            cur_pitch_deg: current he2 angle in degrees.
        """
        if ball is not None:
            _, azimuth, elevation = ball
            # Azimuth/elevation are the error relative to the current head direction.
            # Add a fraction of that error to the current target to drive error to zero.
            self._head_yaw_target   = cur_yaw_deg   + HEAD_YAW_SIGN   * HEAD_YAW_KP   * azimuth
            self._head_pitch_target = cur_pitch_deg + HEAD_PITCH_SIGN * HEAD_PITCH_KP * elevation
            # Reset the scan position so the next search starts from where the ball was.
            self._head_scan_angle   = self._head_yaw_target
        else:
            # Ball not visible — sweep the head left/right while looking slightly down.
            if ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
                self._head_scan_angle += self._search_dir * HEAD_SWEEP_DEG_PER_CYCLE
                if self._head_scan_angle > 80.0:
                    self._search_dir = -1.0
                elif self._head_scan_angle < -80.0:
                    self._search_dir = 1.0
                self._head_yaw_target   = self._head_scan_angle
                self._head_pitch_target = -40.0  # look down while scanning
            # Otherwise keep the last commanded target (don't snap back to centre).

        # Clamp to hardware joint limits.
        self._head_yaw_target   = float(np.clip(self._head_yaw_target,   *HEAD_YAW_LIMIT_DEG))
        self._head_pitch_target = float(np.clip(self._head_pitch_target, *HEAD_PITCH_LIMIT_DEG))

    # ═══════════════════════════════════════════════ body movement
    def _search_for_ball(self) -> np.ndarray:
        """Spin the whole robot body to sweep the camera over the field.

        Turns toward the last known ball direction first, then keeps spinning
        until the ball is found again.

        Returns:
            goal_vel: [forward, sideways, yaw] array for the policy.
        """
        if self._last_ball_x is not None:
            steer   = np.arctan2(self._last_ball_y, self._last_ball_x)
            yaw_vel = float(np.sign(steer)) if abs(steer) > 0.1 else float(self._search_dir)
        else:
            yaw_vel = float(self._search_dir) * SEARCH_YAW_SPEED
        return np.array([0.0, 0.0, yaw_vel], dtype=np.float32)

    def _turn_around(self) -> np.ndarray:
        """Spin ~180° in place, preferring the side the ball was last seen on.

        Use this when the robot needs to reverse direction quickly.

        Returns:
            goal_vel: [forward, sideways, yaw] array for the policy.
        """
        turn_dir = float(np.sign(self._last_ball_y)) if (self._last_ball_y is not None
                                                          and self._last_ball_y != 0.0) \
                   else float(self._search_dir)
        return np.array([0.0, 0.0, turn_dir], dtype=np.float32)

    def _decide_goal_vel(self, ball_raw, cur_head_yaw_deg: float) -> np.ndarray:
        """Compute body goal velocity to walk toward the ball.

        Strategy (from the original clean client):
          - The ball azimuth from the camera is the angle the ball is OFF from
            where the head points. Adding the current head yaw converts that into
            a body-relative angle.
          - We walk forward at FOLLOW_FORWARD_SPEED and steer proportionally.
          - If the ball is not visible and ENABLE_SEARCH is on, spin to search.

        Args:
            ball_raw:         (dist, az_deg, el_deg) from _parse_ball(), or None.
            cur_head_yaw_deg: current he1 angle in degrees.

        Returns:
            goal_vel: [forward, sideways, yaw] for the locomotion policy.
        """
        if not ENABLE_BALL_FOLLOWING:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if ball_raw is not None:
            _, azimuth, _ = ball_raw
            # Convert camera-relative azimuth → body-relative angle (degrees).
            # The head might be turned 20° left and the ball is 10° further left
            # in the camera → the body needs to turn 30° left total.
            body_angle = cur_head_yaw_deg + azimuth
            yaw_vel    = float(np.clip(body_angle * STEER_KP, -1.0, 1.0))
            return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)

        # Ball not visible this cycle.
        if ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            return np.array([0.0, 0.0, self._search_dir * SEARCH_YAW_SPEED], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ═══════════════════════════════════════════════ CSV logging
    def _open_csv(self):
        if not ENABLE_CSV_LOGGING:
            return
        os.makedirs(CSV_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(CSV_DIR, f'robot_log_{self._team}_p{self._player_no}_{ts}.csv')
        self._csv_file   = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'game_time', 'play_mode',
            'player_no', 'team',
            'robot_world_x', 'robot_world_y', 'robot_world_z',
            'ball_visible',
            'ball_rel_x', 'ball_rel_y', 'ball_rel_z',
            'ball_world_x', 'ball_world_y', 'ball_world_z',
            'goal_rel_x', 'goal_rel_y',
        ])
        logger.info('Logging to %s', path)

    def _log_csv(self, game_time, play_mode, robot_world_pos,
                 ball_visible, ball_x, ball_y, ball_z, goal_x, goal_y):
        if not ENABLE_CSV_LOGGING or self._csv_writer is None:
            return
        if self._cycle % CSV_EVERY_N_CYCLES != 0:
            return
        bwp = self._ball_world_pos
        rwp = robot_world_pos
        self._csv_writer.writerow([
            round(game_time, 3), play_mode,
            self._player_no, self._team,
            round(float(rwp[0]), 3) if rwp is not None else '',
            round(float(rwp[1]), 3) if rwp is not None else '',
            round(float(rwp[2]), 3) if rwp is not None else '',
            1 if ball_visible else 0,
            round(ball_x, 3) if ball_x is not None else '',
            round(ball_y, 3) if ball_y is not None else '',
            round(ball_z, 3) if ball_z is not None else '',
            round(float(bwp[0]), 3) if bwp is not None else '',
            round(float(bwp[1]), 3) if bwp is not None else '',
            round(float(bwp[2]), 3) if bwp is not None else '',
            round(goal_x, 3) if goal_x is not None else '',
            round(goal_y, 3) if goal_y is not None else '',
        ])

    # ═══════════════════════════════════════════════ main perception-action loop
    def _action_loop(self):  # noqa: PLR0912, PLR0915
        self.nr_joints = len(self.ROBOT_MOTORS[self._model_name])

        self.p_gain      = 25.0
        self.d_gain      = 0.6
        self.scaling_factor = 0.5
        self.joint_nominal_position = np.array([
            0.0, 0.0,
            0.0, -1.4, 0.0, -0.4,
            0.0,  1.4, 0.0,  0.4,
            0.0,
            -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
            -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
        ], dtype=np.float32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy, self.policy_meta = load_policy_from_files(
            self._policy_checkpoint, self._policy_meta, self.device,
        )
        self._init_policy_runtime_state()
        self._open_csv()

        logger.info('Initializing agent...')
        self._send_message(f'(init {self._model_name} {self._team} {self._player_no})'.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                # ── beam to starting position once ────────────────────────────
                if not self._has_beamed:
                    bp = self.BEAM_POSES[self._player_no]
                    self._send_message(f'(beam {bp[0]} {bp[1]} {bp[2]})'.encode())
                    self._has_beamed = True
                    self._init_policy_runtime_state()
                    continue

                self._cycle     += 1
                self._log_cycle += 1
                perception_msg_str = perception_msg.decode()
                perception_data    = self.parse_sensor_string(perception_msg_str)

                # ── joint sensors ─────────────────────────────────────────────
                joint_pos_degrees = np.array([h['ax'] for h in perception_data['HJ']], dtype=np.float32)
                joint_pos = np.deg2rad(joint_pos_degrees).astype(np.float32)
                joint_vel_degrees = np.array([h['vx'] for h in perception_data['HJ']], dtype=np.float32)
                joint_vel = np.deg2rad(joint_vel_degrees).astype(np.float32)

                cur_head_yaw_deg   = float(joint_pos_degrees[self.HEAD_YAW_IDX])
                cur_head_pitch_deg = float(joint_pos_degrees[self.HEAD_PITCH_IDX])

                scaled_joint_pos       = (joint_pos - self.joint_nominal_position) / 3.14
                scaled_joint_vel       = joint_vel / 100.0
                scaled_previous_action = self.previous_action / 10.0

                ang_vel = np.deg2rad(np.array(perception_data['GYR']['rt'], dtype=np.float32)).astype(np.float32)
                scaled_ang_vel = np.clip(ang_vel / 50.0, -1.0, 1.0)

                # ── orientation ───────────────────────────────────────────────
                q  = np.array(perception_data['quat']['q'], dtype=np.float32)
                robot_rotation    = R.from_quat([q[1], q[2], q[3], q[0]])
                orientation_inv   = robot_rotation.inv()
                projected_gravity = orientation_inv.apply(
                    np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

                # ── robot world position (torso_pos perceptor) ────────────────
                torso_match = re.search(
                    r'\(pos\s+\(n\s+torso_pos\)\s+\(pos\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)\)',
                    perception_msg_str
                )
                robot_world_pos = np.array([
                    float(torso_match.group(1)),
                    float(torso_match.group(2)),
                    float(torso_match.group(3)),
                ], dtype=np.float32) if torso_match else None

                # ── game state ────────────────────────────────────────────────
                gs_match  = re.search(r'\(GS\s+\(t\s+([\d.]+)\)\s+\(pm\s+(\w+)\)', perception_msg_str)
                game_time = float(gs_match.group(1)) if gs_match else 0.0
                play_mode = gs_match.group(2)         if gs_match else 'Unknown'

                # ── BALL: parse vision → Cartesian ────────────────────────────
                ball_raw = self._parse_ball(perception_msg_str)
                ball_visible = ball_raw is not None

                if ball_raw is not None:
                    ball_dist      = ball_raw[0]
                    ball_x, ball_y, ball_z = self._polar_to_cartesian(*ball_raw)
                    self._cycles_since_ball = 0
                else:
                    ball_dist = ball_x = ball_y = ball_z = None
                    self._cycles_since_ball += 1

                # ── GOAL: parse vision → robot-relative Cartesian ─────────────
                goal_x, goal_y = self._parse_goal(perception_msg_str)

                # Persist goal in world coords so it's available when out of view.
                if goal_x is not None and robot_world_pos is not None:
                    self._goal_world_pos = robot_world_pos + robot_rotation.apply(
                        np.array([goal_x, goal_y, 0.0], dtype=np.float32))
                if goal_x is None and self._goal_world_pos is not None and robot_world_pos is not None:
                    offset         = self._goal_world_pos - robot_world_pos
                    goal_local     = orientation_inv.apply(offset)
                    goal_x, goal_y = float(goal_local[0]), float(goal_local[1])

                # ── BALL PERSISTENCE: three-layer system ─────────────────────
                #
                # Layer 1 — world model:
                #   When the ball IS visible, convert to world coordinates.
                #   On cycles where vision returns nothing, project back into the
                #   robot's current frame using orientation_inv.
                #
                if ball_x is not None:
                    # Fresh vision reading — store last known + update world model.
                    self._last_ball_x = ball_x
                    self._last_ball_y = ball_y
                    self._last_ball_z = ball_z

                    # Vision fires every 2 cycles; ball under foot disappears each
                    # other cycle. Large timer (150 cycles) keeps the robot walking.
                    if ball_dist is not None and ball_dist < BALL_CLOSE_DIST_M:
                        self._ball_close_timer = BALL_CLOSE_TIMER_CYCLES

                    if robot_world_pos is not None:
                        local_3d = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
                        self._ball_world_pos = robot_world_pos + robot_rotation.apply(local_3d)
                else:
                    # Ball not seen this cycle — try world model first.
                    if self._ball_world_pos is not None and robot_world_pos is not None:
                        offset    = self._ball_world_pos - robot_world_pos
                        ball_local = orientation_inv.apply(offset)
                        ball_x    = float(ball_local[0])
                        ball_y    = float(ball_local[1])
                        ball_z    = float(ball_local[2])
                        ball_dist = float(np.linalg.norm(ball_local))
                    # Layer 2 — close timer / last known:
                    elif self._last_ball_x is not None and self._ball_close_timer > 0:
                        ball_x    = self._last_ball_x
                        ball_y    = self._last_ball_y
                        ball_z    = self._last_ball_z
                        ball_dist = float(np.sqrt(ball_x**2 + ball_y**2 +
                                                   (ball_z if ball_z is not None else 0.0)**2))

                # Count down the close timer.
                if self._ball_close_timer > 0:
                    self._ball_close_timer -= 1
                    if self._ball_close_timer == 0:
                        # Timer expired — clear smoother so the next search starts fresh.
                        self._head_scan_angle = 0.0
                        self._smooth_ball_x   = None
                        self._smooth_ball_y   = None

                # ── Ball-reset detection ───────────────────────────────────────
                # After a goal the ball teleports to the centre circle. Detect the
                # sudden large jump and wipe all stale tracking state.
                if ball_x is not None and self._smooth_ball_x is not None:
                    raw_dist    = float(np.sqrt(ball_x**2 + ball_y**2))
                    smooth_dist_prev = float(np.sqrt(self._smooth_ball_x**2 + self._smooth_ball_y**2))
                    if raw_dist > BALL_RESET_NEW_DIST_M and smooth_dist_prev < BALL_RESET_OLD_DIST_M:
                        logger.info('[P%d] Ball reset detected (%.1fm → %.1fm) — clearing state',
                                    self._player_no, smooth_dist_prev, raw_dist)
                        self._smooth_ball_x    = None
                        self._smooth_ball_y    = None
                        self._ball_world_pos   = None
                        self._last_ball_x      = None
                        self._last_ball_y      = None
                        self._last_ball_z      = None
                        self._ball_close_timer = 0
                        self._head_scan_angle  = 0.0

                # ── Exponential smoothing of ball position ────────────────────
                # Raw readings are noisy (vision fires every 2 cycles + body rotation).
                # The smoothed estimate is what drives all movement decisions.
                if ball_x is not None:
                    if self._smooth_ball_x is None:
                        self._smooth_ball_x = ball_x
                        self._smooth_ball_y = ball_y
                    else:
                        self._smooth_ball_x = (1 - BALL_SMOOTH_ALPHA) * self._smooth_ball_x + BALL_SMOOTH_ALPHA * ball_x
                        self._smooth_ball_y = (1 - BALL_SMOOTH_ALPHA) * self._smooth_ball_y + BALL_SMOOTH_ALPHA * ball_y
                elif self._ball_close_timer == 0 and self._ball_world_pos is None:
                    # All layers failed — clear the smoother so the robot searches fresh.
                    self._smooth_ball_x = None
                    self._smooth_ball_y = None

                # ── Periodic terminal logging ─────────────────────────────────
                if self._log_cycle % 25 == 0:
                    if ball_x is not None:
                        src = ('vision' if ball_visible
                               else ('world_model' if self._ball_world_pos is not None
                                     else 'last_known'))
                        logger.info('[P%d] Ball x=%.2f y=%.2f z=%.2f dist=%.2f  [%s]',
                                    self._player_no,
                                    ball_x, ball_y,
                                    ball_z if ball_z is not None else 0.0,
                                    ball_dist if ball_dist is not None else 0.0,
                                    src)
                    else:
                        logger.info('[P%d] Ball NOT visible — no estimate available', self._player_no)

                # ── Goal velocity (what the locomotion policy should do) ───────
                self.wait_until_walking = max(0, self.wait_until_walking - 1)
                if self.wait_until_walking > 0:
                    goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    goal_vel = self._decide_goal_vel(ball_raw, cur_head_yaw_deg)

                # ── Head tracking ─────────────────────────────────────────────
                if ENABLE_HEAD_TRACKING:
                    self._track_head(ball_raw, cur_head_yaw_deg, cur_head_pitch_deg)

                # ── Policy inference ──────────────────────────────────────────
                gait_phase_features = self._get_gait_phase_features()
                observation = np.concatenate([
                    scaled_joint_pos, scaled_joint_vel, scaled_previous_action,
                    scaled_ang_vel, goal_vel, gait_phase_features, projected_gravity,
                ])
                observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
                observation = np.clip(observation, -10.0, 10.0)

                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32,
                                              device=self.device).unsqueeze(0)
                    action_tensor, next_policy_hidden = self.policy(obs_tensor, self.policy_hidden)
                nn_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                target_joint_pos_deg = np.rad2deg(
                    self.joint_nominal_position + self.scaling_factor * nn_action)

                # Override head joints with our tracker.
                if ENABLE_HEAD_TRACKING:
                    target_joint_pos_deg[self.HEAD_YAW_IDX]   = self._head_yaw_target
                    target_joint_pos_deg[self.HEAD_PITCH_IDX] = self._head_pitch_target

                # ── Build + send motor commands ───────────────────────────────
                msg_list = [
                    f'({motor} {q:.2f} 0.0 {self.p_gain:.2f} {self.d_gain:.2f} 0.0)'
                    for motor, q in zip(self.ROBOT_MOTORS[self._model_name],
                                        target_joint_pos_deg, strict=False)
                ]
                self.previous_action = nn_action
                self.policy_hidden   = next_policy_hidden
                self._step_gait_manager()

                self._log_csv(game_time, play_mode, robot_world_pos,
                              ball_visible, ball_x, ball_y, ball_z, goal_x, goal_y)

                self._send_message(''.join(msg_list).encode())

            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception details:', exc_info=e.__traceback__)
                break

    # ═══════════════════════════════════════════════ networking
    def _send_message(self, msg: bytes | bytearray) -> None:
        self._sock.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def _receive_message(self) -> bytes | bytearray:
        if self._sock.recv_into(self._rcv_buffer, nbytes=4, flags=socket.MSG_WAITALL) != 4:
            raise ConnectionResetError
        msg_size = int.from_bytes(self._rcv_buffer[:4], byteorder='big', signed=False)
        if msg_size > self._rcv_buffer_size:
            self._rcv_buffer_size = msg_size
            self._rcv_buffer = bytearray(self._rcv_buffer_size)
        if self._sock.recv_into(self._rcv_buffer, nbytes=msg_size, flags=socket.MSG_WAITALL) != msg_size:
            raise ConnectionResetError
        return self._rcv_buffer[:msg_size]

    def parse_sensor_string(self, s: str) -> dict:
        """Parse the top-level sensor groups (time, pos, quat, GYR, HJ, …) into a dict."""
        result = {}
        top_level_pattern = re.compile(r'\((\w+)((?:\s*\([^()]*\))*)\)')
        for tag, inner in top_level_pattern.findall(s):
            items = re.findall(r'\(\s*(\w+)((?:\s+[^()]+)+)\)', inner)
            group = {}
            for key, vals in items:
                tokens = vals.strip().split()
                parsed = []
                for t in tokens:
                    try:
                        parsed.append(float(t))
                    except ValueError:
                        parsed.append(t)
                group[key] = parsed[0] if len(parsed) == 1 else parsed
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(group)
                else:
                    result[tag] = [result[tag], group]
            else:
                result[tag] = group
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RoboCup MuJoCo Soccer NN Client.')
    robots = list(Client.ROBOT_MOTORS.keys())
    parser.add_argument('-s', '--host',      type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port',      type=int, default=60000)
    parser.add_argument('-t', '--team',      type=str, default='Test')
    parser.add_argument('-n', '--player_no', type=int, default=1)
    parser.add_argument('-r', '--robot',     type=str, default=robots[0], choices=robots)
    args = parser.parse_args()

    client = Client(args.host, args.port, args.team, args.player_no, args.robot)

    def signal_handler(sig: int, frame: FrameType | int | signal.Handlers | None) -> None:
        del sig, frame
        client.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    client.run()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              