"""thesis_nn_client.py — Thesis-specific RoboCup MuJoCo client.

Derived from nn_client.py (internship client).  Key differences:
  - Stack-based S-expression parser that handles nested structures correctly.
  - Corrected GS extraction matching RCSSServerMJ 0.2.0 format (sl/sr included).
  - Corrected torso_pos extraction: server sends (p x y z), not (pos x y z).
  - Extra thesis CSV columns: wall_elapsed_time, score_left, score_right, data_valid.
  - --debug-perception flag writes the first complete perception message to a file.
  - Events: agent_started on connect, pm_<mode> on every play-mode transition,
    agent_stopped on shutdown.  CSV is flushed after every event row.

PRESERVATION REQUIREMENT: nn_client.py is NOT modified. This file is a
separate client.  Both may run simultaneously in the same directory.
"""

import argparse
import csv
import datetime
import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Mapping
from types import FrameType
from typing import ClassVar

import numpy as np
import re
from scipy.spatial.transform import Rotation as R
import torch

from torch_policy import load_policy_from_files

# ============================================================================
#  CONFIG — identical to nn_client.py; do not change values here.
# ============================================================================

ENABLE_HEAD_TRACKING     = True
HEAD_YAW_KP              = 0.6
HEAD_PITCH_KP            = 0.6
HEAD_YAW_SIGN            = +1
HEAD_PITCH_SIGN          = -1
HEAD_YAW_LIMIT_RAD       = (-np.pi/2, np.pi/2)
HEAD_PITCH_LIMIT_RAD     = (np.deg2rad(-20.0), np.deg2rad(70.0))
HEAD_SWEEP_RAD_PER_CYCLE = np.deg2rad(4.0)

ENABLE_BALL_FOLLOWING = True
FOLLOW_FORWARD_SPEED  = 1.0
STOP_DIST_M           = 1.0
SLOW_DIST_M           = 2.0
ORBIT_ENGAGE_M        = 1.5
ORBIT_RADIUS_M        = STOP_DIST_M
ORBIT_SPEED           = 0.3
ORBIT_RADIAL_KP       = 0.5
ALIGN_TOL_RAD         = np.deg2rad(5)
ALIGN_THRESHOLD_RAD   = np.deg2rad(8)
STEER_KP              = 1.0 / (np.pi / 4)
APPROACH_SPEED        = 0.7
APPROACH_LATERAL_KP   = 0.4
APPROACH_GOAL_LAT_KP  = 0.0

KICKOFF_STAND_CYCLES   = 150
FORMATION_ARRIVE_DIST  = 0.5
DR_WALK_SPEED_MPS      = 0.83

BALL_CLOSE_DIST_M       = 2.0
BALL_CLOSE_TIMER_CYCLES = 150
BALL_SMOOTH_ALPHA       = 0.25
BALL_RESET_NEW_DIST_M   = 5.0
BALL_RESET_OLD_DIST_M   = 3.0

ENABLE_SEARCH    = True
LOST_BALL_CYCLES = 30
SEARCH_YAW_SPEED = 0.6

TEAMMATE_STALE_CYCLES = 150
SUPPORT_DIST_M        = 6.0

ENABLE_CSV_LOGGING = True
CSV_EVERY_N_CYCLES = 5
CSV_DIR            = "CSV"

# ============================================================================

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
ch.setLevel(logging.INFO)
logging.basicConfig(handlers=[ch], level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ThesisClient:
    """Thesis experiment client — corrected parser + extended CSV logging."""

    BEAM_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
        1:  (27.5,  0.0, 0), 2:  (22.0, 12.0, 0), 3:  (22.0,  4.0, 0),
        4:  (22.0, -4.0, 0), 5:  (22.0,-12.0, 0), 6:  (15.0,  0.0, 0),
        7:  ( 4.0, 16.0, 0), 8:  (11.0,  6.0, 0), 9:  (11.0, -6.0, 0),
        10: ( 4.0,-16.0, 0), 11: ( 7.0,  0.0, 0),
    }

    SIDELINE_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
        1:  (27.5, 20.0, -90), 2:  (22.0, 22.0, -90), 3:  (22.0, 20.0, -90),
        4:  (22.0,-20.0,  90), 5:  (22.0,-22.0,  90), 6:  (15.0, 20.0, -90),
        7:  ( 4.0, 20.0, -90), 8:  (11.0, 20.0, -90), 9:  (11.0,-20.0,  90),
        10: ( 4.0,-20.0,  90), 11: ( 7.0, 20.0, -90),
    }

    ROBOT_MOTORS: ClassVar[Mapping[str, tuple[str, ...]]] = {
        'ant': ('l4e1','l4e2','l1e1','l1e2','l2e1','l2e2','l3e1','l3e2'),
        'T1':  ('he1','he2','lae1','lae2','lae3','lae4','rae1','rae2','rae3','rae4',
                'te1','lle1','lle2','lle3','lle4','lle5','lle6',
                'rle1','rle2','rle3','rle4','rle5','rle6'),
    }

    HEAD_YAW_IDX   = 0
    HEAD_PITCH_IDX = 1

    def __init__(self, host, port, team, player_no, model_name=None,
                 default_role='attacker', ready_file=None, debug_perception=False):
        self._host       = host
        self._port       = port
        self._model_name = 'ant' if model_name is None else model_name
        self._team       = team
        self._player_no  = player_no
        self._ready_file = ready_file
        self._debug_perception = debug_perception
        self._debug_perception_written = False

        self._policy_checkpoint = "locomotion_nn.pth"
        self._policy_meta       = "locomotion_nn_meta.json"
        self._gait_period       = 1.0
        self._policy_dt         = 0.02

        self._rcv_buffer_size = 1024
        self._rcv_buffer      = bytearray(self._rcv_buffer_size)
        self._sock            = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._has_beamed      = False

        # head tracking
        self._head_yaw_target   = 0.0
        self._head_pitch_target = 0.0
        self._head_scan_angle   = 0.0
        self._search_dir        = 1.0

        # ball tracking
        self._last_ball_x       = None
        self._last_ball_y       = None
        self._last_ball_z       = None
        self._smooth_ball_x     = None
        self._smooth_ball_y     = None
        self._ball_world_pos    = None
        self._ball_close_timer  = 0
        self._cycles_since_ball = 999
        self._orbiting          = False
        self._spawn_yaw         = None
        self._goal_world_dir    = None
        self._goal_dir_init     = False
        self._goal_dir_trusted  = False
        self._goal_world_pos    = None

        # multi-robot
        self._teammate_world_pos = {}
        self._teammate_last_seen = {}
        self._robot_world_pos    = None
        self._dr_pos             = None
        self._dr_walk_dir        = None
        self._default_role       = default_role
        self._role               = default_role

        # play mode tracking
        self._play_mode         = None
        self._prev_play_mode    = None
        self._team_side         = None   # 'left' or 'right' — filled from GS tl/tr
        self._formation_phase     = True
        self._at_formation_cycles = 0
        self._formation_arrived   = False
        self._post_kickoff_wait   = 0

        # thesis extras
        self._score_left   = None
        self._score_right  = None
        self._thesis_start_time = time.monotonic()

        # misc
        self._cycle             = 0
        self._log_cycle         = 0
        self._csv_writer        = None
        self._csv_file          = None
        self._csv_header_done   = False
        self._aligned_with_goal = False

        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # ================================================================ sexp parser

    @staticmethod
    def _parse_sexp_tree(s: str) -> list:
        """Parse a string of S-expressions into a nested Python list.

        Each S-expression  (tag child1 child2 ...)  becomes a list
        [tag, child1, child2, ...] where children may be strings or nested lists.
        Top-level atoms (whitespace-separated tokens outside any parens) are
        returned as bare strings.  The return value is a list of top-level items.

        This correctly handles arbitrary nesting and is immune to the flat-regex
        problem that breaks the original nn_client.py parser on 0.2.0 messages.
        """
        stack: list[list] = [[]]          # stack[0] is the implicit top-level list
        current_token: list[str] = []

        def _flush():
            if current_token:
                stack[-1].append(''.join(current_token))
                current_token.clear()

        for ch in s:
            if ch == '(':
                _flush()
                stack.append([])
            elif ch == ')':
                _flush()
                finished = stack.pop()
                stack[-1].append(finished)
            elif ch in (' ', '\t', '\n', '\r'):
                _flush()
            else:
                current_token.append(ch)

        _flush()
        return stack[0]   # top-level list of items

    # ================================================================ GS extractor

    def _extract_gs(self, tree: list) -> dict | None:
        """Find and parse the GS node from a parsed sexp tree.

        Expected structure: ['GS', ['t', '<time>'], ['pm', '<mode>'],
                             ['tl', '<name>'], ['tr', '<name>'],
                             ['sl', '<score>'], ['sr', '<score>']]
        Returns a dict with keys: t (float), pm (str), tl, tr, sl, sr (str or None).
        Returns None when no GS node is present.
        """
        for item in tree:
            if isinstance(item, list) and item and item[0] == 'GS':
                result: dict = {'t': 0.0, 'pm': 'Unknown',
                                'tl': None, 'tr': None,
                                'sl': None, 'sr': None}
                for child in item[1:]:
                    if not isinstance(child, list) or len(child) < 2:
                        continue
                    key = child[0]
                    val = child[1]
                    if key == 't':
                        try:
                            result['t'] = float(val)
                        except (ValueError, TypeError):
                            pass
                    elif key == 'pm':
                        result['pm'] = str(val)
                    elif key == 'tl':
                        result['tl'] = str(val)
                    elif key == 'tr':
                        result['tr'] = str(val)
                    elif key == 'sl':
                        try:
                            result['sl'] = int(val)
                        except (ValueError, TypeError):
                            result['sl'] = val
                    elif key == 'sr':
                        try:
                            result['sr'] = int(val)
                        except (ValueError, TypeError):
                            result['sr'] = val
                return result
        return None

    # ================================================================ torso extractor

    def _extract_torso_pos(self, tree: list) -> tuple | None:
        """Extract torso world position from a parsed sexp tree.

        RCSSServerMJ 0.2.0 format:
            (pos (n torso_pos) (p <x> <y> <z>))

        The original nn_client.py parser used (pos x y z) which never matches.
        This extractor looks for the (p ...) sub-node explicitly.
        Returns (x, y, z) as floats, or None when not found.
        """
        for item in tree:
            if not isinstance(item, list) or item[0] != 'pos':
                continue
            # Find (n torso_pos) marker
            has_marker = any(
                isinstance(c, list) and len(c) >= 2
                and c[0] == 'n' and c[1] == 'torso_pos'
                for c in item[1:]
            )
            if not has_marker:
                continue
            # Find (p x y z) position sub-node
            for c in item[1:]:
                if isinstance(c, list) and len(c) >= 4 and c[0] == 'p':
                    try:
                        return (float(c[1]), float(c[2]), float(c[3]))
                    except (ValueError, TypeError):
                        pass
        return None

    # ================================================================ player extractor

    def _parse_players_from_tree(self, tree: list) -> list:
        """Extract player observations from a parsed sexp tree.

        Each (P (team <name>) (id <n>) (head (pol d az el)) ...) node
        becomes a dict: {team, id, distance, azimuth (rad), elevation (rad)}.
        Multiple (pol ...) blocks are averaged, matching nn_client.py behaviour.
        """
        players = []
        for item in tree:
            if not isinstance(item, list) or not item or item[0] != 'P':
                continue
            team_name = None
            player_id = None
            pol_values: list[tuple[float, float, float]] = []

            for child in item[1:]:
                if not isinstance(child, list) or not child:
                    continue
                tag = child[0]
                if tag == 'team' and len(child) >= 2:
                    team_name = str(child[1])
                elif tag == 'id' and len(child) >= 2:
                    try:
                        player_id = int(child[1])
                    except (ValueError, TypeError):
                        pass
                else:
                    # Any sub-node may contain a (pol d az el) child
                    for sub in child[1:]:
                        if isinstance(sub, list) and len(sub) >= 4 and sub[0] == 'pol':
                            try:
                                pol_values.append((float(sub[1]),
                                                   float(sub[2]),
                                                   float(sub[3])))
                            except (ValueError, TypeError):
                                pass

            if team_name is None or player_id is None or not pol_values:
                continue

            arr = np.array(pol_values, dtype=np.float32)
            d, az, el = arr.mean(axis=0)
            players.append({
                'team':      team_name,
                'id':        player_id,
                'distance':  float(d),
                'azimuth':   float(np.deg2rad(az)),
                'elevation': float(np.deg2rad(el)),
            })
        return players

    # ================================================================ run / shutdown

    def run(self):
        logger.info('Connecting to %s:%d...', self._host, self._port)
        try:
            self._sock.connect((self._host, self._port))
        except ConnectionRefusedError:
            logger.error('Connection refused.')
            return
        client_thread = threading.Thread(target=self._action_loop)
        client_thread.start()
        client_thread.join()
        if self._csv_file:
            self._csv_file.close()
        self._sock.close()

    def shutdown(self):
        self._sock.shutdown(socket.SHUT_RDWR)

    # ================================================================ policy helpers

    def _init_policy_runtime_state(self):
        self.previous_action    = np.zeros(self.nr_joints, dtype=np.float32)
        self.policy_hidden      = self.policy.initialize_carry(batch_size=1, device=self.device)
        self.wait_until_walking = 50

        self.gait_phase_offset = np.array([0.0, -np.pi], dtype=np.float32)
        self.gait_phase        = self.gait_phase_offset.copy()
        self.gait_mean_freq    = 1.0 / self._gait_period
        self.gait_freq         = self.gait_mean_freq
        self.gait_phase_dt     = 2.0 * np.pi * self._policy_dt * self.gait_freq

        self._last_ball_x = self._last_ball_y = self._last_ball_z = None
        self._smooth_ball_x = self._smooth_ball_y = None
        self._ball_world_pos   = None
        self._ball_close_timer = 0
        self._cycles_since_ball = 999
        self._head_scan_angle  = 0.0
        self._goal_world_pos     = None
        self._teammate_world_pos = {}
        self._teammate_last_seen = {}
        self._robot_world_pos    = None
        self._dr_pos             = None
        self._dr_walk_dir        = None
        self._role               = self._default_role
        self._aligned_with_goal  = False
        self._orbiting           = False
        self._spawn_yaw          = None
        self._play_mode          = None
        self._prev_play_mode     = None
        self._formation_phase     = True
        self._at_formation_cycles = 0
        self._formation_arrived   = False
        self._post_kickoff_wait   = 0

    @staticmethod
    def _wrap_to_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _get_gait_phase_features(self):
        phase_tp1 = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt)
        return np.concatenate([np.sin(phase_tp1), np.cos(phase_tp1)]).astype(np.float32)

    def _step_gait_manager(self):
        self.gait_phase = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt).astype(np.float32)

    # ================================================================ vision helpers

    @staticmethod
    def _polar_to_cartesian(dist, az_rad, el_rad):
        return (float(dist*np.cos(el_rad)*np.cos(az_rad)),
                float(dist*np.cos(el_rad)*np.sin(az_rad)),
                float(dist*np.sin(el_rad)))

    @staticmethod
    def _extract_balanced(s, start):
        depth = 0
        for i in range(start, len(s)):
            if s[i] == '(':  depth += 1
            elif s[i] == ')':
                depth -= 1
                if depth == 0: return s[start:i+1]
        return s[start:]

    def _parse_ball(self, msg):
        m = re.search(r'\(B\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)\)', msg)
        return (float(m.group(1)),
                float(np.deg2rad(float(m.group(2)))),
                float(np.deg2rad(float(m.group(3))))) if m else None

    def _parse_goal(self, msg, cur_head_yaw=0.0):
        groups = {}
        for side, names in (('R', ('G1R', 'G2R')), ('L', ('G1L', 'G2L'))):
            xs, ys = [], []
            for name in names:
                m = re.search(rf'\({name}\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\)', msg)
                if m:
                    d, az = float(m.group(1)), np.deg2rad(float(m.group(2)))
                    xs.append(d * np.cos(az)); ys.append(d * np.sin(az))
            if xs:
                groups[side] = (float(np.mean(xs)), float(np.mean(ys)))
        if not groups:
            return (None, None)
        cos_h, sin_h = float(np.cos(cur_head_yaw)), float(np.sin(cur_head_yaw))
        def _body_x(side):
            hx, hy = groups[side]
            return hx * cos_h - hy * sin_h
        best = max(groups, key=_body_x)
        return groups[best]

    # ================================================================ multi-robot helpers

    def _update_teammates(self, players, robot_world_pos, robot_rotation):
        for p in players:
            if p['team'] != self._team or p['id'] == self._player_no:
                continue
            if robot_world_pos is None:
                continue
            lx, ly, lz = self._polar_to_cartesian(p['distance'], p['azimuth'], p['elevation'])
            local = np.array([lx, ly, lz], dtype=np.float32)
            self._teammate_world_pos[p['id']] = robot_world_pos + robot_rotation.apply(local)
            self._teammate_last_seen[p['id']] = self._cycle

        stale = [pid for pid, last in self._teammate_last_seen.items()
                 if self._cycle - last > TEAMMATE_STALE_CYCLES]
        for pid in stale:
            self._teammate_world_pos.pop(pid, None)
            self._teammate_last_seen.pop(pid, None)

    def _compute_role(self, robot_world_pos):
        if self._ball_world_pos is None or robot_world_pos is None:
            return self._default_role
        if not self._teammate_world_pos:
            return self._default_role
        ball_xy = self._ball_world_pos[:2]
        my_dist = float(np.linalg.norm(ball_xy - robot_world_pos[:2]))
        for tw in self._teammate_world_pos.values():
            if float(np.linalg.norm(ball_xy - tw[:2])) < my_dist:
                return 'supporter'
        return 'attacker'

    def _supporter_goal_vel(self, robot_world_pos, orientation_inv):
        if self._ball_world_pos is None or robot_world_pos is None:
            return None
        ball_xy  = self._ball_world_pos[:2]
        robot_xy = robot_world_pos[:2]
        toward_c = -ball_xy
        c_dist   = float(np.linalg.norm(toward_c))
        unit     = toward_c / c_dist if c_dist > 0.5 else np.array([0.0, 1.0])
        support_xy = ball_xy + SUPPORT_DIST_M * unit
        off3 = np.array([support_xy[0]-robot_xy[0], support_xy[1]-robot_xy[1], 0.0], dtype=np.float32)
        loc  = orientation_inv.apply(off3)
        dx, dy = float(loc[0]), float(loc[1])
        dist_to_support = float(np.sqrt(dx*dx + dy*dy))
        if dist_to_support < 1.0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        body_ang = float(np.arctan2(dy, dx))
        yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
        return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)

    def _walk_to_formation(self, robot_world_pos, orientation_inv):
        target_x, target_y, _ = self.BEAM_POSES[self._player_no]
        target_xy  = np.array([target_x, target_y], dtype=np.float32)
        robot_xy   = robot_world_pos[:2]
        offset     = target_xy - robot_xy
        dist       = float(np.linalg.norm(offset))
        if dist < FORMATION_ARRIVE_DIST:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        offset_3d = np.array([offset[0], offset[1], 0.0], dtype=np.float32)
        local     = orientation_inv.apply(offset_3d)
        dx, dy    = float(local[0]), float(local[1])
        body_ang  = float(np.arctan2(dy, dx))
        yaw_vel   = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
        speed     = float(np.clip(dist * 0.5, 0.2, FOLLOW_FORWARD_SPEED))
        if self._log_cycle % 25 == 0:
            logger.info('[P%d] → formation (%.1f, %.1f)  dist=%.2fm  yaw_err=%.1f°',
                        self._player_no, target_x, target_y, dist, np.rad2deg(body_ang))
        return np.array([speed, 0.0, yaw_vel], dtype=np.float32)

    def _decide_goal_vel(self, ball_raw, cur_head_yaw, role, robot_world_pos, orientation_inv):
        if not ENABLE_BALL_FOLLOWING:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if role == 'supporter':
            sv = self._supporter_goal_vel(robot_world_pos, orientation_inv)
            if sv is not None:
                return sv
        if ball_raw is not None:
            dist, az, _ = ball_raw
            body_ang = cur_head_yaw + az
            yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
            if dist > ORBIT_ENGAGE_M:
                return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)
            else:
                vx = float(np.clip((dist - ORBIT_RADIUS_M) * ORBIT_RADIAL_KP,
                                   -FOLLOW_FORWARD_SPEED, FOLLOW_FORWARD_SPEED))
                return np.array([vx, ORBIT_SPEED, yaw_vel], dtype=np.float32)
        if ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            return np.array([0.0, 0.0, self._search_dir * SEARCH_YAW_SPEED], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ================================================================ thesis CSV

    def _open_csv(self):
        if not ENABLE_CSV_LOGGING: return
        os.makedirs(CSV_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(CSV_DIR,
                            f'thesis_robot_log_{self._team}_p{self._player_no}_{ts}.csv')
        self._csv_file   = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'game_time', 'play_mode', 'player_no', 'team', 'team_side',
            'robot_world_x', 'robot_world_y', 'robot_world_z',
            'role', 'ball_visible',
            'ball_rel_x', 'ball_rel_y', 'ball_rel_z',
            'ball_world_x', 'ball_world_y', 'ball_world_z',
            'goal_rel_x', 'goal_rel_y',
            'nearest_teammate_id', 'nearest_teammate_world_x',
            'nearest_teammate_world_y', 'nearest_teammate_dist_to_ball',
            # Thesis-specific extras
            'wall_elapsed_time', 'score_left', 'score_right', 'data_valid',
            'event',
        ])
        self._csv_header_done = True
        logger.info('[ThesisClient] Logging to %s', path)

    def _log_csv(self, game_time, play_mode, rwp, ball_visible,
                 ball_x, ball_y, ball_z, goal_x, goal_y,
                 team_side, score_left, score_right, data_valid, event=''):
        if not ENABLE_CSV_LOGGING or not self._csv_writer: return
        if event == '' and self._cycle % CSV_EVERY_N_CYCLES != 0: return

        bwp  = self._ball_world_pos
        nid  = nwx = nwy = ndtb = ''
        if self._teammate_world_pos and bwp is not None:
            bxy  = bwp[:2]
            best = min(self._teammate_world_pos,
                       key=lambda pid: np.linalg.norm(bxy - self._teammate_world_pos[pid][:2]))
            tw   = self._teammate_world_pos[best]
            nid  = best
            nwx  = round(float(tw[0]), 3)
            nwy  = round(float(tw[1]), 3)
            ndtb = round(float(np.linalg.norm(bxy - tw[:2])), 3)

        wall_elapsed = round(time.monotonic() - self._thesis_start_time, 3)

        self._csv_writer.writerow([
            round(game_time, 3), play_mode, self._player_no, self._team,
            team_side or '',
            round(float(rwp[0]), 3) if rwp is not None else '',
            round(float(rwp[1]), 3) if rwp is not None else '',
            round(float(rwp[2]), 3) if rwp is not None else '',
            self._role, 1 if ball_visible else 0,
            round(ball_x, 3) if ball_x is not None else '',
            round(ball_y, 3) if ball_y is not None else '',
            round(ball_z, 3) if ball_z is not None else '',
            round(float(bwp[0]), 3) if bwp is not None else '',
            round(float(bwp[1]), 3) if bwp is not None else '',
            round(float(bwp[2]), 3) if bwp is not None else '',
            round(goal_x, 3) if goal_x is not None else '',
            round(goal_y, 3) if goal_y is not None else '',
            nid, nwx, nwy, ndtb,
            wall_elapsed,
            score_left if score_left is not None else '',
            score_right if score_right is not None else '',
            data_valid,
            event,
        ])
        if event:
            self._csv_file.flush()

    # ================================================================ main loop

    def _action_loop(self):
        self.nr_joints = len(self.ROBOT_MOTORS[self._model_name])
        self.p_gain = 25.0; self.d_gain = 0.6; self.scaling_factor = 0.5
        self.joint_nominal_position = np.array([
            0.0,0.0, 0.0,-1.4,0.0,-0.4, 0.0,1.4,0.0,0.4, 0.0,
            -0.4,0.0,0.0,0.8,-0.4,0.0, -0.4,0.0,0.0,0.8,-0.4,0.0,
        ], dtype=np.float32)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy, self.policy_meta = load_policy_from_files(
            self._policy_checkpoint, self._policy_meta, self.device)
        self._init_policy_runtime_state()
        self._open_csv()

        # Log agent_started event
        if self._csv_writer and self._csv_header_done:
            self._log_csv(0.0, 'PreConnect', None, False,
                          None, None, None, None, None,
                          None, None, None, 0, event='agent_started')

        logger.info('Initializing agent...')
        init_msg = f'(init {self._model_name} {self._team} {self._player_no})'
        self._send_message(init_msg.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                if not self._has_beamed:
                    self._has_beamed = True
                    self._init_policy_runtime_state()
                    sx, sy, sa = self.SIDELINE_POSES[self._player_no]
                    self._dr_pos = np.array([sx, sy], dtype=np.float32)
                    _sa_rad = np.deg2rad(sa)
                    self._dr_walk_dir = np.array(
                        [np.cos(_sa_rad), np.sin(_sa_rad)], dtype=np.float32)
                    _btx, _bty, _ = self.BEAM_POSES[self._player_no]
                    logger.info('[P%d] DR init: start=(%.1f,%.1f) dir=(%.2f,%.2f) target=(%.1f,%.1f)',
                                self._player_no, sx, sy,
                                float(self._dr_walk_dir[0]), float(self._dr_walk_dir[1]),
                                _btx, _bty)
                    self._send_message(f'(beam {sx:.1f} {sy:.1f} {sa:.1f})'.encode())
                    continue

                self._cycle += 1; self._log_cycle += 1
                perception_msg_str = perception_msg.decode()

                # ------ debug-perception: write first complete message once ------
                if self._debug_perception and not self._debug_perception_written:
                    dbg_path = f'perception_debug_{self._team}_p{self._player_no}.txt'
                    try:
                        with open(dbg_path, 'w') as _dbg:
                            _dbg.write(perception_msg_str)
                        logger.info('[P%d] Debug perception written to %s', self._player_no, dbg_path)
                    except OSError as _e:
                        logger.warning('[P%d] Could not write debug file: %s', self._player_no, _e)
                    self._debug_perception_written = True

                # ------ parse sexp tree (thesis corrected parser) ------
                sexp_tree = self._parse_sexp_tree(perception_msg_str)

                # ------ standard sensor parse (joints / IMU / quat) ------
                perception_data = self.parse_sensor_string(perception_msg_str)

                # joints
                joint_pos_degrees = np.array([h['ax'] for h in perception_data['HJ']], dtype=np.float32)
                joint_pos  = np.deg2rad(joint_pos_degrees).astype(np.float32)
                joint_vel_degrees = np.array([h['vx'] for h in perception_data['HJ']], dtype=np.float32)
                joint_vel  = np.deg2rad(joint_vel_degrees).astype(np.float32)
                cur_head_yaw   = float(joint_pos[self.HEAD_YAW_IDX])
                cur_head_pitch = float(joint_pos[self.HEAD_PITCH_IDX])
                scaled_joint_pos  = (joint_pos - self.joint_nominal_position) / 3.14
                scaled_joint_vel  = joint_vel / 100.0
                scaled_previous_action = self.previous_action / 10.0

                ang_vel = np.deg2rad(np.array(perception_data['GYR']['rt'], dtype=np.float32))
                scaled_and_clipped_ang_vel = np.clip(ang_vel / 50.0, -1.0, 1.0).astype(np.float32)

                orientation_quat_mj_convention = np.array(perception_data['quat']['q'], dtype=np.float32)
                rot = R.from_quat([orientation_quat_mj_convention[1],
                                   orientation_quat_mj_convention[2],
                                   orientation_quat_mj_convention[3],
                                   orientation_quat_mj_convention[0]])
                orientation_quat_inv = rot.inv()
                projected_gravity    = orientation_quat_inv.apply(
                    np.array([0.0, 0.0, -1.0])).astype(np.float32)

                if self._spawn_yaw is None:
                    _sfwd = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                    self._spawn_yaw = float(np.arctan2(float(_sfwd[1]), float(_sfwd[0])))
                    logger.info('[P%d] spawn_yaw recorded: %.1f°',
                                self._player_no, np.rad2deg(self._spawn_yaw))

                # ------ thesis corrected torso_pos extraction ------
                torso_raw = self._extract_torso_pos(sexp_tree)
                if torso_raw is not None:
                    rwp = np.array(torso_raw, dtype=np.float32)
                    self._robot_world_pos = rwp
                else:
                    rwp = self._robot_world_pos

                if rwp is None and self._dr_pos is not None:
                    rwp = np.array(
                        [self._dr_pos[0], self._dr_pos[1], 0.0], dtype=np.float32)

                # ------ thesis corrected GS extraction ------
                gs = self._extract_gs(sexp_tree)
                if gs is not None:
                    game_time = gs['t']
                    play_mode = gs['pm']
                    # score
                    self._score_left  = gs['sl']
                    self._score_right = gs['sr']
                    # team-side detection (done once)
                    if self._team_side is None:
                        if gs['tl'] and gs['tl'] == self._team:
                            self._team_side = 'left'
                        elif gs['tr'] and gs['tr'] == self._team:
                            self._team_side = 'right'
                else:
                    game_time = 0.0
                    play_mode = 'Unknown'

                # data_valid: all four critical fields present and sane
                sim_time_valid   = gs is not None and gs['t'] > 0.0
                play_mode_valid  = gs is not None and gs['pm'] != 'Unknown'
                data_valid = int(
                    sim_time_valid and play_mode_valid
                    and self._team_side is not None
                    and torso_raw is not None
                )

                # ------ play mode transition ------
                self._prev_play_mode = self._play_mode
                self._play_mode      = play_mode

                # ------ players (thesis parser) ------
                players = self._parse_players_from_tree(sexp_tree)
                self._update_teammates(players, rwp, rot)

                # ------ ball ------
                ball_raw     = self._parse_ball(perception_msg_str)
                ball_visible = ball_raw is not None
                if ball_raw is not None:
                    ball_dist = ball_raw[0]
                    ball_x, ball_y, ball_z = self._polar_to_cartesian(*ball_raw)
                    self._cycles_since_ball = 0
                else:
                    ball_dist = ball_x = ball_y = ball_z = None
                    self._cycles_since_ball += 1

                # ------ goal ------
                goal_x, goal_y = self._parse_goal(perception_msg_str, cur_head_yaw)
                goal_cam_az = float(np.arctan2(goal_y, goal_x)) if goal_x is not None else None
                if goal_x is not None and rwp is not None:
                    _ch, _sh = float(np.cos(cur_head_yaw)), float(np.sin(cur_head_yaw))
                    _gbx = goal_x * _ch - goal_y * _sh
                    _gby = goal_x * _sh + goal_y * _ch
                    self._goal_world_pos = rwp + rot.apply(
                        np.array([_gbx, _gby, 0.0], dtype=np.float32))
                if goal_x is None and self._goal_world_pos is not None and rwp is not None:
                    gl = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                    goal_x, goal_y = float(gl[0]), float(gl[1])

                _goal_body_az = None
                if goal_cam_az is not None:
                    _goal_body_az = float(self._wrap_to_pi(goal_cam_az + cur_head_yaw))
                if self._goal_world_pos is not None and rwp is not None:
                    _glb = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                    _goal_body_az = float(self._wrap_to_pi(
                        float(np.arctan2(_glb[1], _glb[0]))))
                _ball_body_az = (float(self._wrap_to_pi(cur_head_yaw + ball_raw[1]))
                                 if ball_raw is not None else None)

                # ball persistence
                if ball_x is not None:
                    self._last_ball_x = ball_x; self._last_ball_y = ball_y; self._last_ball_z = ball_z
                    if ball_dist is not None and ball_dist < BALL_CLOSE_DIST_M:
                        self._ball_close_timer = BALL_CLOSE_TIMER_CYCLES
                    if rwp is not None:
                        self._ball_world_pos = rwp + rot.apply(
                            np.array([ball_x, ball_y, ball_z], dtype=np.float32))
                else:
                    if self._ball_world_pos is not None and rwp is not None:
                        bl = orientation_quat_inv.apply(self._ball_world_pos - rwp)
                        ball_x, ball_y, ball_z = float(bl[0]), float(bl[1]), float(bl[2])
                        ball_dist = float(np.linalg.norm(bl))
                    elif self._last_ball_x is not None and self._ball_close_timer > 0:
                        ball_x, ball_y, ball_z = self._last_ball_x, self._last_ball_y, self._last_ball_z
                        ball_dist = float(np.sqrt(
                            ball_x**2 + ball_y**2 + (ball_z or 0.0)**2))

                if self._ball_close_timer > 0:
                    self._ball_close_timer -= 1
                    if self._ball_close_timer == 0:
                        self._head_scan_angle = 0.0
                        self._smooth_ball_x = self._smooth_ball_y = None

                if ball_x is not None and self._smooth_ball_x is not None:
                    if (float(np.sqrt(ball_x**2+ball_y**2)) > BALL_RESET_NEW_DIST_M and
                            float(np.sqrt(self._smooth_ball_x**2+self._smooth_ball_y**2))
                            < BALL_RESET_OLD_DIST_M):
                        logger.info('[P%d] Ball reset detected — clearing state', self._player_no)
                        self._smooth_ball_x = self._smooth_ball_y = None
                        self._ball_world_pos = None
                        self._last_ball_x = self._last_ball_y = self._last_ball_z = None
                        self._ball_close_timer = 0; self._head_scan_angle = 0.0

                if ball_x is not None:
                    a = BALL_SMOOTH_ALPHA
                    if self._smooth_ball_x is None:
                        self._smooth_ball_x = ball_x; self._smooth_ball_y = ball_y
                    else:
                        self._smooth_ball_x = (1-a)*self._smooth_ball_x + a*ball_x
                        self._smooth_ball_y = (1-a)*self._smooth_ball_y + a*ball_y
                elif self._ball_close_timer == 0 and self._ball_world_pos is None:
                    self._smooth_ball_x = self._smooth_ball_y = None

                # Voronoi role
                self._role = self._compute_role(rwp)

                if (self._prev_play_mode == 'BeforeKickOff'
                        and play_mode != 'BeforeKickOff'):
                    self._formation_phase   = False
                    self._post_kickoff_wait = KICKOFF_STAND_CYCLES
                    logger.info('[P%d] Kickoff! Holding for %d cycles.',
                                self._player_no, KICKOFF_STAND_CYCLES)

                # ------ thesis CSV events ------
                if ENABLE_CSV_LOGGING and self._csv_writer and self._csv_header_done:
                    if (self._prev_play_mode is None and play_mode == 'BeforeKickOff'):
                        self._log_csv(game_time, play_mode, rwp, ball_visible,
                                      ball_x, ball_y, ball_z, goal_x, goal_y,
                                      self._team_side, self._score_left,
                                      self._score_right, data_valid,
                                      event='game_init')
                        logger.info('[P%d] CSV: game_init logged', self._player_no)
                    elif (self._prev_play_mode is not None
                          and self._prev_play_mode != play_mode):
                        self._log_csv(game_time, play_mode, rwp, ball_visible,
                                      ball_x, ball_y, ball_z, goal_x, goal_y,
                                      self._team_side, self._score_left,
                                      self._score_right, data_valid,
                                      event=f'pm_{play_mode}')
                        logger.info('[P%d] CSV: pm_%s logged', self._player_no, play_mode)

                # periodic log
                if self._log_cycle % 25 == 0:
                    src = ('vision' if ball_visible else
                           'world_model' if self._ball_world_pos is not None else 'last_known')
                    if ball_x is not None:
                        logger.info('[P%d] role=%-9s  Ball x=%.2f y=%.2f dist=%.2f  [%s]  '
                                    'side=%s  sl=%s  sr=%s  valid=%d',
                                    self._player_no, self._role.upper(),
                                    ball_x, ball_y, ball_dist or 0.0, src,
                                    self._team_side, self._score_left,
                                    self._score_right, data_valid)
                    else:
                        logger.info('[P%d] role=%-9s  Ball NOT visible  side=%s  valid=%d',
                                    self._player_no, self._role.upper(),
                                    self._team_side, data_valid)

                self.wait_until_walking = max(0, self.wait_until_walking - 1)

                # dead-reckoning update
                if (self._formation_phase and self.wait_until_walking == 0
                        and self._dr_pos is not None and self._dr_walk_dir is not None):
                    _btx2, _bty2, _ = self.BEAM_POSES[self._player_no]
                    _dr_to_target = float(np.linalg.norm(
                        self._dr_pos - np.array([_btx2, _bty2], dtype=np.float32)))
                    if _dr_to_target > 0.0:
                        _step = DR_WALK_SPEED_MPS * self._policy_dt
                        if _step >= _dr_to_target:
                            self._dr_pos = np.array([_btx2, _bty2], dtype=np.float32)
                        else:
                            self._dr_pos = self._dr_pos + self._dr_walk_dir * _step

                # formation walk logic (identical to nn_client.py)
                if self._formation_phase:
                    if self.wait_until_walking > 0:
                        goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    elif rwp is not None:
                        pass
                    else:
                        goal_vel = np.array([FOLLOW_FORWARD_SPEED, 0.0, 0.0], dtype=np.float32)
                    if self.wait_until_walking == 0 and rwp is not None:
                        _tx, _ty, _ = self.BEAM_POSES[self._player_no]
                        _dist_form = float(np.linalg.norm(
                            rwp[:2] - np.array([_tx, _ty], dtype=np.float32)))
                        if _dist_form < FORMATION_ARRIVE_DIST:
                            if not self._formation_arrived:
                                self._formation_arrived = True
                                self._post_kickoff_wait = KICKOFF_STAND_CYCLES
                                logger.info('[P%d] Formation position reached.', self._player_no)
                                if self._ready_file:
                                    try:
                                        with open(self._ready_file, 'a') as _rf:
                                            _rf.write(f'{self._player_no}\n')
                                    except OSError as _e:
                                        logger.warning('[P%d] Could not write ready file: %s',
                                                       self._player_no, _e)
                            _yaw_to_ball = float(np.clip(cur_head_yaw * STEER_KP, -1.0, 1.0))
                            goal_vel = np.array([0.0, 0.0, _yaw_to_ball], dtype=np.float32)
                            if self._post_kickoff_wait > 0:
                                self._post_kickoff_wait -= 1
                                if self._post_kickoff_wait == 0:
                                    self._formation_phase = False
                        else:
                            self._at_formation_cycles = 0
                            goal_vel = self._walk_to_formation(rwp, orientation_quat_inv)

                elif self._post_kickoff_wait > 0:
                    self._post_kickoff_wait -= 1
                    _yaw_to_ball = float(np.clip(cur_head_yaw * STEER_KP, -1.0, 1.0))
                    goal_vel = np.array([0.0, 0.0, _yaw_to_ball], dtype=np.float32)

                elif self._aligned_with_goal:
                    _push_yaw_vel = 0.0
                    if self._spawn_yaw is not None:
                        _sfwd_p = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                        _push_curr_yaw = float(np.arctan2(float(_sfwd_p[1]), float(_sfwd_p[0])))
                        _push_target   = float(self._wrap_to_pi(self._spawn_yaw + np.pi / 2))
                        _push_err      = float(self._wrap_to_pi(_push_curr_yaw - _push_target))
                        _push_yaw_vel  = float(np.clip(-_push_err * STEER_KP, -1.0, 1.0))
                    goal_vel = np.array([APPROACH_SPEED, 0.0, _push_yaw_vel], dtype=np.float32)
                    _err = None
                    if _ball_body_az is not None:
                        _err = float(self._wrap_to_pi(_ball_body_az))
                    elif self._spawn_yaw is not None:
                        _err = float(self._wrap_to_pi(_push_curr_yaw - _push_target))
                    if _err is not None and abs(_err) > 3 * ALIGN_THRESHOLD_RAD:
                        self._aligned_with_goal = False
                        self._orbiting = False

                else:
                    if self._role != 'attacker':
                        goal_vel = self._decide_goal_vel(
                            ball_raw, cur_head_yaw, self._role, rwp, orientation_quat_inv)
                    else:
                        if ball_dist is not None:
                            if ball_dist <= ORBIT_ENGAGE_M + 0.3:
                                self._orbiting = True
                            if ball_dist > SLOW_DIST_M:
                                self._orbiting = False

                        if not self._orbiting:
                            goal_vel = self._decide_goal_vel(
                                ball_raw, cur_head_yaw, self._role, rwp, orientation_quat_inv)
                            if _ball_body_az is not None and _goal_body_az is not None:
                                goal_vel[1] = float(np.clip(
                                    float(self._wrap_to_pi(_ball_body_az - _goal_body_az))
                                    * APPROACH_LATERAL_KP, -0.3, 0.3))
                        else:
                            _yaw_to_ball = float(np.clip(
                                (cur_head_yaw + (ball_raw[1] if ball_raw is not None else 0.0))
                                * STEER_KP, -1.0, 1.0))
                            _vx_orbit = float(np.clip(
                                ((ball_dist or ORBIT_RADIUS_M) - ORBIT_RADIUS_M)
                                * ORBIT_RADIAL_KP,
                                -FOLLOW_FORWARD_SPEED, FOLLOW_FORWARD_SPEED))
                            _sfwd2 = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                            _curr_yaw = float(np.arctan2(float(_sfwd2[1]), float(_sfwd2[0])))
                            _goal_target_yaw = (float(self._wrap_to_pi(self._spawn_yaw + np.pi / 2))
                                                if self._spawn_yaw is not None else 0.0)
                            _expected_goal_body = float(self._wrap_to_pi(_goal_target_yaw - _curr_yaw))
                            _goal_in_right_dir = (_goal_body_az is not None and
                                                  abs(float(self._wrap_to_pi(
                                                      _goal_body_az - _expected_goal_body)))
                                                  < np.deg2rad(45))
                            if _goal_in_right_dir:
                                _baz = _ball_body_az if _ball_body_az is not None else cur_head_yaw
                                _vy = float(np.sign(
                                    float(self._wrap_to_pi(_baz - _goal_body_az)))) * ORBIT_SPEED
                                if _vy == 0.0:
                                    _vy = -ORBIT_SPEED
                            else:
                                _arc = float(self._wrap_to_pi(_curr_yaw - _goal_target_yaw))
                                _vy = float(np.sign(_arc)) * ORBIT_SPEED if _arc != 0.0 else -ORBIT_SPEED
                            goal_vel = np.array([_vx_orbit, _vy, _yaw_to_ball], dtype=np.float32)

                            _aligned = False
                            if (self._ball_world_pos is not None
                                    and self._goal_world_pos is not None and rwp is not None):
                                _bl = orientation_quat_inv.apply(self._ball_world_pos - rwp)
                                _gl = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                                _aligned = (abs(float(self._wrap_to_pi(
                                    float(np.arctan2(_bl[1], _bl[0]))
                                    - float(np.arctan2(_gl[1], _gl[0])))))
                                    < ALIGN_THRESHOLD_RAD)
                            elif _ball_body_az is not None and _goal_in_right_dir:
                                _aligned = (abs(float(self._wrap_to_pi(
                                    _ball_body_az - _goal_body_az))) < ALIGN_THRESHOLD_RAD)
                            elif self._spawn_yaw is not None:
                                _yaw_err  = abs(float(self._wrap_to_pi(_curr_yaw - _goal_target_yaw)))
                                _ball_ahead = abs(cur_head_yaw) < np.deg2rad(15)
                                _aligned = (_yaw_err < ALIGN_THRESHOLD_RAD and _ball_ahead)
                            if _aligned:
                                self._aligned_with_goal = True
                                self._orbiting = False
                                goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                # head
                if ENABLE_HEAD_TRACKING:
                    self._track_head(ball_raw, cur_head_yaw, cur_head_pitch)

                # policy
                gait_phase_features = self._get_gait_phase_features()
                observation = np.concatenate([
                    scaled_joint_pos, scaled_joint_vel, scaled_previous_action,
                    scaled_and_clipped_ang_vel, goal_vel, gait_phase_features, projected_gravity
                ])
                observation = np.clip(
                    np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0),
                    -10.0, 10.0)

                with torch.no_grad():
                    obs_tensor  = torch.tensor(
                        observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor, next_policy_hidden = self.policy(obs_tensor, self.policy_hidden)
                nn_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                target_joint_positions = self.joint_nominal_position + self.scaling_factor * nn_action
                target_joint_positions_degrees = np.rad2deg(target_joint_positions)
                if ENABLE_HEAD_TRACKING:
                    target_joint_positions_degrees[self.HEAD_YAW_IDX]   = np.rad2deg(self._head_yaw_target)
                    target_joint_positions_degrees[self.HEAD_PITCH_IDX] = np.rad2deg(self._head_pitch_target)

                motors   = self.ROBOT_MOTORS[self._model_name]
                msg_list: list[str] = []
                for motor, tjp in zip(motors, target_joint_positions_degrees, strict=False):
                    msg_list.append(f'({motor} {tjp:.2f} 0.0 {self.p_gain:.2f} {self.d_gain:.2f} 0.0)')

                self.previous_action = nn_action
                self.policy_hidden   = next_policy_hidden
                self._step_gait_manager()

                self._log_csv(game_time, play_mode, rwp, ball_visible,
                              ball_x, ball_y, ball_z, goal_x, goal_y,
                              self._team_side, self._score_left,
                              self._score_right, data_valid)

                action_msg = ''.join(msg_list)
                self._send_message(action_msg.encode())

            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception:', exc_info=e.__traceback__)
                break

        # agent_stopped event
        if self._csv_writer and self._csv_header_done:
            self._log_csv(0.0, self._play_mode or 'Unknown', None, False,
                          None, None, None, None, None,
                          self._team_side, self._score_left, self._score_right,
                          0, event='agent_stopped')

    # ================================================================ head tracking

    def _track_head(self, ball, cur_yaw, cur_pitch):
        if ball is not None:
            _, az, el = ball
            self._head_yaw_target   = cur_yaw   + HEAD_YAW_SIGN   * HEAD_YAW_KP   * az
            self._head_pitch_target = cur_pitch + HEAD_PITCH_SIGN * HEAD_PITCH_KP * el
            self._head_scan_angle   = self._head_yaw_target
        elif ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            self._head_scan_angle += self._search_dir * HEAD_SWEEP_RAD_PER_CYCLE
            _sweep_limit = np.deg2rad(80.0)
            if   self._head_scan_angle >  _sweep_limit: self._search_dir = -1.0
            elif self._head_scan_angle < -_sweep_limit: self._search_dir =  1.0
            self._head_yaw_target   = self._head_scan_angle
            self._head_pitch_target = np.deg2rad(-40.0)
        self._head_yaw_target   = float(np.clip(self._head_yaw_target,   *HEAD_YAW_LIMIT_RAD))
        self._head_pitch_target = float(np.clip(self._head_pitch_target, *HEAD_PITCH_LIMIT_RAD))

    # ================================================================ networking

    def _send_message(self, msg: bytes | bytearray) -> None:
        self._sock.send(len(msg).to_bytes(4, byteorder='big') + msg)

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
        """Flat sensor parse (joints/IMU/quat) — identical to nn_client.py."""
        result = {}
        top_level_pattern = re.compile(r'\((\w+)((?:\s*\([^()]*\))*)\)')
        for tag, inner in top_level_pattern.findall(s):
            items = re.findall(r'\(\s*(\w+)((?:\s+[^()]+)+)\)', inner)
            group = {}
            for key, vals in items:
                tokens = vals.strip().split()
                parsed_vals = []
                for t in tokens:
                    try:
                        parsed_vals.append(float(t))
                    except ValueError:
                        parsed_vals.append(t)
                group[key] = parsed_vals[0] if len(parsed_vals) == 1 else parsed_vals
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(group)
                else:
                    result[tag] = [result[tag], group]
            else:
                result[tag] = group
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Thesis experiment client for RoboCup MuJoCo simulation.')
    robots = list(ThesisClient.ROBOT_MOTORS.keys())
    parser.add_argument('-s', '--host',       type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port',       type=int, default=60000)
    parser.add_argument('-t', '--team',       type=str, default='Test')
    parser.add_argument('-n', '--player_no',  type=int, default=1)
    parser.add_argument('-r', '--robot',      type=str, default=robots[0], choices=robots)
    parser.add_argument('--ready-file',       type=str, default='formation_ready.txt')
    parser.add_argument('--debug-perception', action='store_true',
                        help='Write the first complete perception message to a file.')
    parser.add_argument('--default-role',     type=str, default='attacker',
                        choices=['attacker', 'supporter'])

    args = parser.parse_args()

    def _shutdown(sig: int, frame: FrameType | None) -> None:
        client.shutdown()

    client = ThesisClient(
        host=args.host,
        port=args.port,
        team=args.team,
        player_no=args.player_no,
        model_name=args.robot,
        default_role=args.default_role,
        ready_file=args.ready_file,
        debug_perception=args.debug_perception,
    )
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    client.run()
