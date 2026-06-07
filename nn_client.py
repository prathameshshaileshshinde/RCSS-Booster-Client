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
#  CONFIG  -  beginner-friendly knobs.
# ============================================================================

# Vision / head tracking
ENABLE_HEAD_TRACKING     = True
HEAD_YAW_KP              = 0.6
HEAD_PITCH_KP            = 0.6
HEAD_YAW_SIGN            = +1
HEAD_PITCH_SIGN          = -1
HEAD_YAW_LIMIT_RAD       = (-np.pi/2, np.pi/2)        # ±90°
HEAD_PITCH_LIMIT_RAD     = (np.deg2rad(-20.0), np.deg2rad(70.0))  # -20° to 70°
HEAD_SWEEP_RAD_PER_CYCLE = np.deg2rad(4.0)             # 4° per cycle

# Ball following
ENABLE_BALL_FOLLOWING = True
FOLLOW_FORWARD_SPEED  = 1.0
STEER_KP              = 1.0 / (np.pi / 4)  # radians -> yaw goal velocity (π/4 rad = 45°)

# Ball persistence
BALL_CLOSE_DIST_M       = 2.0
BALL_CLOSE_TIMER_CYCLES = 150
BALL_SMOOTH_ALPHA       = 0.25
BALL_RESET_NEW_DIST_M   = 5.0
BALL_RESET_OLD_DIST_M   = 3.0

# Search
ENABLE_SEARCH    = True
LOST_BALL_CYCLES = 30
SEARCH_YAW_SPEED = 0.6

# Multi-robot Voronoi role assignment
# The ball is in my Voronoi cell  <=>  I am closer to it than any teammate.
# Closest robot  -> ATTACKER (goes for ball).
# Everyone else  -> SUPPORTER (takes support position behind ball).
TEAMMATE_STALE_CYCLES = 150   # forget teammate pos after N cycles without seeing them
SUPPORT_DIST_M        = 6.0   # supporter stands this far from ball (toward own goal)

# CSV logging
ENABLE_CSV_LOGGING = True
CSV_EVERY_N_CYCLES = 5
CSV_DIR            = "CSV"

# ============================================================================

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
ch.setLevel(logging.INFO)
logging.basicConfig(handlers=[ch], level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Client:
    BEAM_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
        1:  (27.5,  0.0, 0), 2:  (22.0, 12.0, 0), 3:  (22.0,  4.0, 0),
        4:  (22.0, -4.0, 0), 5:  (22.0,-12.0, 0), 6:  (15.0,  0.0, 0),
        7:  ( 4.0, 16.0, 0), 8:  (11.0,  6.0, 0), 9:  (11.0, -6.0, 0),
        10: ( 4.0,-16.0, 0), 11: ( 7.0,  0.0, 0),
    }

    ROBOT_MOTORS: ClassVar[Mapping[str, tuple[str, ...]]] = {
        'ant': ('l4e1','l4e2','l1e1','l1e2','l2e1','l2e2','l3e1','l3e2'),
        'T1':  ('he1','he2','lae1','lae2','lae3','lae4','rae1','rae2','rae3','rae4',
                'te1','lle1','lle2','lle3','lle4','lle5','lle6',
                'rle1','rle2','rle3','rle4','rle5','rle6'),
    }

    HEAD_YAW_IDX   = 0
    HEAD_PITCH_IDX = 1

    def __init__(self, host, port, team, player_no, model_name=None, default_role='attacker'):
        self._host       = host
        self._port       = port
        self._model_name = 'ant' if model_name is None else model_name
        self._team       = team
        self._player_no  = player_no

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
        self._goal_world_pos    = None

        # multi-robot
        self._teammate_world_pos = {}   # player_no -> world np.ndarray
        self._teammate_last_seen = {}   # player_no -> cycle
        self._robot_world_pos    = None
        self._default_role       = default_role   # role when no teammate visible
        self._role               = default_role

        # misc
        self._cycle      = 0
        self._log_cycle  = 0
        self._csv_writer = None
        self._csv_file   = None

        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def run(self):
        logger.info('Connecting to %s:%d...', self._host, self._port)
        try:
            self._sock.connect((self._host, self._port))
        except ConnectionRefusedError:
            logger.error('Connection refused.')
            return
        t = threading.Thread(target=self._action_loop)
        t.start()
        t.join()
        if self._csv_file:
            self._csv_file.close()
        self._sock.close()

    def shutdown(self):
        self._sock.shutdown(socket.SHUT_RDWR)

    # ------------------------------------------------------------------ policy
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
        self._goal_world_pos   = None
        self._teammate_world_pos = {}
        self._teammate_last_seen = {}
        self._robot_world_pos    = None
        self._role               = self._default_role

    @staticmethod
    def _wrap_to_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _get_gait_phase_features(self):
        phase_tp1 = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt)
        return np.concatenate([np.sin(phase_tp1), np.cos(phase_tp1)]).astype(np.float32)

    def _step_gait_manager(self):
        self.gait_phase = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt).astype(np.float32)

    # ------------------------------------------------------------------ vision
    @staticmethod
    def _polar_to_cartesian(dist, az_rad, el_rad):
        """Input angles must be in radians (already converted by _parse_ball/_parse_players)."""
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
        # Convert server degrees to radians on the way in
        return (float(m.group(1)),
                float(np.deg2rad(float(m.group(2)))),
                float(np.deg2rad(float(m.group(3))))) if m else None

    def _parse_goal(self, msg):
        m1 = re.search(r'\(G1R\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\)\)', msg)
        m2 = re.search(r'\(G2R\s+\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+-?[\d.]+\)\)', msg)
        xs, ys = [], []
        for m in (m1, m2):
            if m:
                d, az = float(m.group(1)), np.deg2rad(float(m.group(2)))
                xs.append(d*np.cos(az)); ys.append(d*np.sin(az))
        return (float(np.mean(xs)), float(np.mean(ys))) if xs else (None, None)

    def _parse_players(self, msg):
        """Parse (P ...) S-expression blocks -> list of player dicts.

        Each dict: {team, id, distance, azimuth, elevation}
        S-expression format:
            (P (team <name>) (id <n>) (head (pol <dist> <az> <el>)) ...)
        Polar values are camera-relative (head frame), degrees.
        """
        players = []
        for m in re.finditer(r'\(P\s+\(team\s+(\S+)\)\s+\(id\s+(\d+)\)', msg):
            team, pid = m.group(1), int(m.group(2))
            block = self._extract_balanced(msg, m.start())
            pols  = re.findall(r'\(pol\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)', block)
            if not pols: continue
            arr = np.array(pols, dtype=np.float32)
            d, az, el = arr.mean(axis=0)
            # Convert server degrees to radians on the way in
            players.append({'team': team, 'id': pid,
                            'distance': float(d),
                            'azimuth':  float(np.deg2rad(az)),
                            'elevation': float(np.deg2rad(el))})
        return players

    # ------------------------------------------------------------------ head
    def _track_head(self, ball, cur_yaw, cur_pitch):
        if ball is not None:
            _, az, el = ball
            self._head_yaw_target   = cur_yaw   + HEAD_YAW_SIGN   * HEAD_YAW_KP   * az
            self._head_pitch_target = cur_pitch + HEAD_PITCH_SIGN * HEAD_PITCH_KP * el
            self._head_scan_angle   = self._head_yaw_target
        elif ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            self._head_scan_angle += self._search_dir * HEAD_SWEEP_RAD_PER_CYCLE
            _sweep_limit = np.deg2rad(80.0)  # ±80° in radians
            if   self._head_scan_angle >  _sweep_limit: self._search_dir = -1.0
            elif self._head_scan_angle < -_sweep_limit: self._search_dir =  1.0
            self._head_yaw_target   = self._head_scan_angle
            self._head_pitch_target = np.deg2rad(-40.0)  # look down while searching
        self._head_yaw_target   = float(np.clip(self._head_yaw_target,   *HEAD_YAW_LIMIT_RAD))
        self._head_pitch_target = float(np.clip(self._head_pitch_target, *HEAD_PITCH_LIMIT_RAD))

    # ------------------------------------------------------------------ multi-robot
    def _update_teammates(self, players, robot_world_pos, robot_rotation):
        """Convert visible teammates from camera-polar to world coords and cache them.

        How it works:
          1. Filter for same-team, different-player entries in the (P ...) blocks.
          2. Convert polar (dist, az, el) -> local Cartesian -> world frame.
          3. Store in _teammate_world_pos keyed by player_no.
          4. Expire entries older than TEAMMATE_STALE_CYCLES.
        """
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
        """Voronoi role: am I the closest robot to the ball?

        A robot's Voronoi cell contains all points closer to it than to any
        other robot.  The ball is in my cell  <=>  my distance to the ball
        is less than or equal to every teammate's distance.

        When no teammate is visible the robot falls back to _default_role
        (set at launch via --default-role). This prevents two robots both
        rushing the ball before they can see each other.

        Returns 'attacker' if closest, 'supporter' otherwise.
        """
        if self._ball_world_pos is None or robot_world_pos is None:
            return self._default_role
        if not self._teammate_world_pos:
            # No teammate in view — hold the assigned starting role until we
            # can actually see a teammate and compare distances properly.
            return self._default_role

        ball_xy = self._ball_world_pos[:2]
        my_dist = float(np.linalg.norm(ball_xy - robot_world_pos[:2]))
        for tw in self._teammate_world_pos.values():
            if float(np.linalg.norm(ball_xy - tw[:2])) < my_dist:
                return 'supporter'
        return 'attacker'

    def _supporter_goal_vel(self, robot_world_pos, orientation_inv):
        """Navigate to a support position SUPPORT_DIST_M behind the ball.

        'Behind the ball' = in the direction from ball toward field centre (0,0),
        which approximates 'toward own goal'.  This keeps the supporter out of
        the attacker's path and ready to receive a pass.

        Returns None when world data is unavailable (caller uses attacker fallback).
        """
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
        body_ang = float(np.arctan2(dy, dx))  # radians; STEER_KP = 1/(π/4)
        yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
        return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)

    def _spin_to_search(self):
        """Spin the body on the spot to scan for the ball.

        Direction matches the head sweep direction (_search_dir) so the
        head and body rotate together.  Called when the ball has been
        missing from direct camera vision for more than LOST_BALL_CYCLES
        cycles AND the world model has no stored position to fall back on.
        """
        return np.array([0.0, 0.0, self._search_dir * SEARCH_YAW_SPEED], dtype=np.float32)

    def _decide_goal_vel(self, ball_raw, cur_head_yaw, role, robot_world_pos, orientation_inv,
                         ball_local_xy=None):
        """Body goal velocity based on role.

        Priority order for the attacker:
          1. Direct vision  (ball_raw)        — most accurate
          2. World model / persistence        — ball_local_xy from caller
          3. Spin to search                   — ball completely lost

        SUPPORTER: move to support position; falls back to attacker logic
                   if world model not yet available.
        """
        if not ENABLE_BALL_FOLLOWING:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if role == 'supporter':
            sv = self._supporter_goal_vel(robot_world_pos, orientation_inv)
            if sv is not None:
                return sv

        # --- Priority 1: direct camera vision ---
        if ball_raw is not None:
            _, az, _ = ball_raw           # az already in radians from _parse_ball
            body_ang = cur_head_yaw + az
            yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
            return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)

        # --- Priority 2: world model / persistence ---
        if ball_local_xy is not None:
            bx, by = ball_local_xy
            az_est   = float(np.arctan2(by, bx))   # radians
            body_ang = cur_head_yaw + az_est
            yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
            return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)

        # --- Priority 3: ball completely lost — spin to search ---
        if ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            return self._spin_to_search()

        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------ CSV
    def _open_csv(self):
        if not ENABLE_CSV_LOGGING: return
        os.makedirs(CSV_DIR, exist_ok=True)
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(CSV_DIR, f'robot_log_{self._team}_p{self._player_no}_{ts}.csv')
        self._csv_file   = open(path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'game_time','play_mode','player_no','team',
            'robot_world_x','robot_world_y','robot_world_z',
            'role','ball_visible',
            'ball_rel_x','ball_rel_y','ball_rel_z',
            'ball_world_x','ball_world_y','ball_world_z',
            'goal_rel_x','goal_rel_y',
            'nearest_teammate_id','nearest_teammate_world_x',
            'nearest_teammate_world_y','nearest_teammate_dist_to_ball',
        ])
        logger.info('Logging to %s', path)

    def _log_csv(self, game_time, play_mode, rwp, ball_visible,
                 ball_x, ball_y, ball_z, goal_x, goal_y):
        if not ENABLE_CSV_LOGGING or not self._csv_writer: return
        if self._cycle % CSV_EVERY_N_CYCLES != 0: return
        bwp = self._ball_world_pos
        nid = nwx = nwy = ndtb = ''
        if self._teammate_world_pos and bwp is not None:
            bxy = bwp[:2]
            best = min(self._teammate_world_pos,
                       key=lambda pid: np.linalg.norm(bxy - self._teammate_world_pos[pid][:2]))
            tw = self._teammate_world_pos[best]
            nid = best; nwx = round(float(tw[0]),3); nwy = round(float(tw[1]),3)
            ndtb = round(float(np.linalg.norm(bxy - tw[:2])),3)
        self._csv_writer.writerow([
            round(game_time,3), play_mode, self._player_no, self._team,
            round(float(rwp[0]),3) if rwp is not None else '',
            round(float(rwp[1]),3) if rwp is not None else '',
            round(float(rwp[2]),3) if rwp is not None else '',
            self._role, 1 if ball_visible else 0,
            round(ball_x,3) if ball_x is not None else '',
            round(ball_y,3) if ball_y is not None else '',
            round(ball_z,3) if ball_z is not None else '',
            round(float(bwp[0]),3) if bwp is not None else '',
            round(float(bwp[1]),3) if bwp is not None else '',
            round(float(bwp[2]),3) if bwp is not None else '',
            round(goal_x,3) if goal_x is not None else '',
            round(goal_y,3) if goal_y is not None else '',
            nid, nwx, nwy, ndtb,
        ])

    # ------------------------------------------------------------------ main loop
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

        logger.info('Initializing agent...')
        self._send_message(f'(init {self._model_name} {self._team} {self._player_no})'.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                if not self._has_beamed:
                    bp = self.BEAM_POSES[self._player_no]
                    self._send_message(f'(beam {bp[0]} {bp[1]} {bp[2]})'.encode())
                    self._has_beamed = True
                    self._init_policy_runtime_state()
                    continue

                self._cycle += 1; self._log_cycle += 1
                msg = perception_msg.decode()
                perception_data = self.parse_sensor_string(msg)

                # joints
                jpd = np.array([h['ax'] for h in perception_data['HJ']], dtype=np.float32)
                jp  = np.deg2rad(jpd).astype(np.float32)
                jvd = np.array([h['vx'] for h in perception_data['HJ']], dtype=np.float32)
                jv  = np.deg2rad(jvd).astype(np.float32)
                cur_head_yaw   = float(jp[self.HEAD_YAW_IDX])   # radians (from jp)
                cur_head_pitch = float(jp[self.HEAD_PITCH_IDX])  # radians
                s_jp  = (jp - self.joint_nominal_position) / 3.14
                s_jv  = jv / 100.0
                s_pa  = self.previous_action / 10.0

                av = np.deg2rad(np.array(perception_data['GYR']['rt'], dtype=np.float32))
                s_av = np.clip(av / 50.0, -1.0, 1.0).astype(np.float32)

                # orientation
                q = np.array(perception_data['quat']['q'], dtype=np.float32)
                rot     = R.from_quat([q[1], q[2], q[3], q[0]])
                rot_inv = rot.inv()
                grav    = rot_inv.apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

                # robot world pos
                tm = re.search(
                    r'\(pos\s+\(n\s+torso_pos\)\s+\(pos\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)\)', msg)
                rwp = (np.array([float(tm.group(1)), float(tm.group(2)), float(tm.group(3))], dtype=np.float32)
                       if tm else None)
                self._robot_world_pos = rwp

                # game state
                gsm = re.search(r'\(GS\s+\(t\s+([\d.]+)\)\s+\(pm\s+(\w+)\)', msg)
                game_time = float(gsm.group(1)) if gsm else 0.0
                play_mode = gsm.group(2)         if gsm else 'Unknown'

                # update teammate world-position cache from (P ...) blocks
                players = self._parse_players(msg)
                self._update_teammates(players, rwp, rot)

                # ball
                ball_raw     = self._parse_ball(msg)
                ball_visible = ball_raw is not None
                if ball_raw is not None:
                    ball_dist = ball_raw[0]
                    ball_x, ball_y, ball_z = self._polar_to_cartesian(*ball_raw)
                    self._cycles_since_ball = 0
                else:
                    ball_dist = ball_x = ball_y = ball_z = None
                    self._cycles_since_ball += 1

                # goal
                goal_x, goal_y = self._parse_goal(msg)
                if goal_x is not None and rwp is not None:
                    self._goal_world_pos = rwp + rot.apply(
                        np.array([goal_x, goal_y, 0.0], dtype=np.float32))
                if goal_x is None and self._goal_world_pos is not None and rwp is not None:
                    gl = rot_inv.apply(self._goal_world_pos - rwp)
                    goal_x, goal_y = float(gl[0]), float(gl[1])

                # ball persistence layer 0: fresh vision
                if ball_x is not None:
                    self._last_ball_x = ball_x; self._last_ball_y = ball_y; self._last_ball_z = ball_z
                    if ball_dist is not None and ball_dist < BALL_CLOSE_DIST_M:
                        self._ball_close_timer = BALL_CLOSE_TIMER_CYCLES
                    if rwp is not None:
                        self._ball_world_pos = rwp + rot.apply(
                            np.array([ball_x, ball_y, ball_z], dtype=np.float32))
                else:
                    # layer 1: world model
                    if self._ball_world_pos is not None and rwp is not None:
                        bl = rot_inv.apply(self._ball_world_pos - rwp)
                        ball_x, ball_y, ball_z = float(bl[0]), float(bl[1]), float(bl[2])
                        ball_dist = float(np.linalg.norm(bl))
                    # layer 2: close timer
                    elif self._last_ball_x is not None and self._ball_close_timer > 0:
                        ball_x, ball_y, ball_z = self._last_ball_x, self._last_ball_y, self._last_ball_z
                        ball_dist = float(np.sqrt(ball_x**2 + ball_y**2 + (ball_z or 0.0)**2))

                if self._ball_close_timer > 0:
                    self._ball_close_timer -= 1
                    if self._ball_close_timer == 0:
                        self._head_scan_angle = 0.0
                        self._smooth_ball_x = self._smooth_ball_y = None

                # ball-reset detection
                if ball_x is not None and self._smooth_ball_x is not None:
                    if (float(np.sqrt(ball_x**2+ball_y**2)) > BALL_RESET_NEW_DIST_M and
                            float(np.sqrt(self._smooth_ball_x**2+self._smooth_ball_y**2)) < BALL_RESET_OLD_DIST_M):
                        logger.info('[P%d] Ball reset detected — clearing state', self._player_no)
                        self._smooth_ball_x = self._smooth_ball_y = None
                        self._ball_world_pos = None
                        self._last_ball_x = self._last_ball_y = self._last_ball_z = None
                        self._ball_close_timer = 0; self._head_scan_angle = 0.0

                # exponential smoothing
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

                # periodic logging
                if self._log_cycle % 25 == 0:
                    src = ('vision' if ball_visible else
                           'world_model' if self._ball_world_pos is not None else 'last_known')
                    if ball_x is not None:
                        logger.info('[P%d] role=%-9s  Ball x=%.2f y=%.2f dist=%.2f  [%s]',
                                    self._player_no, self._role.upper(),
                                    ball_x, ball_y, ball_dist or 0.0, src)
                    else:
                        logger.info('[P%d] role=%-9s  Ball NOT visible', self._player_no, self._role.upper())
                    for pid, tw in self._teammate_world_pos.items():
                        age = self._cycle - self._teammate_last_seen.get(pid, self._cycle)
                        bxy = self._ball_world_pos[:2] if self._ball_world_pos is not None else None
                        dtb = f'{np.linalg.norm(bxy-tw[:2]):.1f}m' if bxy is not None else '?'
                        logger.info('  teammate P%d  world=(%.1f,%.1f)  dist_ball=%s  age=%dcyc',
                                    pid, tw[0], tw[1], dtb, age)

                # goal velocity
                # Pass persistence-resolved local position when direct vision is absent.
                # ball_x/ball_y are already in robot-local frame (from world model or
                # close-timer layers), so they can steer the body even without a live
                # camera hit on the ball.
                ball_local_xy = (
                    (ball_x, ball_y)
                    if (ball_raw is None and ball_x is not None)
                    else None
                )

                self.wait_until_walking = max(0, self.wait_until_walking - 1)
                if self.wait_until_walking > 0:
                    goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    goal_vel = self._decide_goal_vel(
                        ball_raw, cur_head_yaw, self._role, rwp, rot_inv,
                        ball_local_xy=ball_local_xy,
                    )

                # head
                if ENABLE_HEAD_TRACKING:
                    self._track_head(ball_raw, cur_head_yaw, cur_head_pitch)

                # policy
                gait_f = self._get_gait_phase_features()
                obs    = np.concatenate([s_jp, s_jv, s_pa, s_av, goal_vel, gait_f, grav])
                obs    = np.clip(np.nan_to_num(obs), -10.0, 10.0)

                with torch.no_grad():
                    obs_t  = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_t, next_h = self.policy(obs_t, self.policy_hidden)
                nn_action = act_t.squeeze(0).cpu().numpy().astype(np.float32)

                tj_deg = np.rad2deg(self.joint_nominal_position + self.scaling_factor * nn_action)
                if ENABLE_HEAD_TRACKING:
                    # Targets are in radians; motor command protocol uses degrees
                    tj_deg[self.HEAD_YAW_IDX]   = np.rad2deg(self._head_yaw_target)
                    tj_deg[self.HEAD_PITCH_IDX] = np.rad2deg(self._head_pitch_target)

                motors  = self.ROBOT_MOTORS[self._model_name]
                msg_out = ''.join(f'({m} {q:.2f} 0.0 {self.p_gain:.2f} {self.d_gain:.2f} 0.0)'
                                  for m, q in zip(motors, tj_deg, strict=False))

                self.previous_action = nn_action
                self.policy_hidden   = next_h
                self._step_gait_manager()

                self._log_csv(game_time, play_mode, rwp, ball_visible,
                              ball_x, ball_y, ball_z, goal_x, goal_y)
                self._send_message(msg_out.encode())

            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception:', exc_info=e.__traceback__)
                break

    # ------------------------------------------------------------------ networking
    def _send_message(self, msg):
        self._sock.send(len(msg).to_bytes(4, byteorder='big') + msg)

    def _receive_message(self):
        if self._sock.recv_into(self._rcv_buffer, nbytes=4, flags=socket.MSG_WAITALL) != 4:
            raise ConnectionResetError
        sz = int.from_bytes(self._rcv_buffer[:4], byteorder='big', signed=False)
        if sz > self._rcv_buffer_size:
            self._rcv_buffer_size = sz; self._rcv_buffer = bytearray(sz)
        if self._sock.recv_into(self._rcv_buffer, nbytes=sz, flags=socket.MSG_WAITALL) != sz:
            raise ConnectionResetError
        return self._rcv_buffer[:sz]

    def parse_sensor_string(self, s):
        result = {}
        for tag, inner in re.compile(r'\((\w+)((?:\s*\([^()]*\))*)\)').findall(s):
            group = {}
            for key, vals in re.findall(r'\(\s*(\w+)((?:\s+[^()]+)+)\)', inner):
                tokens = vals.strip().split()
                parsed = []
                for t in tokens:
                    try:    parsed.append(float(t))
                    except: parsed.append(t)
                group[key] = parsed[0] if len(parsed) == 1 else parsed
            if tag in result:
                result[tag] = ([result[tag]] if not isinstance(result[tag], list) else result[tag]) + [group]
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
    parser.add_argument('--default-role',    type=str, default='attacker',
                        choices=['attacker', 'supporter'],
                        help='Starting role when no teammate is visible (default: attacker)')
    args = parser.parse_args()

    client = Client(args.host, args.port, args.team, args.player_no, args.robot,
                    default_role=args.default_role)

    def signal_handler(sig, frame):
        del sig, frame
        client.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    client.run()
