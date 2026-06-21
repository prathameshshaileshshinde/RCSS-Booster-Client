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
STOP_DIST_M           = 1.0                 # orbit radius / stop distance (horizontal)
SLOW_DIST_M           = 2.0                 # start slowing at this distance (linear ramp)
ORBIT_ENGAGE_M        = 1.5                 # switch from approach to orbit below this distance
ORBIT_RADIUS_M        = STOP_DIST_M        # maintain this radius while orbiting
ORBIT_SPEED           = 0.3                 # lateral sidestep speed during orbit
ORBIT_RADIAL_KP       = 0.5                 # gain to hold 1 m from ball
ALIGN_TOL_RAD         = np.deg2rad(5)       # stop orbiting when ball-goal angle < this
ALIGN_THRESHOLD_RAD   = np.deg2rad(8)      # stop condition for spawn-yaw fallback (8°)
STEER_KP              = 1.0 / (np.pi / 4)  # radians -> yaw goal velocity (π/4 rad = 45°)
APPROACH_SPEED        = 0.7                 # gentle push speed when walking ball toward goal
APPROACH_LATERAL_KP   = 0.4                 # vy gain to keep ball centred during approach
APPROACH_GOAL_LAT_KP  = 0.0                 # lateral goal-centre correction (disabled)

# Play mode
KICKOFF_STAND_CYCLES   = 150  # cycles to hold formation after kickoff (~3 s at 50 Hz)
FORMATION_ARRIVE_DIST  = 0.5  # metres — stop walking to formation when this close to target
DR_WALK_SPEED_MPS      = 0.83  # calibrated to real walk speed

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

    # Sideline spawn positions — each robot beams to the sideline directly above/below
    # its BEAM_POSES target, facing inward (-90 = face toward -y, +90 = face toward +y).
    # This means the robot only needs to walk straight forward to reach its position.
    SIDELINE_POSES: ClassVar[Mapping[int, tuple[float, float, float]]] = {
        # Players 2&3 share target x=22 (top sideline) — stagger by 2 m so they don't clash at spawn.
        # Players 4&5 share target x=22 (bottom sideline) — same fix.
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

    def __init__(self, host, port, team, player_no, model_name=None, default_role='attacker',
                 ready_file=None):
        self._host       = host
        self._port       = port
        self._model_name = 'ant' if model_name is None else model_name
        self._team       = team
        self._player_no  = player_no
        self._ready_file = ready_file   # path to write player no when formation reached

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
        self._spawn_yaw         = None   # robot world-frame yaw recorded right after beam
        self._goal_world_dir    = None   # world-frame unit vector toward opponent goal
        self._goal_dir_init     = False  # True once bootstrapped
        self._goal_dir_trusted  = False  # True once a both-posts observation was used
        self._goal_world_pos    = None

        # multi-robot
        self._teammate_world_pos = {}   # player_no -> world np.ndarray
        self._teammate_last_seen = {}   # player_no -> cycle
        self._robot_world_pos    = None
        self._dr_pos             = None   # dead-reckoned XY world position [x, y]
        self._dr_walk_dir        = None   # unit vector in world frame for DR walk
        self._default_role       = default_role   # role when no teammate visible
        self._role               = default_role

        # play mode tracking
        self._play_mode         = None   # None until first server message received
        self._prev_play_mode    = None
        self._team_side         = None   # 'left' or 'right', filled from GS tl/tr
        self._formation_phase     = True   # walk to formation before chasing ball
        self._at_formation_cycles = 0      # cycles spent within FORMATION_ARRIVE_DIST
        self._formation_arrived   = False  # True once we've written to the ready file
        self._post_kickoff_wait   = 0      # countdown cycles after kickoff before chasing ball

        # misc
        self._cycle             = 0
        self._log_cycle         = 0
        self._csv_writer        = None
        self._csv_file          = None
        self._csv_header_done   = False
        self._aligned_with_goal = False  # True once robot-ball-goal are collinear; stays True

        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

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
        self._play_mode          = None   # None until first server message received
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

    def _parse_goal(self, msg, cur_head_yaw=0.0):
        """Check all four posts (G1R,G2R,G1L,G2L), average each side, then pick
        whichever side is most in front in body frame — that's the opponent goal."""
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
        # Pick the goal most forward in body frame (head→body rotation)
        cos_h, sin_h = float(np.cos(cur_head_yaw)), float(np.sin(cur_head_yaw))
        def _body_x(side):
            hx, hy = groups[side]
            return hx * cos_h - hy * sin_h
        best = max(groups, key=_body_x)
        return groups[best]

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

    def _walk_to_formation(self, robot_world_pos, orientation_inv):
        """Steer toward this player's formation position during BeforeKickOff.

        Uses the same BEAM_POSES XY coordinates as a walking target instead of
        teleporting.  Returns [0,0,0] once within FORMATION_ARRIVE_DIST metres.
        """
        target_x, target_y, _ = self.BEAM_POSES[self._player_no]
        target_xy  = np.array([target_x, target_y], dtype=np.float32)
        robot_xy   = robot_world_pos[:2]
        offset     = target_xy - robot_xy
        dist       = float(np.linalg.norm(offset))

        if dist < FORMATION_ARRIVE_DIST:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Rotate world-frame offset into body frame
        offset_3d = np.array([offset[0], offset[1], 0.0], dtype=np.float32)
        local     = orientation_inv.apply(offset_3d)
        dx, dy    = float(local[0]), float(local[1])

        body_ang = float(np.arctan2(dy, dx))
        yaw_vel  = float(np.clip(body_ang * STEER_KP, -1.0, 1.0))
        # Slow down as we get close; cap at full forward speed
        speed    = float(np.clip(dist * 0.5, 0.2, FOLLOW_FORWARD_SPEED))

        if self._log_cycle % 25 == 0:
            logger.info('[P%d] → formation (%.1f, %.1f)  dist=%.2fm  yaw_err=%.1f°',
                        self._player_no, target_x, target_y, dist, np.rad2deg(body_ang))

        return np.array([speed, 0.0, yaw_vel], dtype=np.float32)

    def _decide_goal_vel(self, ball_raw, cur_head_yaw, role, robot_world_pos, orientation_inv):
        """Body goal velocity based on role.

        ATTACKER : walk toward the ball (head_yaw + camera_azimuth steering).
        SUPPORTER: move to support position; fall back to attacker if world
                   model not yet available.
        """
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
                # Approach phase: walk straight toward ball
                return np.array([FOLLOW_FORWARD_SPEED, 0.0, yaw_vel], dtype=np.float32)
            else:
                # Orbit phase: face the ball with yaw, correct radius with vx, strafe with vy
                vx = float(np.clip((dist - ORBIT_RADIUS_M) * ORBIT_RADIAL_KP,
                                   -FOLLOW_FORWARD_SPEED, FOLLOW_FORWARD_SPEED))
                return np.array([vx, ORBIT_SPEED, yaw_vel], dtype=np.float32)
        if ENABLE_SEARCH and self._cycles_since_ball > LOST_BALL_CYCLES:
            return np.array([0.0, 0.0, self._search_dir * SEARCH_YAW_SPEED], dtype=np.float32)
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
            'game_time','play_mode','player_no','team','team_side',
            'robot_world_x','robot_world_y','robot_world_z',
            'role','ball_visible',
            'ball_rel_x','ball_rel_y','ball_rel_z',
            'ball_world_x','ball_world_y','ball_world_z',
            'goal_rel_x','goal_rel_y',
            'nearest_teammate_id','nearest_teammate_world_x',
            'nearest_teammate_world_y','nearest_teammate_dist_to_ball',
            'event',
        ])
        self._csv_header_done = True
        logger.info('Logging to %s', path)

    def _log_csv(self, game_time, play_mode, rwp, ball_visible,
                 ball_x, ball_y, ball_z, goal_x, goal_y, event=''):
        if not ENABLE_CSV_LOGGING or not self._csv_writer: return
        # Events (game_init, kickoff) are always written; normal rows obey the N-cycle gate
        if event == '' and self._cycle % CSV_EVERY_N_CYCLES != 0: return
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
            self._team_side or '',
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
            event,
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
        init_msg = f'(init {self._model_name} {self._team} {self._player_no})'
        self._send_message(init_msg.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                if not self._has_beamed:
                    # Beam to the sideline — establishes torso_pos so we can navigate precisely.
                    # Robot then walks from sideline to its BEAM_POSES formation position.
                    self._has_beamed = True
                    self._init_policy_runtime_state()
                    sx, sy, sa = self.SIDELINE_POSES[self._player_no]
                    # Init dead-reckoning from known beam position
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
                perception_data = self.parse_sensor_string(perception_msg_str)

                # joints
                joint_pos_degrees = np.array([h['ax'] for h in perception_data['HJ']], dtype=np.float32)
                joint_pos  = np.deg2rad(joint_pos_degrees).astype(np.float32)
                joint_vel_degrees = np.array([h['vx'] for h in perception_data['HJ']], dtype=np.float32)
                joint_vel  = np.deg2rad(joint_vel_degrees).astype(np.float32)
                cur_head_yaw   = float(joint_pos[self.HEAD_YAW_IDX])   # radians (from joint_pos)
                cur_head_pitch = float(joint_pos[self.HEAD_PITCH_IDX])  # radians
                scaled_joint_pos  = (joint_pos - self.joint_nominal_position) / 3.14
                scaled_joint_vel  = joint_vel / 100.0
                scaled_previous_action = self.previous_action / 10.0

                ang_vel = np.deg2rad(np.array(perception_data['GYR']['rt'], dtype=np.float32))
                scaled_and_clipped_ang_vel = np.clip(ang_vel / 50.0, -1.0, 1.0).astype(np.float32)

                # orientation
                orientation_quat_mj_convention = np.array(perception_data['quat']['q'], dtype=np.float32)
                rot     = R.from_quat([orientation_quat_mj_convention[1],
                                       orientation_quat_mj_convention[2],
                                       orientation_quat_mj_convention[3],
                                       orientation_quat_mj_convention[0]])
                orientation_quat_inv = rot.inv()
                projected_gravity    = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

                # Record body yaw on first cycle after beam — used as goal-direction proxy
                if self._spawn_yaw is None:
                    _sfwd = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                    self._spawn_yaw = float(np.arctan2(float(_sfwd[1]), float(_sfwd[0])))
                    logger.info('[P%d] spawn_yaw recorded: %.1f°', self._player_no, np.rad2deg(self._spawn_yaw))

                # robot world pos — update if server sends it; otherwise keep last known
                tm = re.search(
                    r'\(pos\s+\(n\s+torso_pos\)\s+\(pos\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\)\)', perception_msg_str)
                if tm:
                    rwp = np.array([float(tm.group(1)), float(tm.group(2)), float(tm.group(3))], dtype=np.float32)
                    self._robot_world_pos = rwp
                else:
                    rwp = self._robot_world_pos  # use last known position

                # Dead-reckoning fallback: when server doesn't provide torso_pos, use DR estimate
                if rwp is None and self._dr_pos is not None:
                    rwp = np.array(
                        [self._dr_pos[0], self._dr_pos[1], 0.0], dtype=np.float32)

                # game state
                gsm = re.search(
                    r'\(GS\s+\(t\s+([\d.]+)\)\s+\(pm\s+(\w+)\)'
                    r'(?:\s+\(tl\s+(\S+)\))?(?:\s+\(tr\s+(\S+)\))?',
                    perception_msg_str)
                game_time = float(gsm.group(1)) if gsm else 0.0
                play_mode = gsm.group(2)         if gsm else 'Unknown'
                if gsm and self._team_side is None:
                    tl = gsm.group(3); tr = gsm.group(4)
                    if tl and tl == self._team:
                        self._team_side = 'left'
                    elif tr and tr == self._team:
                        self._team_side = 'right'

                # play mode transition tracking
                self._prev_play_mode = self._play_mode
                self._play_mode      = play_mode

                # update teammate world-position cache from (P ...) blocks
                players = self._parse_players(perception_msg_str)
                self._update_teammates(players, rwp, rot)

                # ball
                ball_raw     = self._parse_ball(perception_msg_str)
                ball_visible = ball_raw is not None
                if ball_raw is not None:
                    ball_dist = ball_raw[0]
                    ball_x, ball_y, ball_z = self._polar_to_cartesian(*ball_raw)
                    self._cycles_since_ball = 0
                else:
                    ball_dist = ball_x = ball_y = ball_z = None
                    self._cycles_since_ball += 1

                # goal
                goal_x, goal_y = self._parse_goal(perception_msg_str, cur_head_yaw)
                # Head-frame azimuth — only valid when goal directly visible
                goal_cam_az = float(np.arctan2(goal_y, goal_x)) if goal_x is not None else None
                if goal_x is not None and rwp is not None:
                    # goal_x,y are HEAD frame — rotate to body frame before world transform
                    _ch, _sh = float(np.cos(cur_head_yaw)), float(np.sin(cur_head_yaw))
                    _gbx = goal_x * _ch - goal_y * _sh
                    _gby = goal_x * _sh + goal_y * _ch
                    self._goal_world_pos = rwp + rot.apply(np.array([_gbx, _gby, 0.0], dtype=np.float32))
                if goal_x is None and self._goal_world_pos is not None and rwp is not None:
                    gl = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                    goal_x, goal_y = float(gl[0]), float(gl[1])

                # Body-frame goal azimuth — used everywhere for orbit and alignment.
                # Computed from world model when available (smoothest), else from direct vision.
                _goal_body_az = None
                if goal_cam_az is not None:
                    _goal_body_az = float(self._wrap_to_pi(goal_cam_az + cur_head_yaw))
                if self._goal_world_pos is not None and rwp is not None:
                    _glb = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                    _goal_body_az = float(self._wrap_to_pi(float(np.arctan2(_glb[1], _glb[0]))))
                # Body-frame ball azimuth
                _ball_body_az = (float(self._wrap_to_pi(cur_head_yaw + ball_raw[1]))
                                 if ball_raw is not None else None)

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
                        bl = orientation_quat_inv.apply(self._ball_world_pos - rwp)
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

                # Release formation on kickoff (always, regardless of CSV logging)
                if (self._prev_play_mode == 'BeforeKickOff'
                        and play_mode != 'BeforeKickOff'):
                    self._formation_phase   = False
                    self._post_kickoff_wait = KICKOFF_STAND_CYCLES   # hold 3 s before chasing
                    logger.info('[P%d] Kickoff! Holding position for %d cycles then chasing ball.',
                                self._player_no, KICKOFF_STAND_CYCLES)

                # CSV event rows: game_init (first BeforeKickOff) and kickoff transition
                if ENABLE_CSV_LOGGING and self._csv_writer and self._csv_header_done:
                    if (self._prev_play_mode is None
                            and play_mode == 'BeforeKickOff'):
                        # Very first cycle — robots have just been placed, game not yet started
                        self._log_csv(game_time, play_mode, rwp, ball_visible,
                                      ball_x, ball_y, ball_z, goal_x, goal_y,
                                      event='game_init')
                        logger.info('[P%d] CSV: game_init logged (BeforeKickOff)', self._player_no)
                    elif (self._prev_play_mode == 'BeforeKickOff'
                          and play_mode != 'BeforeKickOff'):
                        # Log the kickoff transition row
                        self._log_csv(game_time, play_mode, rwp, ball_visible,
                                      ball_x, ball_y, ball_z, goal_x, goal_y,
                                      event='kickoff_start')
                        logger.info('[P%d] CSV: kickoff_start logged (→ %s)', self._player_no, play_mode)

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
                    # Alignment diagnostics: show which check path is available
                    if ball_raw is not None and ball_raw[0] <= ORBIT_ENGAGE_M:
                        if self._ball_world_pos is not None and self._goal_world_pos is not None:
                            btg = self._goal_world_pos[:2] - self._ball_world_pos[:2]
                            exp = float(np.arctan2(btg[1], btg[0]))
                            fwd = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                            yaw = float(np.arctan2(fwd[1], fwd[0]))
                            logger.info('[P%d] align PRIMARY  robot_yaw=%.1f°  expected=%.1f°  diff=%.1f°',
                                        self._player_no, np.rad2deg(yaw), np.rad2deg(exp),
                                        np.rad2deg(abs(float(self._wrap_to_pi(yaw - exp)))))
                        elif goal_cam_az is not None:
                            logger.info('[P%d] align FALLBACK-A  ball_az=%.1f°  goal_cam_az=%.1f°  diff=%.1f°',
                                        self._player_no, np.rad2deg(ball_raw[1]), np.rad2deg(goal_cam_az),
                                        np.rad2deg(abs(float(self._wrap_to_pi(ball_raw[1] - goal_cam_az)))))
                        else:
                            logger.info('[P%d] align NO-DATA  ball_world=%s  goal_world=%s  goal_cam_az=%s  goal_x=%s',
                                        self._player_no,
                                        'ok' if self._ball_world_pos is not None else 'NONE',
                                        'ok' if self._goal_world_pos is not None else 'NONE',
                                        f'{np.rad2deg(goal_cam_az):.1f}°' if goal_cam_az is not None else 'NONE',
                                        f'{goal_x:.2f}' if goal_x is not None else 'NONE')
                    for pid, tw in self._teammate_world_pos.items():
                        age = self._cycle - self._teammate_last_seen.get(pid, self._cycle)
                        bxy = self._ball_world_pos[:2] if self._ball_world_pos is not None else None
                        dtb = f'{np.linalg.norm(bxy-tw[:2]):.1f}m' if bxy is not None else '?'
                        logger.info('  teammate P%d  world=(%.1f,%.1f)  dist_ball=%s  age=%dcyc',
                                    pid, tw[0], tw[1], dtb, age)

                # goal velocity — debug: log perception once so we can see torso_pos format
                if self._cycle == 2:
                    logger.info('[P%d] PERCEPTION SAMPLE (full): %s', self._player_no, perception_msg_str[:1500])

                self.wait_until_walking = max(0, self.wait_until_walking - 1)

                # Update dead-reckoning position during formation walk
                if (self._formation_phase and self.wait_until_walking == 0
                        and self._dr_pos is not None and self._dr_walk_dir is not None):
                    _btx2, _bty2, _ = self.BEAM_POSES[self._player_no]
                    _dr_to_target = float(np.linalg.norm(
                        self._dr_pos - np.array([_btx2, _bty2], dtype=np.float32)))
                    if _dr_to_target > 0.0:
                        _step = DR_WALK_SPEED_MPS * self._policy_dt
                        if _step >= _dr_to_target:
                            # Would reach/overshoot — snap exactly to target
                            self._dr_pos = np.array([_btx2, _bty2], dtype=np.float32)
                        else:
                            self._dr_pos = (
                                self._dr_pos
                                + self._dr_walk_dir * _step)
                    if self._log_cycle % 25 == 0:
                        logger.info('[P%d] DR pos=(%.1f,%.1f) dist_to_target=%.1f',
                                    self._player_no,
                                    float(self._dr_pos[0]), float(self._dr_pos[1]),
                                    _dr_to_target)

                # diagnostic: log state every 25 cycles during formation
                if self._formation_phase and self._log_cycle % 25 == 0:
                    logger.info('[P%d] formation_phase  pm=%s  rwp=%s  wait=%d',
                                self._player_no, play_mode,
                                f'({rwp[0]:.1f},{rwp[1]:.1f})' if rwp is not None else 'None',
                                self.wait_until_walking)

                if self._formation_phase:
                    if self.wait_until_walking > 0:
                        goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    elif rwp is not None:
                        pass  # handled below
                    else:
                        goal_vel = np.array([FOLLOW_FORWARD_SPEED, 0.0, 0.0], dtype=np.float32)
                    if self.wait_until_walking == 0 and rwp is not None:
                        _tx, _ty, _ = self.BEAM_POSES[self._player_no]
                        _dist_form = float(np.linalg.norm(
                            rwp[:2] - np.array([_tx, _ty], dtype=np.float32)))
                        if _dist_form < FORMATION_ARRIVE_DIST:
                            # Signal arrival the first time we're within range
                            if not self._formation_arrived:
                                self._formation_arrived = True
                                self._post_kickoff_wait = KICKOFF_STAND_CYCLES  # 3 s hold
                                logger.info('[P%d] Formation position reached — holding for %d cycles',
                                            self._player_no, KICKOFF_STAND_CYCLES)
                                if self._ready_file:
                                    try:
                                        with open(self._ready_file, 'a') as _rf:
                                            _rf.write(f'{self._player_no}\n')
                                    except OSError as _e:
                                        logger.warning('[P%d] Could not write ready file: %s',
                                                       self._player_no, _e)
                            # Count down — then release to chase ball
                            _yaw_to_ball = float(np.clip(cur_head_yaw * STEER_KP, -1.0, 1.0))
                            goal_vel = np.array([0.0, 0.0, _yaw_to_ball], dtype=np.float32)
                            if self._post_kickoff_wait > 0:
                                self._post_kickoff_wait -= 1
                                if self._post_kickoff_wait == 0:
                                    self._formation_phase = False
                                    logger.info('[P%d] Hold done — chasing ball!', self._player_no)
                        else:
                            self._at_formation_cycles = 0
                            goal_vel = self._walk_to_formation(rwp, orientation_quat_inv)

                elif self._post_kickoff_wait > 0:
                    # Just kicked off — stand still and face the ball for 3 seconds
                    self._post_kickoff_wait -= 1
                    _yaw_to_ball = float(np.clip(cur_head_yaw * STEER_KP, -1.0, 1.0))
                    goal_vel = np.array([0.0, 0.0, _yaw_to_ball], dtype=np.float32)
                    if self._post_kickoff_wait == 0:
                        logger.info('[P%d] Post-kickoff hold done — chasing ball!', self._player_no)

                elif self._aligned_with_goal:
                    # Aligned: walk forward to push ball into goal.
                    # Keep body pointed at goal using spawn-yaw-based correction.
                    _push_yaw_vel = 0.0
                    if self._spawn_yaw is not None:
                        _sfwd_p = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                        _push_curr_yaw = float(np.arctan2(float(_sfwd_p[1]), float(_sfwd_p[0])))
                        _push_target   = float(self._wrap_to_pi(self._spawn_yaw + np.pi / 2))
                        _push_err      = float(self._wrap_to_pi(_push_curr_yaw - _push_target))
                        _push_yaw_vel  = float(np.clip(-_push_err * STEER_KP, -1.0, 1.0))
                    goal_vel = np.array([APPROACH_SPEED, 0.0, _push_yaw_vel], dtype=np.float32)

                    # Re-orbit if the ball is no longer in front (robot missed the ball).
                    _err = None
                    if _ball_body_az is not None:
                        _err = float(self._wrap_to_pi(_ball_body_az))
                    elif self._spawn_yaw is not None:
                        # Use body-yaw drift as proxy when ball not directly visible.
                        _err = float(self._wrap_to_pi(_push_curr_yaw - _push_target))
                    if _err is not None and abs(_err) > 3 * ALIGN_THRESHOLD_RAD:
                        self._aligned_with_goal = False
                        self._orbiting = False
                        logger.info('[P%d] Re-orbiting (drift %.1f°)', self._player_no, np.rad2deg(abs(_err)))

                else:
                    # Supporter: use standard decision.
                    if self._role != 'attacker':
                        goal_vel = self._decide_goal_vel(ball_raw, cur_head_yaw, self._role, rwp, orientation_quat_inv)
                    else:
                        # ATTACKER: use ball_dist (works from last_known/world-model, not just fresh vision)
                        # so orbit never stalls on blind frames.
                        if ball_dist is not None:
                            if ball_dist <= ORBIT_ENGAGE_M + 0.3:
                                self._orbiting = True
                            if ball_dist > SLOW_DIST_M:
                                self._orbiting = False

                        if not self._orbiting:
                            # APPROACH: walk toward ball + lateral pre-position.
                            goal_vel = self._decide_goal_vel(ball_raw, cur_head_yaw, self._role, rwp, orientation_quat_inv)
                            if _ball_body_az is not None and _goal_body_az is not None:
                                goal_vel[1] = float(np.clip(
                                    float(self._wrap_to_pi(_ball_body_az - _goal_body_az)) * APPROACH_LATERAL_KP,
                                    -0.3, 0.3))
                        else:
                            # ORBIT: circle ball, maintaining radius, shortest arc toward goal.
                            # Use cur_head_yaw as ball-direction estimate (head tracks ball).
                            _yaw_to_ball = float(np.clip(
                                (cur_head_yaw + (ball_raw[1] if ball_raw is not None else 0.0)) * STEER_KP,
                                -1.0, 1.0))
                            _vx_orbit = float(np.clip(
                                ((ball_dist or ORBIT_RADIUS_M) - ORBIT_RADIUS_M) * ORBIT_RADIAL_KP,
                                -FOLLOW_FORWARD_SPEED, FOLLOW_FORWARD_SPEED))
                            # Compute goal-target yaw from spawn yaw (always available).
                            # spawn_yaw = direction robot faces at beam (-90° = south).
                            # Opponent goal is 90° CCW from that (east = 0°).
                            _sfwd2 = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                            _curr_yaw = float(np.arctan2(float(_sfwd2[1]), float(_sfwd2[0])))
                            _goal_target_yaw = (float(self._wrap_to_pi(self._spawn_yaw + np.pi / 2))
                                                if self._spawn_yaw is not None else 0.0)

                            # Only trust goal detection when it's approximately in the expected direction.
                            # This prevents the own goal (behind the robot during orbit) from being
                            # mistaken for the opponent goal and triggering a premature alignment.
                            # Expected goal body-frame azimuth = wrap(target_world_yaw - curr_body_yaw).
                            _expected_goal_body = float(self._wrap_to_pi(_goal_target_yaw - _curr_yaw))
                            _goal_in_right_dir = (_goal_body_az is not None and
                                                  abs(float(self._wrap_to_pi(_goal_body_az - _expected_goal_body)))
                                                  < np.deg2rad(45))

                            # Direction: use goal body-az when visible AND in expected direction;
                            # otherwise fall back to shortest arc toward spawn-based target yaw.
                            if _goal_in_right_dir:
                                _baz = _ball_body_az if _ball_body_az is not None else cur_head_yaw
                                _vy = float(np.sign(float(self._wrap_to_pi(_baz - _goal_body_az)))) * ORBIT_SPEED
                                if _vy == 0.0:
                                    _vy = -ORBIT_SPEED
                            else:
                                # Shortest arc to alignment target using body yaw.
                                _arc = float(self._wrap_to_pi(_curr_yaw - _goal_target_yaw))
                                # _arc < 0 → robot is CW from target → need CCW orbit → _vy < 0
                                # _arc > 0 → robot is CCW from target → need CW orbit → _vy > 0
                                _vy = float(np.sign(_arc)) * ORBIT_SPEED if _arc != 0.0 else -ORBIT_SPEED
                            goal_vel = np.array([_vx_orbit, _vy, _yaw_to_ball], dtype=np.float32)

                            # Alignment check.
                            # FALLBACK A: world model (requires torso_pos — rarely available)
                            # FALLBACK B: both ball and goal visible in head frame
                            # FALLBACK C: body yaw vs spawn-derived target (always available)
                            _aligned = False
                            if (self._ball_world_pos is not None and self._goal_world_pos is not None
                                    and rwp is not None):
                                _bl = orientation_quat_inv.apply(self._ball_world_pos - rwp)
                                _gl = orientation_quat_inv.apply(self._goal_world_pos - rwp)
                                _aligned = (abs(float(self._wrap_to_pi(
                                    float(np.arctan2(_bl[1], _bl[0])) - float(np.arctan2(_gl[1], _gl[0])))))
                                            < ALIGN_THRESHOLD_RAD)
                            elif _ball_body_az is not None and _goal_in_right_dir:
                                # _goal_in_right_dir already confirms _goal_body_az is not None
                                # AND the goal is in the expected direction (not own goal).
                                _aligned = (abs(float(self._wrap_to_pi(_ball_body_az - _goal_body_az)))
                                            < ALIGN_THRESHOLD_RAD)
                            elif self._spawn_yaw is not None:
                                _yaw_err  = abs(float(self._wrap_to_pi(_curr_yaw - _goal_target_yaw)))
                                _ball_ahead = abs(cur_head_yaw) < np.deg2rad(15)
                                _aligned = (_yaw_err < ALIGN_THRESHOLD_RAD and _ball_ahead)
                                logger.info('[P%d] align-C  curr=%.1f°  target=%.1f°  err=%.1f°  head=%.1f°  vy=%.2f  ok=%s',
                                            self._player_no, np.rad2deg(_curr_yaw), np.rad2deg(_goal_target_yaw),
                                            np.rad2deg(_yaw_err), np.rad2deg(cur_head_yaw), _vy, _aligned)
                            if _aligned:
                                self._aligned_with_goal = True
                                self._orbiting = False
                                logger.info('[P%d] Aligned with goal — stopping', self._player_no)
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
                observation = np.clip(np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0), -10.0, 10.0)

                with torch.no_grad():
                    obs_tensor  = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor, next_policy_hidden = self.policy(obs_tensor, self.policy_hidden)
                nn_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

                target_joint_positions = self.joint_nominal_position + self.scaling_factor * nn_action
                target_joint_positions_degrees = np.rad2deg(target_joint_positions)
                if ENABLE_HEAD_TRACKING:
                    # Targets are in radians; motor command protocol uses degrees
                    target_joint_positions_degrees[self.HEAD_YAW_IDX]   = np.rad2deg(self._head_yaw_target)
                    target_joint_positions_degrees[self.HEAD_PITCH_IDX] = np.rad2deg(self._head_pitch_target)

                motors = self.ROBOT_MOTORS[self._model_name]
                msg_list: list[str] = []
                for motor, target_joint_position in zip(motors, target_joint_positions_degrees, strict=False):
                    msg_list.append(f'({motor} {target_joint_position:.2f} 0.0 {self.p_gain:.2f} {self.d_gain:.2f} 0.0)')

                self.previous_action = nn_action
                self.policy_hidden   = next_policy_hidden
                self._step_gait_manager()

                self._log_csv(game_time, play_mode, rwp, ball_visible,
                              ball_x, ball_y, ball_z, goal_x, goal_y)
                action_msg = ''.join(msg_list)
                self._send_message(action_msg.encode())

            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception:', exc_info=e.__traceback__)
                break

    # ------------------------------------------------------------------ networking
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
    parser = argparse.ArgumentParser(description='The RoboCup MuJoCo Soccer Simulation Booster Client.')
    robots = list(Client.ROBOT_MOTORS.keys())
    parser.add_argument('-s', '--host',      type=str, help='The server address.', default='127.0.0.1', required=False)
    parser.add_argument('-p', '--port',      type=int, help='The server port.',    default=60000,       required=False)
    parser.add_argument('-t', '--team',      type=str, help='The team name.',      default='Test',      required=False)
    parser.add_argument('-n', '--player_no', type=int, help='The player number.',  default=1,           required=False)
    parser.add_argument('-r', '--robot',      type=str, help='The robot model.',    default=robots[0],   required=False, choices=robots)
    parser.add_argument('--ready-file',       type=str, help='Path to formation-ready status file shared with trainer.py.',
                        default='formation_ready.txt', required=False)

    args = parser.parse_args()

    def _shutdown(sig: int, frame: FrameType | None) -> None:
        client.shutdown()

    client = Client(
        host=args.host,
        port=args.port,
        team=args.team,
        player_no=args.player_no,
        model_name=args.robot,
        ready_file=args.ready_file,
    )
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    client.run()
