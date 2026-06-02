import argparse
import logging
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

# ---------- LOGGING CONFIG ----------
# console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
ch.setLevel(logging.INFO)

# configure logging
logging.basicConfig(handlers=[ch], level=logging.DEBUG)
# ---------- LOGGING CONFIG ----------


logger = logging.getLogger(__name__)


class Client:
    """
    Example client performing random actions.
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
        'T1': ('he1', 'he2', 'lae1', 'lae2', 'lae3', 'lae4', 'rae1', 'rae2', 'rae3', 'rae4', 'te1', 'lle1', 'lle2', 'lle3', 'lle4', 'lle5', 'lle6', 'rle1', 'rle2', 'rle3', 'rle4', 'rle5', 'rle6'),
    }

    def __init__(self, host: str, port: int, team: str, player_no: int, model_name: str | None = None):
        """
        Construct a new agent connecting to the given server.
        """

        self._host: str = host
        self._port: int = port

        self._model_name: str = 'ant' if model_name is None else model_name
        self._team: str = team
        self._player_no: int = player_no

        self._policy_checkpoint = "locomotion_nn.pth"
        self._policy_meta = "locomotion_nn_meta.json"
        self._gait_period = 1.0
        self._policy_dt = 0.02

        self._rcv_buffer_size = 1024
        self._rcv_buffer = bytearray(self._rcv_buffer_size)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._has_beamed: bool = False

        # Ball tracking and search state (all positions in metres)
        self._last_ball_x:      float | None       = None  # last known ball x (metres forward)
        self._last_ball_y:      float | None       = None  # last known ball y (metres sideways)
        self._ball_world_pos:   np.ndarray | None  = None  # ball position in world coordinates
        self._search_dir:       int                = 1     # 1 = turn left, -1 = turn right
        self._head_scan_angle:  float              = 0.0   # head sweep angle when searching (degrees)
        self._ball_close_timer: int                = 0     # counts down after ball was under feet
        self.kick_state:        str                = 'seek' # states: 'seek', 'align', 'attack'

        # set TCP_NODELAY option to send messages immediately (without buffering)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def run(self):
        """
        Run the simulation client.
        """

        # connect to server
        logger.info('Connecting to server at %s:%d...', self._host, self._port)
        try:
            self._sock.connect((self._host, self._port))
        except ConnectionRefusedError:
            logger.error('Connection refused. Make sure the server is running and listening on the specified interface.')  # noqa: TRY400
            return
        # logger.info('Server connection established.')

        # create client thread
        client_thread = threading.Thread(target=self._action_loop)
        client_thread.start()

        # wait for client thread to finish
        client_thread.join()

        # logger.info('Shutting down.')

        # close server connection
        self._sock.close()

    def shutdown(self) -> None:
        """
        Shutdown the client.
        """

        self._sock.shutdown(socket.SHUT_RDWR)

    def _init_policy_runtime_state(self):
        self.previous_action = np.zeros(self.nr_joints, dtype=np.float32)
        self.policy_hidden = self.policy.initialize_carry(batch_size=1, device=self.device)

        self.wait_until_walking = 50

        self.gait_phase_offset = np.array([0.0, -np.pi], dtype=np.float32)
        self.gait_phase = self.gait_phase_offset.copy()
        self.gait_mean_freq = 1.0 / self._gait_period
        self.gait_freq = self.gait_mean_freq
        self.gait_phase_dt = (2.0 * np.pi * self._policy_dt) * self.gait_freq

        # Reset ball tracking and kick state on spawn
        self._last_ball_x      = None
        self._last_ball_y      = None
        self._ball_world_pos   = None
        self._search_dir       = 1
        self._head_scan_angle  = 0.0
        self._ball_close_timer = 0
        self.kick_state        = 'seek'

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _get_gait_phase_features(self) -> np.ndarray:
        phase_tp1 = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt)
        return np.concatenate([np.sin(phase_tp1), np.cos(phase_tp1)], axis=-1).astype(np.float32)

    def _step_gait_manager(self):
        self.gait_phase = self._wrap_to_pi(self.gait_phase + self.gait_phase_dt).astype(np.float32)

    def _get_head_target(self, ball_x: float | None, ball_y: float | None, ball_z: float | None) -> tuple[float, float]:
        """
        Returns (he1_degrees, he2_degrees) to point the head at the ball.
        Inputs are Cartesian ball position in metres relative to the robot.
        ball_x = forward, ball_y = sideways (+ = left), ball_z = height (- = below camera).
        When ball is visible: compute head angles from arctan2 — no raw degrees used.
        When ball is not visible: sweep head left/right and tilt down to search.
        """
        if ball_x is not None:
            # Horizontal head angle: arctan2(sideways, forward)
            he1_rad = np.arctan2(ball_y, ball_x)
            he1 = float(np.rad2deg(np.clip(he1_rad, np.deg2rad(-60.0), np.deg2rad(60.0))))

            # Vertical head angle: tilt down when ball is close and low
            dist_horiz = np.sqrt(ball_x**2 + ball_y**2)
            he2_rad = np.arctan2(-ball_z, dist_horiz)

            # When ball is very close, force maximum downward tilt to keep it in view
            if dist_horiz < 1.0:
                he2 = -70.0  # max tilt down — ball is right at the feet
            else:
                he2 = float(np.rad2deg(np.clip(he2_rad, np.deg2rad(-70.0), np.deg2rad(20.0))))

            # Store last known position and sync head scan angle
            self._last_ball_x     = ball_x
            self._last_ball_y     = ball_y
            self._head_scan_angle = float(np.rad2deg(he1_rad))
        else:
            # Ball not visible — sweep head left and right while looking down to search
            self._head_scan_angle += self._search_dir * 4.0  # 4 degrees per cycle
            if abs(self._head_scan_angle) >= 60.0:
                self._search_dir      *= -1
                self._head_scan_angle  = float(np.clip(self._head_scan_angle, -60.0, 60.0))
            he1 = float(self._head_scan_angle)
            he2 = -40.0  # tilt down further while searching — ball is likely on the ground nearby
        return he1, he2

    def _search_for_ball(self) -> np.ndarray:
        """
        Returns goal_vel to rotate the body in place and search for the ball.
        Uses last known Cartesian position (metres) to decide which way to turn.
        """
        if self._last_ball_x is not None:
            # Compute turn direction from last known position using arctan2
            steer   = np.arctan2(self._last_ball_y, self._last_ball_x)
            yaw_vel = float(np.clip(steer / (np.pi / 2), -1.0, 1.0))
            # If ball was almost straight ahead, just spin slowly
            if abs(yaw_vel) < 0.1:
                yaw_vel = self._search_dir * 0.4
        else:
            # No history — spin slowly
            yaw_vel = self._search_dir * 0.4
        # Stand still and rotate only (x=0, y=0)
        return np.array([0.0, 0.0, yaw_vel], dtype=np.float32)

    def _action_loop(self):
        """
        Main loop of the agent.
        """

        self.nr_joints = len(self.ROBOT_MOTORS[self._model_name])

        self.p_gain = 25.0
        self.d_gain = 0.6
        self.scaling_factor = 0.5
        self.joint_nominal_position = np.array([
            0.0, 0.0,
            0.0, -1.4, 0.0, -0.4,
            0.0, 1.4, 0.0, 0.4,
            0.0,
            -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
            -0.4, 0.0, 0.0, 0.8, -0.4, 0.0,
        ], dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy, self.policy_meta = load_policy_from_files(
            self._policy_checkpoint,
            self._policy_meta,
            self.device,
        )
        self._init_policy_runtime_state()

        logger.info('Initializing agent...')
        init_msg = f'(init {self._model_name} {self._team} {self._player_no})'
        self._send_message(init_msg.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                # Beam first, but do not advance GRU or gait state yet
                if not self._has_beamed:
                    beam_pose = self.BEAM_POSES[self._player_no]
                    action_msg = f'(beam {beam_pose[0]} {beam_pose[1]} {beam_pose[2]})'
                    self._send_message(action_msg.encode())
                    self._has_beamed = True
                    self._init_policy_runtime_state()
                    continue

                perception_msg_str = perception_msg.decode()
                perception_data = self.parse_sensor_string(perception_msg_str)

                joint_pos_degrees = np.array([h["ax"] for h in perception_data["HJ"]], dtype=np.float32)
                joint_pos = np.deg2rad(joint_pos_degrees).astype(np.float32)

                joint_vel_degrees = np.array([h["vx"] for h in perception_data["HJ"]], dtype=np.float32)
                joint_vel = np.deg2rad(joint_vel_degrees).astype(np.float32)

                scaled_joint_pos = (joint_pos - self.joint_nominal_position) / 3.14
                scaled_joint_vel = joint_vel / 100.0
                scaled_previous_action = self.previous_action / 10.0

                ang_vel = np.deg2rad(np.array(perception_data["GYR"]["rt"], dtype=np.float32)).astype(np.float32)
                scaled_and_clipped_ang_vel = np.clip(ang_vel / 50.0, -1.0, 1.0)

                orientation_quat_mj_convention = np.array(perception_data["quat"]["q"], dtype=np.float32)
                robot_rotation = R.from_quat([
                    orientation_quat_mj_convention[1],
                    orientation_quat_mj_convention[2],
                    orientation_quat_mj_convention[3],
                    orientation_quat_mj_convention[0],
                ])
                orientation_quat_inv = robot_rotation.inv()
                projected_gravity = orientation_quat_inv.apply(np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)

                # --- Parse robot's absolute world position (torso_pos) ---
                torso_match = re.search(
                    r'\(pos\s+\(n\s+torso_pos\)\s+\(pos\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\)\)',
                    perception_msg_str
                )
                robot_world_pos = np.array([
                    float(torso_match.group(1)),
                    float(torso_match.group(2)),
                    float(torso_match.group(3)),
                ], dtype=np.float32) if torso_match else None

                # --- Parse ball and goal: convert polar -> Cartesian (metres) immediately ---
                ball_match  = re.search(r'\(B\s+\(pol\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\)\)', perception_msg_str)
                goal_match1 = re.search(r'\(G1R\s+\(pol\s+([\d.+-]+)\s+([\d.+-]+)\s+[\d.+-]+\)\)', perception_msg_str)
                goal_match2 = re.search(r'\(G2R\s+\(pol\s+([\d.+-]+)\s+([\d.+-]+)\s+[\d.+-]+\)\)', perception_msg_str)

                if ball_match:
                    ball_dist   = float(ball_match.group(1))
                    _baz        = np.deg2rad(float(ball_match.group(2)))
                    _bel        = np.deg2rad(float(ball_match.group(3)))
                    # Full 3D Cartesian position (metres relative to robot)
                    ball_x = ball_dist * np.cos(_bel) * np.cos(_baz)  # forward
                    ball_y = ball_dist * np.cos(_bel) * np.sin(_baz)  # sideways (+ = left)
                    ball_z = ball_dist * np.sin(_bel)                  # height (- = below camera)
                else:
                    ball_dist = ball_x = ball_y = ball_z = None

                if goal_match1 and goal_match2:
                    _g1x = float(goal_match1.group(1)) * np.cos(np.deg2rad(float(goal_match1.group(2))))
                    _g1y = float(goal_match1.group(1)) * np.sin(np.deg2rad(float(goal_match1.group(2))))
                    _g2x = float(goal_match2.group(1)) * np.cos(np.deg2rad(float(goal_match2.group(2))))
                    _g2y = float(goal_match2.group(1)) * np.sin(np.deg2rad(float(goal_match2.group(2))))
                    goal_x = (_g1x + _g2x) / 2.0   # goal centre x (metres)
                    goal_y = (_g1y + _g2y) / 2.0   # goal centre y (metres)
                elif goal_match1:
                    goal_x = float(goal_match1.group(1)) * np.cos(np.deg2rad(float(goal_match1.group(2))))
                    goal_y = float(goal_match1.group(1)) * np.sin(np.deg2rad(float(goal_match1.group(2))))
                elif goal_match2:
                    goal_x = float(goal_match2.group(1)) * np.cos(np.deg2rad(float(goal_match2.group(2))))
                    goal_y = float(goal_match2.group(1)) * np.sin(np.deg2rad(float(goal_match2.group(2))))
                else:
                    goal_x = goal_y = None

                # --- Ball persistence: always keep the last known position ---
                if ball_x is not None:
                    # Ball is visible — update last known local position and close timer
                    self._last_ball_x = ball_x
                    self._last_ball_y = ball_y
                    if ball_dist is not None and ball_dist < 1.5:
                        self._ball_close_timer = 40  # keep for 40 cycles after ball goes under feet
                    # World model: also store absolute world position if torso available
                    if robot_world_pos is not None:
                        ball_local = np.array([ball_x, ball_y, ball_z if ball_z is not None else 0.0], dtype=np.float32)
                        self._ball_world_pos = robot_world_pos + robot_rotation.apply(ball_local)
                else:
                    # Ball not visible — try world model first, fall back to last known local position
                    if self._ball_world_pos is not None and robot_world_pos is not None:
                        offset     = self._ball_world_pos - robot_world_pos
                        ball_local = orientation_quat_inv.apply(offset)
                        ball_x     = float(ball_local[0])
                        ball_y     = float(ball_local[1])
                        ball_z     = float(ball_local[2])
                        ball_dist  = float(np.linalg.norm(ball_local))
                    elif self._last_ball_x is not None and self._ball_close_timer > 0:
                        # Fallback: use last known local position (ball under feet / very close)
                        ball_x = self._last_ball_x
                        ball_y = self._last_ball_y

                # Count down the close timer
                if self._ball_close_timer > 0:
                    self._ball_close_timer -= 1

                # --- Goal velocity: always walk toward the ball ---
                # The world model above ensures ball_x/ball_y are populated even when
                # the ball is not visible in the camera, so the robot never truly loses it.
                self.wait_until_walking = max(0, self.wait_until_walking - 1)
                if self.wait_until_walking > 0:
                    goal_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                elif ball_x is not None:
                    # Steer directly toward the ball and walk forward at full speed
                    steer   = np.arctan2(ball_y, ball_x)
                    yaw_vel = float(np.clip(steer / (np.pi / 2), -1.0, 1.0))
                    goal_vel = np.array([1.0, 0.0, yaw_vel], dtype=np.float32)

                else:
                    # No ball information at all (never seen it) — search by rotating
                    goal_vel = self._search_for_ball()

                gait_phase_features = self._get_gait_phase_features()

                observation = np.concatenate([
                    scaled_joint_pos,
                    scaled_joint_vel,
                    scaled_previous_action,
                    scaled_and_clipped_ang_vel,
                    goal_vel,
                    gait_phase_features,
                    projected_gravity,
                ])

                observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
                observation = np.clip(observation, -10.0, 10.0)

                with torch.no_grad():
                    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action_tensor, next_policy_hidden = self.policy(obs_tensor, self.policy_hidden)
                nn_action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
                target_joint_positions = self.joint_nominal_position + self.scaling_factor * nn_action

                msg_list: list[str] = []
                motors = self.ROBOT_MOTORS[self._model_name]
                target_joint_positions_degrees = np.rad2deg(target_joint_positions)

                # Head tracking — override NN output for head joints (Cartesian inputs)
                he1_target, he2_target = self._get_head_target(ball_x, ball_y, ball_z)

                for motor, pos in zip(motors, target_joint_positions_degrees, strict=False):
                    if motor == 'he1':
                        pos = he1_target   # horizontal head: track ball azimuth
                    elif motor == 'he2':
                        pos = he2_target   # vertical head: track ball elevation
                    msg_list.append(f'({motor} {pos:.2f} 0.0 {self.p_gain:.2f} {self.d_gain:.2f} 0.0)')

                self.previous_action = nn_action
                self.policy_hidden = next_policy_hidden
                self._step_gait_manager()

                # send action message
                action_msg = ''.join(msg_list)
                self._send_message(action_msg.encode())
            
            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception details:', exc_info=e.__traceback__)
                break

    def _send_message(self, msg: bytes | bytearray) -> None:
        """
        Receive the next message from the TCP/IP socket.
        """

        self._sock.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def _receive_message(self) -> bytes | bytearray:
        """
        Receive the next message from the TCP/IP socket.
        """

        # receive message length information
        if self._sock.recv_into(self._rcv_buffer, nbytes=4, flags=socket.MSG_WAITALL) != 4:
            raise ConnectionResetError

        msg_size = int.from_bytes(self._rcv_buffer[:4], byteorder='big', signed=False)

        # ensure receive buffer is large enough to hold the message
        if msg_size > self._rcv_buffer_size:
            self._rcv_buffer_size = msg_size
            self._rcv_buffer = bytearray(self._rcv_buffer_size)

        # receive message with the specified length
        if self._sock.recv_into(self._rcv_buffer, nbytes=msg_size, flags=socket.MSG_WAITALL) != msg_size:
            raise ConnectionResetError

        return self._rcv_buffer[:msg_size]


    def parse_sensor_string(self, s: str) -> dict:
        """
        Parses a sensor data string of nested parenthesis groups into a structured dictionary.
        Repeated top-level tags are aggregated into lists.
        """
        result = {}
        # Top-level groups: (TAG ...content...)
        top_level_pattern = re.compile(r'\((\w+)((?:\s*\([^()]*\))*)\)')
        
        for tag, inner in top_level_pattern.findall(s):
            # Find inner key-value or key-list groups: (key val1 val2 ...)
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
                # Single value vs. list
                group[key] = parsed_vals[0] if len(parsed_vals) == 1 else parsed_vals
            
            # Merge into result, handling repeated tags as lists
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(group)
                else:
                    result[tag] = [result[tag], group]
            else:
                result[tag] = group
        
        return result


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='The RoboCup MuJoCo Soccer Simulation Example Client.')

    robots = list(Client.ROBOT_MOTORS.keys())

    # fmt: off
    parser.add_argument('-s', '--host',      type=str, help='The server address.', default='127.0.0.1', required=False)
    parser.add_argument('-p', '--port',      type=int, help='The server port.',    default=60000,       required=False)
    parser.add_argument('-t', '--team',      type=str, help='The team name.',      default='Test',      required=False)
    parser.add_argument('-n', '--player_no', type=int, help='The player number.',  default=1,           required=False)
    parser.add_argument('-r', '--robot',     type=str, help='The robot model.',    default=robots[0],   required=False, choices=robots)
    # fmt: on

    args = parser.parse_args()

    # create client
    client = Client(args.host, args.port, args.team, args.player_no, args.robot)

    # register SIGINT handler
    def signal_handler(sig: int, frame: FrameType | int | signal.Handlers | None) -> None:
        del sig, frame  # signal unused parameter
        client.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # run client
    client.run()
