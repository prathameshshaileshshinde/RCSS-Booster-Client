# Neural Network Client — RCSSServerMJ

A Python client for the [RCSSServerMJ](https://gitlab.com/robocup-sim/rcssservermj) RoboCup Soccer Simulation Server. It connects a T1 humanoid robot to the simulation, uses a pre-trained GRU-based neural network for locomotion, and includes autonomous ball-tracking and pursuit behaviour.

## Features

- **Pre-trained locomotion policy** — GRU recurrent neural network trained with [RL-X](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robocup_soccer) for the T1 humanoid robot
- **Ball tracking** — robot steers toward the ball using Cartesian coordinates (polar → x/y/z in metres via `arctan2`)
- **Head tracking** — `he1`/`he2` head joints actively point at the ball; sweeps left/right when ball not visible
- **World model** — stores ball position in world coordinates so the robot keeps pursuing the ball even when it leaves the camera's field of view
- **Ball persistence** — last known ball position is remembered for short periods so the robot walks through the ball rather than stopping to search

## Files

| File | Description |
|---|---|
| `nn_client.py` | Main client — connects to server, runs perception-action loop |
| `torch_policy.py` | GRU-based neural network policy architecture and loader |
| `locomotion_nn.pth` | Pre-trained model weights (T1 robot) |
| `locomotion_nn_meta.json` | Model architecture config |
| `requirements.txt` | Python dependencies |
| `start_team.sh` | Shell script to start a full team of 11 agents |

## Requirements

- Python 3.11+
- [RCSSServerMJ](https://gitlab.com/robocup-sim/rcssservermj) (`pip install rcsssmj`)
- See `requirements.txt` for client dependencies

## Setup

1. Install the simulation server:

   ```bash
   pip install rcsssmj
   ```

2. Install client dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Terminal 1 — Start the simulation server:**

```bash
python -m rcsssmj
```

**Terminal 2 — Run a single agent:**

```bash
python nn_client.py --robot T1 --team MyTeam --player_no 1
```

**Or start a full team of 11 agents (Linux/macOS/Git Bash):**

```bash
bash start_team.sh team1 T1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-s` / `--host` | `127.0.0.1` | Server IP address |
| `-p` / `--port` | `60000` | Server port |
| `-t` / `--team` | `Test` | Team name |
| `-n` / `--player_no` | `1` | Player number (1–11) |
| `-r` / `--robot` | `ant` | Robot model (`ant` or `T1`) |

> **Note:** The pre-trained model only works with `--robot T1`.

## How It Works

Each simulation cycle (50 Hz):

1. Server sends perception data — joint angles, gyroscope, orientation quaternion, vision
2. Ball position is parsed from the vision perceptor and converted from polar to Cartesian `(x, y, z)` metres
3. If ball is not visible, position is estimated from the stored world-coordinate position and current robot pose
4. Head joints (`he1`, `he2`) are pointed at the ball using `arctan2`; sweep left/right when searching
5. Steering is computed as `arctan2(ball_y, ball_x) / (π/2)` — no raw degree values used
6. Observation vector is built and passed to the GRU policy network
7. Motor commands are sent back to the server

## Robot Models

| Model | Joints | Notes |
|---|---|---|
| `ant` | 8 | Simple four-legged robot |
| `T1` | 23 | Full humanoid — required for pre-trained model |

## Starting Position

Each player number maps to a beam position on the field defined in `BEAM_POSES`. To start from a different location, edit the `(x, y, angle)` values or pass `--player_no` to select a preset.
