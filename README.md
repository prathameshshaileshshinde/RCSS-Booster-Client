# Neural Network Client — RCSSServerMJ

A Python client for the [RCSSServerMJ](https://gitlab.com/robocup-sim/rcssservermj) RoboCup Soccer Simulation Server.  
It connects a T1 humanoid robot to the simulation, uses a pre-trained GRU-based neural network for locomotion, and implements autonomous ball-tracking, head control, and multi-robot team play.

---

## Table of Contents

1. [Features](#features)
2. [File Reference](#file-reference)
3. [Requirements & Setup](#requirements--setup)
4. [Usage](#usage)
5. [Architecture: How It Works](#architecture-how-it-works)
6. [Vision Control & Ball Tracking](#vision-control--ball-tracking)
7. [Head Control](#head-control)
8. [Robot Turning Logic](#robot-turning-logic)
9. [Multi-Robot Simulation & Communication](#multi-robot-simulation--communication)
10. [CSV Data Collection](#csv-data-collection)
11. [Voronoi Diagrams for Team Strategy](#voronoi-diagrams-for-team-strategy)
12. [Git Workflow](#git-workflow)
13. [Learning Resources](#learning-resources)

---

## Features

- **Pre-trained locomotion policy** — GRU recurrent neural network for the T1 humanoid robot
- **Robust vision control** — ball tracking with world-model fallback; head sweeps left/right when ball is lost
- **Head tracking** — `he1`/`he2` joints actively point at the ball; exponential smoothing prevents jitter
- **Intelligent movement** — three-phase SEEK → ORBIT → PUSH state machine to approach and shoot
- **Turn function** — reusable `_turn_around()` and `_search_for_ball()` for in-place rotation
- **Multi-robot awareness** — parses teammate/opponent positions from the vision S-expression
- **CSV data logging** — every cycle logs game time, player position, ball position, and nearest teammate
- **Team launcher** — Python `start_team.py` and bash `start_team.sh` to start all 11 players at once

---

## File Reference

| File | Description |
|---|---|
| `nn_client.py` | Main client — perception-action loop, all robot logic |
| `torch_policy.py` | GRU policy architecture and weight loader |
| `locomotion_nn.pth` | Pre-trained model weights (T1 robot) |
| `locomotion_nn_meta.json` | Model architecture configuration |
| `requirements.txt` | Python dependencies |
| `start_two_robots.py` | Launches 2 agents at different starting positions |
| `start_team.py` | Cross-platform launcher for a full team of 11 agents |
| `start_team.sh` | Bash launcher for 11 agents (Linux/macOS) |
| `CSV/` | Auto-created folder containing position log CSV files |
| `VORONOI.md` | Research: Voronoi diagrams for team positioning strategy |
| `GIT_WORKFLOW.md` | Git & GitHub workflow guide |

---

## Requirements & Setup

**Python 3.11+** is required.

### 1. Install the simulation server

```bash
pip install rcsssmj
```

### 2. Install client dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes: `numpy`, `scipy`, `torch`

---

## Usage

### Single agent

**Terminal 1 — Start the simulation server:**
```bash
python -m rcsssmj
```

**Terminal 2 — Run one agent:**
```bash
python nn_client.py --robot T1 --team MyTeam --player_no 1
```

### Two robots at different positions

```bash
python start_two_robots.py --team MyTeam --robot T1
```

Player 1 spawns at `(27.5, 0.0)` and Player 2 at `(22.0, 12.0)`.

### Full team of 11 robots (Python — cross-platform)

```bash
python start_team.py --team MyTeam --robot T1
```

### Full team of 11 robots (Bash — Linux/macOS)

```bash
bash start_team.sh MyTeam T1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-s` / `--host` | `127.0.0.1` | Server IP address |
| `-p` / `--port` | `60000` | Server port |
| `-t` / `--team` | `Test` | Team name |
| `-n` / `--player_no` | `1` | Player number (1–11) |
| `-r` / `--robot` | `ant` | Robot model (`ant` or `T1`) |

> **Note:** The pre-trained locomotion model only works with `--robot T1`.

---

## Architecture: How It Works

Each simulation cycle (50 Hz):

1. Server sends perception data — joint angles, gyroscope, orientation quaternion, vision
2. Ball position is parsed from the `(B (pol dist az el))` S-expression and converted to Cartesian `(x, y, z)` in metres
3. If the ball is not visible, position is estimated from the stored world-coordinate model and current robot pose
4. Head joints (`he1`, `he2`) are pointed at the ball; they sweep left/right when searching
5. Movement goal velocity is computed via the SEEK → ORBIT → PUSH state machine
6. Observation vector (joints, IMU, goal velocity, gait phase, gravity projection) is fed to the GRU policy
7. Motor torque commands are sent back to the server
8. One row is appended to the CSV log

---

## Vision Control & Ball Tracking

### The Challenge

The simulation's vision perceptor does **not** fire every cycle — it fires every **two** cycles (25 Hz), so `ball_match` is `None` on alternating steps even when the ball is directly in front of the robot. Additionally, the ball can leave the camera's field of view entirely when the robot turns.

### Three-Layer Persistence System

The client handles missing ball data through three layers (checked in order):

1. **World model** — when the ball is visible, its position is converted to world coordinates (`_ball_world_pos`). On cycles where the ball is not visible, the stored world position is projected back into the robot's current frame using the robot's rotation quaternion. This works as long as the robot's absolute `torso_pos` is available in the perceptor.

2. **Close-ball timer** (`_ball_close_timer`) — when the ball is within 2 m, a 150-cycle (3 s) timer is set. Within that window, the last known `(ball_x, ball_y, ball_z)` is reused directly. This bridges the every-2-cycle vision gap and ensures the robot keeps walking through the ball.

3. **Ball-reset detection** — after a goal, the ball teleports. If a sudden position jump is detected (new reading > 5 m away while smoothed estimate < 3 m), all tracking state is cleared so the robot starts a fresh search.

### Head Sweep Search

When all three layers have no estimate, `_search_for_ball()` rotates the robot body and `_get_head_target()` sweeps `he1` left/right (±60°) at 4°/cycle, looking down (`he2 = -40°`) to scan the ground.

```python
# Sweep direction reverses at ±60°
self._head_scan_angle += self._search_dir * 4.0
if abs(self._head_scan_angle) >= 60.0:
    self._search_dir *= -1
```

### Exponential Smoothing

Raw vision readings are noisy. Ball position is smoothed with α = 0.25:

```python
smooth_x = 0.75 * smooth_x + 0.25 * raw_x
```

All steering and head decisions use the smoothed value to prevent jerky movement.

---

## Head Control

The `_get_head_target(ball_x, ball_y, ball_z)` function computes `he1` (horizontal pan) and `he2` (vertical tilt) in degrees.

```
he1 = arctan2(ball_y, ball_x)        # pan left/right to track azimuth
he2 = arctan2(ball_z, sqrt(x²+y²))  # tilt down/up to track elevation
```

Sign convention:
- `he1 > 0` → head turns left; `he1 < 0` → turns right
- `he2 < 0` → head tilts **down** (ball below camera, which is the usual case)
- `he2 > 0` → head tilts up

When `dist_horiz < 1.0 m` (ball nearly underfoot), `he2` is forced to `−70°` (maximum down tilt).

The horizontal angle is also exponentially smoothed to prevent oscillation:

```python
self._head_scan_angle = 0.7 * self._head_scan_angle + 0.3 * target_he1
```

Both `he1` and `he2` are clamped to hardware limits: `he1 ∈ [−60°, 60°]`, `he2 ∈ [−70°, 20°]`.

---

## Robot Turning Logic

### `_turn_around()`

Returns a `goal_vel = [0, 0, yaw]` that spins the robot ~180° in place. The turn direction is chosen based on the last known ball side so re-acquisition is faster:

```python
turn_dir = sign(self._last_ball_y)   # positive = turn left, negative = right
return np.array([0.0, 0.0, turn_dir])
```

### `_search_for_ball()`

Rotates the robot body while searching. If the ball was seen before, it rotates toward the last known direction. Otherwise it spins continuously in `_search_dir`:

```python
steer   = arctan2(last_ball_y, last_ball_x)
yaw_vel = sign(steer) if |steer| > 0.1 else search_dir
return np.array([0.0, 0.0, yaw_vel])
```

### SEEK → ORBIT → PUSH State Machine

Once the ball is located, movement follows three phases:

| State | Condition | Behaviour |
|---|---|---|
| **SEEK** | `dist > 3.5 m` | Walk toward ball; rotate first if misaligned > 25° |
| **ORBIT** | `0.8 m < dist ≤ 3.5 m` | Circle around the ball at 1.0 m radius to get behind it |
| **PUSH** | `dist < 0.8 m` and aligned | Walk straight through ball toward goal |

The **alignment** check ensures the robot is behind the ball relative to the goal before pushing:

```
dot = (robot→ball · ball→goal) / |robot→ball|
# dot > 0.65 means robot is well behind the ball
```

---

## Multi-Robot Simulation & Communication

### How Robots Detect Each Other

Robots are detected through the **vision perceptor** S-expression. When another robot (teammate or opponent) is in the camera's field of view, the server includes a `(P ...)` block:

```
(P (team TeamName) (id 3) (head (pol 4.2 15.3 -2.1)) ...)
```

The `_parse_teammates()` method parses all visible agents:

```python
@staticmethod
def _parse_teammates(perception_msg_str: str) -> list[dict]:
    # Returns: [{'team': str, 'id': int, 'x': float, 'y': float, 'z': float, 'dist': float}, ...]
```

Polar coordinates `(dist, azimuth_deg, elevation_deg)` are immediately converted to Cartesian metres:

```python
x = dist * cos(el) * cos(az)   # forward
y = dist * cos(el) * sin(az)   # sideways
z = dist * sin(el)              # height
```

### S-Expression Message Structure

Every perception cycle the server sends a message like:

```
(time (now 123.45))
(GS (t 12.3) (pm PlayOn))
(GYR (n torso) (rt 0.1 0.2 0.3))
(HJ (n he1) (ax -5.2) (vx 0.1)) ...
(see
  (B (pol 3.5 10.2 -5.1))
  (G1R (pol 22.1 -3.0 1.2))
  (P (team Opponent) (id 7) (head (pol 5.0 25.0 -1.0)))
  ...
)
```

Key tags:

| Tag | Meaning |
|---|---|
| `GS` | Game state: time `t`, play mode `pm` |
| `HJ` | Hinge joint: name `n`, angle `ax`, velocity `vx` |
| `GYR` | Gyroscope angular velocity |
| `quat` | Orientation quaternion |
| `pos (n torso_pos)` | Robot world position |
| `B` | Ball — polar `(dist, azimuth°, elevation°)` |
| `G1R` / `G2R` | Goal posts (right team) |
| `P` | Player (teammate or opponent) |

### Two-Robot Test

`start_two_robots.py` spawns Player 1 at `(27.5, 0.0)` and Player 2 at `(22.0, 12.0)`. Each runs independently and logs to its own CSV. Since both run `_parse_teammates()`, each robot logs the closest visible agent (teammate or opponent) every cycle — enabling post-hoc analysis of inter-robot proximity.

### 11-Robot Team Test

`start_team.py` / `start_team.sh` starts all 11 players. Observed behaviour: all players walk toward the ball, which causes crowding. Known limitation: without communication or role assignment, multiple robots compete for the ball. The CSV logs capture all positions, enabling analysis of clustering and coverage.

---

## CSV Data Collection

### Folder & Naming

The `CSV/` folder is created automatically (`Path('CSV').mkdir(exist_ok=True)`). Each agent writes one file:

```
CSV/robot_log_<team>_p<player_no>_<YYYYMMDD_HHMMSS>.csv
```

### Schema

```
game_time, play_mode,
player_no, team,
robot_world_x, robot_world_y, robot_world_z,
ball_visible,
ball_rel_x, ball_rel_y, ball_rel_z,
ball_world_x, ball_world_y, ball_world_z,
goal_rel_x, goal_rel_y,
teammate_team, teammate_id,
teammate_rel_x, teammate_rel_y, teammate_rel_z, teammate_dist
```

| Column | Description |
|---|---|
| `game_time` | Simulation clock (seconds) |
| `play_mode` | Server play-mode string (e.g. `PlayOn`, `KickOff_Left`) |
| `robot_world_*` | Absolute world coordinates of the robot torso |
| `ball_visible` | `1` if ball was seen this cycle, `0` if estimated |
| `ball_rel_*` | Ball position relative to robot (metres) |
| `ball_world_*` | Ball absolute world position (from world model) |
| `goal_rel_*` | Nearest visible goalpost centre, robot-relative |
| `teammate_*` | Closest visible agent — team, ID, position, distance |

### Intended Uses

- **Team behaviour analysis** — plot robot positions over time to study formation and crowding
- **Machine learning** — use position sequences to train positioning policies
- **Debugging** — `ball_visible=0` rows reveal how often the world model is relied upon

Logging adds negligible overhead (one `csv.writer.writerow` per cycle).

---

## Voronoi Diagrams for Team Strategy

See [`VORONOI.md`](VORONOI.md) for a full research document. Summary:

A **Voronoi diagram** partitions the field into regions — one per player — where each cell contains all points closer to that player than to any other. This gives every robot a natural "ownership zone."

Key applications:
- **Positioning** — move to the centroid of your Voronoi cell to maximise space coverage
- **Space control** — larger cells = more defensive coverage; small cells near the ball = pressing
- **Decision making** — pass to the teammate whose Voronoi cell contains open space toward goal
- **Formation** — adjust positions to make all cells roughly equal (balanced formation) or bias cells toward the opponent goal (attacking shape)

Quick implementation with SciPy:

```python
from scipy.spatial import Voronoi
positions = np.array([[x1,y1], [x2,y2], ...])  # all robot world positions
vor = Voronoi(positions)
```

---

## Git Workflow

See [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) for the full guide. Quick reference:

```bash
# Clone
git clone <repo-url>
cd nn_client

# Create a branch for new work
git checkout -b feature/my-improvement

# Stage and commit
git add nn_client.py
git commit -m "feat: improve ball tracking with world model fallback"

# Push branch to GitHub
git push origin feature/my-improvement

# Pull latest changes from main
git checkout master
git pull origin master

# Merge and resolve conflicts
git checkout feature/my-improvement
git merge master
# Edit conflicting files, then:
git add <resolved-file>
git commit -m "merge: resolve conflict in nn_client.py"
```

---

## Learning Resources

### Python Fundamentals
- [Official Python Tutorial](https://docs.python.org/3/tutorial/) — core language, data structures, OOP
- [Real Python](https://realpython.com/) — practical articles; good on `argparse`, `subprocess`, `pathlib`
- [Python Type Hints](https://docs.python.org/3/library/typing.html) — this codebase uses type hints throughout

### Git & GitHub
- [Pro Git (free book)](https://git-scm.com/book/en/v2) — definitive reference
- [GitHub Skills](https://skills.github.com/) — interactive labs for branching, PRs, actions
- [Oh Shit, Git!](https://ohshitgit.com/) — how to recover from common mistakes

### Robotics Simulation
- [RCSSServerMJ GitLab](https://gitlab.com/robocup-sim/rcssservermj) — server documentation and S-expression protocol
- [RL-X Training Framework](https://github.com/nico-bohlinger/RL-X) — the framework used to train the locomotion policy
- [RoboCup Simulation 3D League](https://ssim.robocup.org/) — competition rules, agent examples

### Reinforcement Learning & Team Behaviour
- [Spinning Up in Deep RL (OpenAI)](https://spinningup.openai.com/) — clear intro to policy gradient methods
- [Multi-Agent RL survey (Lowe et al.)](https://arxiv.org/abs/1706.02275) — MADDPG, cooperative vs. competitive agents
- [SciPy Voronoi docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html) — computing Voronoi diagrams for positioning

### Numerical Computing
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) — arrays, broadcasting, trig functions
- [SciPy spatial transforms](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html) — quaternion rotation (used in this project)
