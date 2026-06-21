# RoboCup NN Client

A Python client for the [RCSSServerMJ](https://gitlab.com/robocup-sim/rcssservermj) MuJoCo-based soccer simulation. Robots use a pre-trained neural network for locomotion and automatically walk to formation positions before playing.

---

## Setup

**Python 3.11+** required.

```bash
pip install rcsssmj
pip install -r requirements.txt
```

---

## Quick Start

Always start the server first, then the robots.

**Terminal 1 — Start the server:**
```bash
python -m rcsssmj
```

**Terminal 2 — Single robot (recommended for testing):**
```bash
python nn_client.py --team RedTeam --robot T1 --player_no 11
```

**Terminal 2 — Full team (5 robots):**
```bash
python start_team.py --team RedTeam --robot T1 --count 5
```

**Terminal 3 — Auto-kickoff when all robots reach their positions:**
```bash
python trainer.py --count 5
```

**Terminal 4 — Live field visualizer (optional):**
```bash
python visualizer.py
```

---

## Robot Behaviour

Each robot follows this sequence automatically:

1. **Beam** — teleports to its sideline spawn position
2. **Walk** — walks from the sideline to its formation position (dead-reckoning navigation)
3. **Hold** — stands at formation position, rotating to face the ball
4. **Wait** — holds for 3 seconds after arriving (or after kickoff)
5. **Chase** — walks toward the ball and kicks it

The robot goes through all these steps on its own. You only need `trainer.py` if you want auto-kickoff when running multiple robots.

---

## Formation Positions

Each player number maps to a fixed field position (`BEAM_POSES` in `nn_client.py`):

| Player | Position (x, y) | Role |
|--------|----------------|------|
| 1 | (27.5, 0.0) | Goalkeeper |
| 2 | (22.0, 12.0) | Defender |
| 3 | (22.0, 4.0) | Defender |
| 4 | (22.0, -4.0) | Defender |
| 5 | (22.0, -12.0) | Defender |
| 6 | (15.0, 0.0) | Midfielder |
| 7 | (4.0, 16.0) | Winger |
| 8 | (11.0, 6.0) | Midfielder |
| 9 | (11.0, -6.0) | Midfielder |
| 10 | (4.0, -16.0) | Winger |
| 11 | (7.0, 0.0) | Striker |

---

## Files

| File | What it does |
|------|-------------|
| `nn_client.py` | Main robot agent — all perception, navigation, and play logic |
| `torch_policy.py` | Loads the GRU locomotion neural network |
| `locomotion_nn.pth` | Pre-trained model weights (T1 robot only) |
| `locomotion_nn_meta.json` | Model architecture config |
| `start_team.py` | Launches N robots (players 1–N) in separate processes |
| `trainer.py` | Connects to monitor port and sends kickoff once all robots are in position |
| `visualizer.py` | Real-time ASCII field view — shows ball and players live |
| `requirements.txt` | Python dependencies (`numpy`, `scipy`, `torch`) |
| `CSV/` | Auto-created folder with per-robot position logs |

---

## Changing the Ball Weight

The ball is defined in the server's MuJoCo world file. To make it lighter (flies further when kicked):

1. Open:
   ```
   rcsssmj_env\Lib\site-packages\rcsssmj\resources\environments\soccer\world.xml
   ```
2. Find the line:
   ```xml
   <geom name="ball" ... mass="0.41" .../>
   ```
3. Change `mass="0.41"` to your desired value (e.g. `mass="0.2"` for a lighter ball)
4. Restart the server

Or run this PowerShell command (replace the path if needed):
```powershell
(Get-Content "...\world.xml") -replace 'mass="0.41"', 'mass="0.2"' | Set-Content "...\world.xml"
```

---

## Key Arguments

### `nn_client.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--team` | `Test` | Team name |
| `--robot` | `ant` | Robot model (`T1` recommended) |
| `--player_no` | `1` | Player number (1–11) |
| `--host` | `127.0.0.1` | Server address |
| `--port` | `60000` | Server port |
| `--ready-file` | `formation_ready.txt` | Shared file used to signal trainer.py |

### `start_team.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--count` | `11` | Number of robots to spawn (players 1–N) |
| `--team` | `Test` | Team name |
| `--robot` | `T1` | Robot model |

### `trainer.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--count` | `0` | Wait for this many robots before kickoff (0 = use `--delay`) |
| `--delay` | `3.0` | Seconds to wait before kickoff (used when `--count` is 0) |
| `--side` | `Left` | Which team kicks off |
