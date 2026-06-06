# Voronoi Diagrams for RoboCup Team Strategy

## What is a Voronoi Diagram?

A Voronoi diagram takes a set of points (robots on a soccer field) and partitions the plane into regions called **cells**. Every point in a cell is closer to its "owner" robot than to any other robot.

For a soccer team, this means: your Voronoi cell is the area of the field you are closest to — your natural zone of responsibility.

```
Before positioning (clustered):     After Voronoi-optimal positioning:
  . . . . . . . . . .                 . . . | . . | . . . .
  . . ●●● . . . . .                  . . ● | . ● | . . ● .
  . . ●●● . . . . .                  . . . | . . | . . . .
  . . ●●● . . . . .                  . . . | . . | . . . .
  (3 robots crowded)                  (3 robots spread out)
```

---

## Why Voronoi Diagrams Matter for RoboCup

### 1. Positioning and Space Control

The **total area of your team's Voronoi cells** is the proportion of the field your team "controls." A larger combined area means more space to receive passes, fewer opponents to beat.

The classic result from Delaunay geometry: to maximise your cell, move to the **centroid** of your current cell. In practice:

```python
# Pseudo-code for one robot's positioning step
my_cell_centroid = compute_voronoi_centroid(my_voronoi_cell)
move_toward(my_cell_centroid)
```

### 2. Team Strategy and Formation

| Shape goal | Voronoi rule |
|---|---|
| Balanced / zonal defence | Equalise all cell areas |
| High press | Move all cells forward so opponents have no space |
| Counter-attack | Keep 2–3 players with large cells behind the ball |
| Overload a flank | Push 3+ players to one side, shrinking opponent cells there |

The team can switch strategy by publishing a target formation vector and each robot adjusting its position to fit.

### 3. Decision Making: Passing

A good pass target is a teammate whose Voronoi cell contains open space **toward the opponent goal**. The passer computes:

```
for each teammate t:
    score(t) = area_of_cell_between(t, opponent_goal) 
             - distance(me, t)          # penalise long passes
             - defender_proximity(t)    # penalise contested teammates
best_pass = argmax(score)
```

### 4. Marking and Pressing

Each defender can be assigned to mark the **opponent whose Voronoi cell overlaps most with a dangerous zone** (the area near your own goal). This is more stable than nearest-opponent marking and avoids two defenders chasing the same player.

---

## Implementation for This Project

This project already logs each robot's world position to CSV every cycle. The Voronoi diagram can be computed **offline** from those CSVs, or **online** (every N cycles) inside `nn_client.py`.

### Offline Analysis (from CSV)

```python
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

# Load all 11 robots for a given timestamp
df = pd.read_csv('CSV/robot_log_Test_p1_....csv')
snapshot = df[df['game_time'] == 10.0][['robot_world_x', 'robot_world_y']].dropna()

points = snapshot.values
vor = Voronoi(points)
voronoi_plot_2d(vor)
plt.xlim(-15, 15); plt.ylim(-10, 10)
plt.title('Team Voronoi — t=10s')
plt.show()
```

### Online (Inside the Simulation)

To share positions between robots in real time, each robot would need to:
1. Broadcast its world position via the **say perceptor** (if the server supports it), OR
2. Estimate teammates' world positions from the vision `(P ...)` blocks and each robot's own world position.

Option 2 is already partially available — `_parse_teammates()` gives robot-relative positions, and combined with `robot_world_pos` you can triangulate teammates' world coordinates:

```python
teammate_world_pos = robot_world_pos + robot_rotation.apply(
    np.array([tm['x'], tm['y'], tm['z']])
)
```

Once you have all visible teammates' world positions, compute the Voronoi diagram:

```python
from scipy.spatial import Voronoi
all_positions = np.array([[me_x, me_y]] + [[t['wx'], t['wy']] for t in world_teammates])
vor = Voronoi(all_positions)
# My cell = vor.regions[vor.point_region[0]]
```

### Suggested Positioning Function

```python
def voronoi_target_position(my_pos, teammate_positions, field_bounds=(-15, 15, -10, 10)):
    """
    Returns the centroid of this robot's Voronoi cell, clipped to the field.
    my_pos: (x, y)
    teammate_positions: list of (x, y) for all other robots
    """
    from scipy.spatial import Voronoi
    import numpy as np

    all_pts = np.array([my_pos] + teammate_positions)
    if len(all_pts) < 2:
        return np.array(my_pos)

    vor = Voronoi(all_pts)
    region_idx = vor.point_region[0]
    region = vor.regions[region_idx]

    if -1 in region or len(region) == 0:
        # Open cell (touches field boundary) — use field-clipped centroid
        return np.array(my_pos)   # fallback; in production clip to field bbox

    vertices = vor.vertices[region]
    centroid = vertices.mean(axis=0)
    # Clip to field bounds
    x = np.clip(centroid[0], field_bounds[0], field_bounds[1])
    y = np.clip(centroid[1], field_bounds[2], field_bounds[3])
    return np.array([x, y])
```

---

## Limitations and Notes

- **Voronoi is positional, not predictive** — it says where to stand, not how to get there. The locomotion policy handles movement.
- **Open cells** near the field boundary require clipping against the rectangular field. SciPy's `Voronoi` produces half-open cells at the edges; clip vertices to the field rectangle before computing centroids.
- **Not all robots are visible** — the vision perceptor has a limited field of view. Positions need to be inferred from the last known world model, especially for robots behind the camera.
- **Computational cost** — computing Voronoi for 11 points takes < 1 ms in Python/SciPy. Running it every 10 cycles (5 Hz) adds negligible overhead.

---

## Further Reading

- [Voronoi Diagrams in Sports Analytics (general)](https://en.wikipedia.org/wiki/Voronoi_diagram)
- [Space Control in Football via Voronoi (Fernández & Bornn, 2018)](https://www.researchgate.net/publication/325950688) — academic paper applying Voronoi to real soccer
- [SciPy Voronoi documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Voronoi.html)
- [RoboCup 3D Simulation multi-agent positioning strategies](https://ssim.robocup.org/)
