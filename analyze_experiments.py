"""
analyze_experiments.py
======================
Analyses CSV logs from RCSSServerMJ experiments and computes three
evaluation metrics for the RedTeam (adaptive Voronoi team):

  1. Time-to-first-ball-contact  : seconds from KickOff to first cycle
                                   where any RedTeam robot sees the ball
                                   at distance < CONTACT_THRESHOLD metres.
  2. Role-switch frequency       : average role switches per robot per minute
                                   of play time.
  3. Mean Voronoi field coverage : average fraction of the field area covered
                                   by the RedTeam's Voronoi diagram over the
                                   course of the match.

Results are printed as a comparison table across experiment conditions
(baseline / vs-passive / vs-aggressive).

Usage:
  python analyze_experiments.py --csv-dir CSV --output results.csv

The script auto-detects condition from the CSV filename:
  - RedTeam logs (robot_log_RedTeam_*)  → your adaptive team data
  - opponent logs (opponent_passive_*)  → passive condition
  - opponent logs (opponent_aggressive_*) → aggressive condition

Run conditions are grouped by their start timestamp (first 15 chars of
filename).
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIELD_X_MIN, FIELD_X_MAX = -15.0, 15.0
FIELD_Y_MIN, FIELD_Y_MAX = -10.0, 10.0
FIELD_AREA = (FIELD_X_MAX - FIELD_X_MIN) * (FIELD_Y_MAX - FIELD_Y_MIN)  # m²

CONTACT_THRESHOLD = 1.5   # metres — robot is "in contact" with ball

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_redteam_logs(csv_dir: str, run_ts: str) -> Optional[pd.DataFrame]:
    """Load and concatenate all RedTeam player logs for a given run timestamp."""
    pattern = os.path.join(csv_dir, f'robot_log_RedTeam_*_{run_ts}.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def time_to_first_contact(df: pd.DataFrame) -> float:
    """
    Seconds from KickOff play mode start to the first cycle where any
    robot is within CONTACT_THRESHOLD metres of the ball.
    """
    play_df = df[df['play_mode'].str.contains('KickOff|PlayOn', na=False)].copy()
    if play_df.empty:
        return float('nan')

    t0 = play_df['game_time'].min()

    # ball_rel distance = sqrt(ball_rel_x² + ball_rel_y²)
    if 'ball_rel_x' in play_df.columns and 'ball_rel_y' in play_df.columns:
        play_df = play_df.dropna(subset=['ball_rel_x', 'ball_rel_y'])
        play_df['ball_dist'] = np.sqrt(
            play_df['ball_rel_x']**2 + play_df['ball_rel_y']**2
        )
        visible_contact = play_df[
            (play_df['ball_visible'] == 1) &
            (play_df['ball_dist'] < CONTACT_THRESHOLD)
        ]
        if not visible_contact.empty:
            return float(visible_contact['game_time'].min() - t0)
    return float('nan')


def role_switch_frequency(df: pd.DataFrame) -> float:
    """
    Average role switches per robot per minute of play time.
    A role switch is any change from 'attacker' ↔ 'supporter'.
    """
    if 'role' not in df.columns or 'player_no' not in df.columns:
        return float('nan')

    play_df = df[df['play_mode'].str.contains('KickOff|PlayOn', na=False)]
    if play_df.empty:
        return float('nan')

    total_minutes = (play_df['game_time'].max() -
                     play_df['game_time'].min()) / 60.0
    if total_minutes <= 0:
        return float('nan')

    switch_counts = []
    for player_id, group in play_df.groupby('player_no'):
        group = group.sort_values('game_time')
        roles = group['role'].dropna().values
        switches = int(np.sum(roles[1:] != roles[:-1]))
        switch_counts.append(switches)

    if not switch_counts:
        return float('nan')
    avg_switches = np.mean(switch_counts)
    return float(avg_switches / total_minutes)


def voronoi_coverage(df: pd.DataFrame, n_samples: int = 50) -> float:
    """
    Estimate the fraction of the field covered by the RedTeam's Voronoi
    diagram, averaged over time snapshots.

    For each snapshot, takes all robot world positions and computes what
    fraction of a grid of field points is closer to a RedTeam robot than
    to any point outside the team (approximated by field boundary points).

    Simplified approach: fraction of field-grid points that are within
    the convex hull of the team's positions (a good proxy for Voronoi
    coverage when the field is much larger than the team spread).
    """
    if 'robot_world_x' not in df.columns or 'robot_world_y' not in df.columns:
        return float('nan')

    play_df = df[df['play_mode'].str.contains('KickOff|PlayOn', na=False)].dropna(
        subset=['robot_world_x', 'robot_world_y'])
    if play_df.empty:
        return float('nan')

    # Sample evenly-spaced time points
    times = np.linspace(play_df['game_time'].min(),
                        play_df['game_time'].max(), n_samples)

    # Field grid (100 × 67 points ≈ 6700 grid points)
    gx = np.linspace(FIELD_X_MIN, FIELD_X_MAX, 100)
    gy = np.linspace(FIELD_Y_MIN, FIELD_Y_MAX, 67)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    coverages = []
    for t in times:
        # Closest game_time snapshot
        idx = (play_df['game_time'] - t).abs().argsort()
        snapshot = play_df.iloc[idx[:5]][['robot_world_x', 'robot_world_y']].values
        if len(snapshot) < 2:
            continue

        # Clip to field bounds
        snapshot[:, 0] = np.clip(snapshot[:, 0], FIELD_X_MIN, FIELD_X_MAX)
        snapshot[:, 1] = np.clip(snapshot[:, 1], FIELD_Y_MIN, FIELD_Y_MAX)

        # For each grid point, find nearest robot
        dists = np.linalg.norm(
            grid_pts[:, None, :] - snapshot[None, :, :], axis=2
        )  # (n_grid, n_robots)
        min_dists = dists.min(axis=1)

        # Coverage: fraction of field within 5m of a team robot
        # (a rough but interpretable proxy for Voronoi zone area)
        coverages.append(float(np.mean(min_dists < 5.0)))

    return float(np.nanmean(coverages)) if coverages else float('nan')


# ---------------------------------------------------------------------------
# Run detection
# ---------------------------------------------------------------------------

def detect_runs(csv_dir: str):
    """
    Returns a dict mapping run_condition → list of run timestamps.
    Conditions: 'baseline', 'vs_passive', 'vs_aggressive'
    """
    runs = defaultdict(set)

    # RedTeam logs: robot_log_RedTeam_p1_YYYYMMDD_HHMMSS.csv
    for f in glob.glob(os.path.join(csv_dir, 'robot_log_RedTeam_p1_*.csv')):
        m = re.search(r'_p1_(\d{8}_\d{6})\.csv$', f)
        if m:
            ts = m.group(1)
            # Check if there's a matching opponent log
            passive_opp = glob.glob(
                os.path.join(csv_dir, f'opponent_passive_*_{ts}.csv'))
            agg_opp = glob.glob(
                os.path.join(csv_dir, f'opponent_aggressive_*_{ts}.csv'))
            if passive_opp:
                runs['vs_passive'].add(ts)
            elif agg_opp:
                runs['vs_aggressive'].add(ts)
            else:
                runs['baseline'].add(ts)

    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_metrics_for_run(csv_dir: str, run_ts: str) -> dict:
    df = load_redteam_logs(csv_dir, run_ts)
    if df is None or df.empty:
        return {'ttfc': float('nan'), 'rsf': float('nan'), 'cov': float('nan')}
    return {
        'ttfc': time_to_first_contact(df),
        'rsf':  role_switch_frequency(df),
        'cov':  voronoi_coverage(df),
    }


def main():
    parser = argparse.ArgumentParser(description='Analyse RCSSServerMJ experiment CSV logs')
    parser.add_argument('--csv-dir', default='CSV', help='Directory containing CSV logs')
    parser.add_argument('--output', default='results.csv', help='Output results CSV')
    args = parser.parse_args()

    if not os.path.isdir(args.csv_dir):
        print(f'[ERROR] CSV directory not found: {args.csv_dir}')
        return

    runs = detect_runs(args.csv_dir)
    if not runs:
        # Fallback: treat all RedTeam logs from same date as one run
        for f in glob.glob(os.path.join(args.csv_dir, 'robot_log_RedTeam_p1_*.csv')):
            m = re.search(r'_p1_(\d{8}_\d{6})\.csv$', f)
            if m:
                runs['baseline'].add(m.group(1))

    print(f'\nFound runs:')
    for cond, tss in runs.items():
        print(f'  {cond}: {len(tss)} run(s)')

    # Aggregate metrics per condition
    summary = {}
    for condition, timestamps in runs.items():
        metrics_list = [compute_metrics_for_run(args.csv_dir, ts)
                        for ts in sorted(timestamps)]
        ttfcs = [m['ttfc'] for m in metrics_list if not np.isnan(m['ttfc'])]
        rsfs  = [m['rsf']  for m in metrics_list if not np.isnan(m['rsf'])]
        covs  = [m['cov']  for m in metrics_list if not np.isnan(m['cov'])]
        summary[condition] = {
            'n_runs': len(timestamps),
            'time_to_first_contact_mean_s': round(np.nanmean(ttfcs), 2) if ttfcs else 'N/A',
            'time_to_first_contact_std_s':  round(np.nanstd(ttfcs),  2) if ttfcs else 'N/A',
            'role_switches_per_robot_min_mean': round(np.nanmean(rsfs), 3) if rsfs else 'N/A',
            'voronoi_coverage_mean_pct': round(np.nanmean(covs) * 100, 1) if covs else 'N/A',
        }

    # Print table
    print('\n' + '='*75)
    print(f'{"Condition":<20} {"Runs":>4} {"TTFC mean (s)":>14} '
          f'{"TTFC std":>9} {"Role sw/robot/min":>18} {"Coverage %":>11}')
    print('-'*75)
    for cond, m in summary.items():
        print(f'{cond:<20} {m["n_runs"]:>4} '
              f'{str(m["time_to_first_contact_mean_s"]):>14} '
              f'{str(m["time_to_first_contact_std_s"]):>9} '
              f'{str(m["role_switches_per_robot_min_mean"]):>18} '
              f'{str(m["voronoi_coverage_mean_pct"]):>11}')
    print('='*75)
    print('\nMetric definitions:')
    print('  TTFC          : Time to first ball contact after kickoff (lower = better)')
    print('  Role sw/r/min : Role switches per robot per minute (higher = more adaptive)')
    print('  Coverage %    : % of field within 5 m of a team robot (higher = better spacing)')

    # Save to CSV
    rows = []
    for cond, m in summary.items():
        rows.append({'condition': cond, **m})
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f'\nResults saved → {args.output}')


if __name__ == '__main__':
    main()
