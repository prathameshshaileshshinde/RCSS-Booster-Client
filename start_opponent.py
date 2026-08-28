"""
start_opponent.py
=================
Launcher for the predefined opponent team.

Usage:
  python start_opponent.py --mode passive   --count 5
  python start_opponent.py --mode aggressive --count 5

Each robot runs as a separate process (same as start_team.py).
Press Ctrl+C to stop all robots.

Terminal sequence for a full experiment run:
  Terminal 1:  python -m rcsssmj
  Terminal 2:  python start_team.py --team RedTeam --robot T1 --count 5
  Terminal 3:  python start_opponent.py --mode passive --count 5
  Terminal 4:  python trainer.py --count 5
"""

import argparse
import signal
import subprocess
import sys
import time

processes: list[subprocess.Popen] = []


def shutdown(sig=None, frame=None):
    print('\n[INFO] Stopping all opponent robots...')
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    for p in processes:
        try:
            p.wait(timeout=3)
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, shutdown)


def main():
    parser = argparse.ArgumentParser(description='Start a team of predefined opponent robots')
    parser.add_argument('-s', '--host', default='127.0.0.1')
    parser.add_argument('-p', '--port', type=int, default=60000)
    parser.add_argument('-t', '--team', default='BlueTeam',
                        help='Opponent team name (must differ from your team)')
    parser.add_argument('-r', '--robot', default='T1', choices=['ant', 'T1'])
    parser.add_argument('-c', '--count', type=int, default=5,
                        help='Number of robots to spawn (1-11)')
    parser.add_argument('--mode', default='passive',
                        choices=['passive', 'aggressive'],
                        help='passive = hold formation; aggressive = all chase ball')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Seconds between spawning each robot')
    parser.add_argument('--csv-dir', default='CSV')
    args = parser.parse_args()

    count = max(1, min(11, args.count))
    print(f'[INFO] Spawning {count} opponent robot(s) | team={args.team} '
          f'mode={args.mode}')

    base_cmd = [
        sys.executable, 'predefined_opponent.py',
        '--host', args.host,
        '--port', str(args.port),
        '--team', args.team,
        '--robot', args.robot,
        '--mode', args.mode,
        '--csv-dir', args.csv_dir,
    ]

    for i in range(1, count + 1):
        cmd = base_cmd + ['--player_no', str(i)]
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f'[INFO] Spawned player {i} (PID {p.pid})')
        time.sleep(args.delay)

    print(f'[INFO] All {count} opponent robots running. Press Ctrl+C to stop.')

    # Wait for all processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        shutdown()


if __name__ == '__main__':
    main()
