"""
start_team.py
-------------
Cross-platform launcher that starts NN agents for RCSSServerMJ.
Works on Windows, Linux, and macOS (unlike the bash shell script).

Usage:
    python start_team.py                        # 11 robots (full team)
    python start_team.py --count 5              # 5 robots (players 1-5)
    python start_team.py --team RedTeam --robot T1
    python start_team.py --team BlueTeam --robot T1 --host 192.168.1.10

All agents start in separate processes, each with its own CSV log file.
Press Ctrl+C to stop all agents cleanly.
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description='Launch NN agents for RCSSServerMJ.'
    )
    parser.add_argument('-s', '--host',  type=str, default='127.0.0.1',
                        help='Server IP address (default: 127.0.0.1)')
    parser.add_argument('-p', '--port',  type=int, default=60000,
                        help='Server port (default: 60000)')
    parser.add_argument('-t', '--team',  type=str, default='Test',
                        help='Team name (default: Test)')
    parser.add_argument('-r', '--robot', type=str, default='T1',
                        choices=['ant', 'T1'],
                        help='Robot model (default: T1)')
    parser.add_argument('-n', '--count', type=int, default=11,
                        help='Number of robots to spawn, players 1..N (default: 11)')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay in seconds between spawning each agent (default: 0.2)')
    parser.add_argument('--ready-file', type=str, default='formation_ready.txt',
                        help='Shared file agents write to when they reach formation '
                             '(default: formation_ready.txt). Pass the same value to trainer.py.')
    args = parser.parse_args()

    count = max(1, min(args.count, 11))   # clamp to 1-11

    python = sys.executable
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nn_client.py')

    base_cmd = [
        python, script,
        '--host',       args.host,
        '--port',       str(args.port),
        '--team',       args.team,
        '--robot',      args.robot,
        '--ready-file', args.ready_file,
    ]

    processes: list[subprocess.Popen] = []

    print(f'[INFO] Starting team "{args.team}" — {count} x {args.robot} robots')
    print(f'[INFO] Server: {args.host}:{args.port}')
    print(f'[INFO] Formation ready file: {args.ready_file}')
    print('[INFO] Press Ctrl+C to stop all agents.\n')

    for player_no in range(1, count + 1):
        cmd = base_cmd + ['--player_no', str(player_no)]
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f'[INFO] Agent {player_no:>2}/{count} started  (PID {proc.pid})')
        time.sleep(args.delay)   # stagger spawns so the server isn't overwhelmed

    print(f'\n[INFO] All {count} agents running. Waiting for them to finish...')

    # ── Graceful shutdown on Ctrl+C ──────────────────────────────────────────
    def _shutdown(sig, frame):
        print('\n[INFO] Ctrl+C received — stopping all agents...')
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass

    signal.signal(signal.SIGINT, _shutdown)

    # Wait for all processes to exit
    for p in processes:
        p.wait()

    print('[INFO] All agents stopped.')


if __name__ == '__main__':
    main()
