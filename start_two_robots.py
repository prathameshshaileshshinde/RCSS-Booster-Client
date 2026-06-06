"""
start_two_robots.py
-------------------
Launches two nn_client agents simultaneously with different starting positions.

Usage:
    python start_two_robots.py
    python start_two_robots.py --team MyTeam --robot T1

Each agent runs in its own process and logs to its own CSV file.
Stop both agents with Ctrl+C.
"""

import argparse
import subprocess
import sys
import signal
import os


def main():
    parser = argparse.ArgumentParser(description='Launch two NN agents.')
    parser.add_argument('-s', '--host',  type=str, default='127.0.0.1')
    parser.add_argument('-p', '--port',  type=int, default=60000)
    parser.add_argument('-t', '--team',  type=str, default='Test')
    parser.add_argument('-r', '--robot', type=str, default='T1', choices=['ant', 'T1'])
    args = parser.parse_args()

    python  = sys.executable
    script  = os.path.join(os.path.dirname(__file__), 'nn_client.py')
    base    = [python, script,
               '--host',  args.host,
               '--port',  str(args.port),
               '--team',  args.team,
               '--robot', args.robot]

    # Player 1 starts at position 1 (27.5m forward, centre)
    # Player 2 starts at position 2 (22.0m forward, 12m left)
    p1 = subprocess.Popen(base + ['--player_no', '1'])
    p2 = subprocess.Popen(base + ['--player_no', '2'])

    print(f'[INFO] Agent 1 started (PID {p1.pid}) — player_no=1')
    print(f'[INFO] Agent 2 started (PID {p2.pid}) — player_no=2')
    print('[INFO] Press Ctrl+C to stop both agents.')

    def shutdown(sig, frame):
        print('\n[INFO] Shutting down both agents...')
        p1.terminate()
        p2.terminate()

    signal.signal(signal.SIGINT, shutdown)

    p1.wait()
    p2.wait()
    print('[INFO] Both agents stopped.')


if __name__ == '__main__':
    main()
