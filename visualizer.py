"""
visualizer.py
-------------
Real-time ASCII field visualizer for RCSSServerMJ.
Connects to the monitor port and draws the field, ball, and players every frame.

Usage:
    python visualizer.py
    python visualizer.py --host 127.0.0.1 --monitor-port 60001
"""

import argparse
import os
import re
import socket
import sys
import time


# ── Field dimensions (metres) ────────────────────────────────────────────────
FIELD_HALF_X = 30.0   # half-length  (x axis)
FIELD_HALF_Y = 20.0   # half-width   (y axis)

# ── ASCII canvas size ────────────────────────────────────────────────────────
COLS = 80
ROWS = 30


def _recvn(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly n bytes — Windows-safe (no MSG_WAITALL)."""
    data = b''
    while len(data) < n:
        try:
            chunk = sock.recv(n - len(data))
        except OSError:
            return None
        if not chunk:
            return None
        data += chunk
    return data


def _recv(sock: socket.socket) -> bytes | None:
    hdr = _recvn(sock, 4)
    if hdr is None:
        return None
    size = int.from_bytes(hdr, 'big')
    return _recvn(sock, size)


def _parse_play_mode(msg: bytes) -> str | None:
    m = re.search(rb'\(pm\s+(\w+)\)', msg)
    return m.group(1).decode() if m else None


def _parse_game_time(msg: bytes) -> float | None:
    m = re.search(rb'\(t\s+([\d.]+)\)', msg)
    return float(m.group(1)) if m else None


def _parse_ball(msg: bytes) -> tuple[float, float] | None:
    """Parse ball position. Tries several known monitor message formats."""
    # Format: (Ball (pos X Y Z) ...)
    m = re.search(rb'\(Ball[^)]*\(pos\s+([-\d.]+)\s+([-\d.]+)', msg)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Flat format: (ball X Y Z)
    m = re.search(rb'\(ball\s+([-\d.]+)\s+([-\d.]+)', msg)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _parse_players(msg: bytes) -> list[tuple[str, int, float, float]]:
    """
    Returns list of (team, player_id, x, y).
    Tries two common monitor formats.
    """
    players = []
    # Format A: (P (team NAME) (id N) (pos X Y Z) ...)
    for m in re.finditer(
            rb'\(P\s+\(team\s+(\w+)\)\s+\(id\s+(\d+)\)\s+\(pos\s+([-\d.]+)\s+([-\d.]+)', msg):
        players.append((m.group(1).decode(), int(m.group(2)),
                        float(m.group(3)), float(m.group(4))))
    if players:
        return players
    # Format B: (player (team NAME) (unum N) (pos X Y Z) ...)
    for m in re.finditer(
            rb'\(player\s+\(team\s+(\w+)\)\s+\(unum\s+(\d+)\)\s+\(pos\s+([-\d.]+)\s+([-\d.]+)', msg):
        players.append((m.group(1).decode(), int(m.group(2)),
                        float(m.group(3)), float(m.group(4))))
    return players


def _world_to_canvas(x: float, y: float) -> tuple[int, int]:
    """Convert world (x,y) in metres to canvas (col, row). x=right, y=up on field."""
    col = int((x + FIELD_HALF_X) / (2 * FIELD_HALF_X) * (COLS - 1))
    row = int((FIELD_HALF_Y - y) / (2 * FIELD_HALF_Y) * (ROWS - 1))
    col = max(0, min(COLS - 1, col))
    row = max(0, min(ROWS - 1, row))
    return col, row


def _draw(play_mode: str, game_time: float,
          ball: tuple[float, float] | None,
          players: list[tuple[str, int, float, float]],
          teams: list[str]) -> None:
    """Render the field to stdout."""
    # Blank canvas
    grid = [['·'] * COLS for _ in range(ROWS)]

    # Field border
    for c in range(COLS):
        grid[0][c] = '-'
        grid[ROWS - 1][c] = '-'
    for r in range(ROWS):
        grid[r][0] = '|'
        grid[r][COLS - 1] = '|'

    # Centre line (x=0)
    cx, _ = _world_to_canvas(0.0, 0.0)
    for r in range(1, ROWS - 1):
        if grid[r][cx] == '·':
            grid[r][cx] = ':'

    # Goals (left x=-30, right x=+30, goal width ±3.66 m → half=3.66)
    goal_half = 3.66
    for gx, side in [(-FIELD_HALF_X, '<'), (FIELD_HALF_X, '>')]:
        gc, _ = _world_to_canvas(gx, 0.0)
        for gy in [goal_half, -goal_half]:
            _, gr = _world_to_canvas(0.0, gy)
            if 0 <= gr < ROWS:
                if 0 <= gc < COLS:
                    grid[gr][gc] = side

    # Players — pick symbol by team
    team_symbols = {}
    for i, t in enumerate(sorted(set(p[0] for p in players))):
        team_symbols[t] = str(i + 1)   # team 1 → '1', team 2 → '2'

    for team, pid, px, py in players:
        c, r = _world_to_canvas(px, py)
        sym = team_symbols.get(team, 'P')
        grid[r][c] = sym

    # Ball
    if ball:
        bc, br = _world_to_canvas(ball[0], ball[1])
        grid[br][bc] = 'O'

    # ── Print ────────────────────────────────────────────────────────────────
    # Clear screen (ANSI)
    sys.stdout.write('\033[2J\033[H')

    print(f'  RCSSServerMJ Visualizer   t={game_time:.1f}s   [{play_mode}]')
    print()

    # Left goal label
    print('  [LEFT GOAL]' + ' ' * (COLS - 20) + '[RIGHT GOAL]')

    for row in grid:
        print('  ' + ''.join(row))

    print()
    # Legend
    ball_str = f'({ball[0]:+.1f}, {ball[1]:+.1f})' if ball else '(unknown)'
    print(f'  Ball: {ball_str}    O=ball  ·=field  :=centre line')
    if team_symbols:
        for t, sym in team_symbols.items():
            print(f'  {sym} = {t}')
    print()
    print('  Ctrl+C to quit')
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description='RCSSServerMJ ASCII field visualizer.')
    parser.add_argument('-s', '--host',         default='127.0.0.1')
    parser.add_argument('-p', '--monitor-port', type=int, default=60001)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    print(f'Connecting to {args.host}:{args.monitor_port} …')
    try:
        sock.connect((args.host, args.monitor_port))
    except ConnectionRefusedError:
        print('ERROR: connection refused — is the server running?')
        sys.exit(1)
    print('Connected. Waiting for data…')
    time.sleep(0.5)

    play_mode = 'Unknown'
    game_time = 0.0
    ball: tuple[float, float] | None = None
    players: list[tuple[str, int, float, float]] = []
    teams: list[str] = []

    try:
        while True:
            msg = _recv(sock)
            if msg is None:
                print('Server disconnected.')
                break

            pm = _parse_play_mode(msg)
            if pm:
                play_mode = pm

            gt = _parse_game_time(msg)
            if gt is not None:
                game_time = gt

            b = _parse_ball(msg)
            if b is not None:
                ball = b

            p = _parse_players(msg)
            if p:
                players = p
                teams = list(set(x[0] for x in players))

            _draw(play_mode, game_time, ball, players, teams)

    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print('\nBye.')


if __name__ == '__main__':
    main()
