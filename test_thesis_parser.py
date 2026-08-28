"""test_thesis_parser.py — unit tests for ThesisClient's corrected sexp parser.

Run without a GPU or real server:
    python -m unittest test_thesis_parser -v

All torch / torch_policy / scipy imports are stubbed so the tests run in any
environment that has numpy installed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub heavy dependencies so the module can be imported without a GPU or
# the real rcss-booster environment.
# ---------------------------------------------------------------------------

def _make_torch_stub():
    torch_mod = types.ModuleType('torch')
    torch_mod.no_grad   = MagicMock(return_value=MagicMock(
        __enter__=MagicMock(return_value=None),
        __exit__=MagicMock(return_value=False)))
    torch_mod.device    = MagicMock(return_value='cpu')
    torch_mod.tensor    = MagicMock()
    torch_mod.float32   = None
    torch_mod.cuda      = MagicMock()
    torch_mod.cuda.is_available = MagicMock(return_value=False)
    return torch_mod


def _make_scipy_stub():
    scipy_mod  = types.ModuleType('scipy')
    spatial    = types.ModuleType('scipy.spatial')
    transform  = types.ModuleType('scipy.spatial.transform')

    class _FakeR:
        @staticmethod
        def from_quat(q):
            return _FakeR()
        def inv(self):
            return self
        def apply(self, v):
            import numpy as np
            return np.asarray(v, dtype=float)

    transform.Rotation = _FakeR
    spatial.transform  = transform
    scipy_mod.spatial  = spatial
    return scipy_mod, spatial, transform


def _make_policy_stub():
    mod = types.ModuleType('torch_policy')
    mod.load_policy_from_files = MagicMock(return_value=(MagicMock(), MagicMock()))
    return mod


# Register stubs before importing the module under test.
torch_stub = _make_torch_stub()
scipy_stub, spatial_stub, transform_stub = _make_scipy_stub()
policy_stub = _make_policy_stub()

sys.modules.setdefault('torch',                   torch_stub)
sys.modules.setdefault('scipy',                   scipy_stub)
sys.modules.setdefault('scipy.spatial',           spatial_stub)
sys.modules.setdefault('scipy.spatial.transform', transform_stub)
sys.modules.setdefault('torch_policy',            policy_stub)

from thesis_nn_client import ThesisClient   # noqa: E402  (import after stubs)


# ---------------------------------------------------------------------------
# Helper: create a ThesisClient with a patched socket (no real connection).
# ---------------------------------------------------------------------------

def _make_client(team='BlueTeam'):
    with patch('socket.socket'):
        c = ThesisClient(host='127.0.0.1', port=60000,
                         team=team, player_no=1)
    return c


# ===========================================================================
# Tests
# ===========================================================================

class TestParseSexpTree(unittest.TestCase):
    """_parse_sexp_tree: basic structure."""

    def test_flat_single(self):
        tree = ThesisClient._parse_sexp_tree('(foo bar)')
        self.assertEqual(tree, [['foo', 'bar']])

    def test_nested_two_levels(self):
        tree = ThesisClient._parse_sexp_tree('(outer (inner val))')
        self.assertEqual(tree, [['outer', ['inner', 'val']]])

    def test_multiple_top_level(self):
        tree = ThesisClient._parse_sexp_tree('(A 1)(B 2)')
        self.assertEqual(len(tree), 2)
        self.assertEqual(tree[0][0], 'A')
        self.assertEqual(tree[1][0], 'B')

    def test_deeply_nested(self):
        tree = ThesisClient._parse_sexp_tree('(pos (n torso_pos)(p 1.0 2.0 3.0))')
        self.assertIsInstance(tree[0], list)
        self.assertEqual(tree[0][0], 'pos')


class TestExtractGS(unittest.TestCase):
    """_extract_gs: GS field extraction."""

    GS_MSG = ('(GS (t 12.34)(pm PlayOn)'
              '(tl BlueTeam)(tr RedTeam)(sl 1)(sr 0))')

    def setUp(self):
        self.c = _make_client('BlueTeam')

    def _gs(self, msg):
        return self.c._extract_gs(ThesisClient._parse_sexp_tree(msg))

    def test_game_time(self):
        gs = self._gs(self.GS_MSG)
        self.assertAlmostEqual(gs['t'], 12.34)

    def test_play_mode(self):
        gs = self._gs(self.GS_MSG)
        self.assertEqual(gs['pm'], 'PlayOn')

    def test_team_left(self):
        gs = self._gs(self.GS_MSG)
        self.assertEqual(gs['tl'], 'BlueTeam')

    def test_team_right(self):
        gs = self._gs(self.GS_MSG)
        self.assertEqual(gs['tr'], 'RedTeam')

    def test_score_left(self):
        gs = self._gs(self.GS_MSG)
        self.assertEqual(gs['sl'], 1)

    def test_score_right(self):
        gs = self._gs(self.GS_MSG)
        self.assertEqual(gs['sr'], 0)

    def test_missing_gs_returns_none(self):
        gs = self._gs('(HJ (n he1)(ax 0.0)(vx 0.0))')
        self.assertIsNone(gs)

    def test_partial_gs_defaults(self):
        gs = self._gs('(GS (t 5.0)(pm BeforeKickOff))')
        self.assertAlmostEqual(gs['t'], 5.0)
        self.assertIsNone(gs['tl'])
        self.assertIsNone(gs['tr'])


class TestTeamSideDetection(unittest.TestCase):
    """Team-side is set from tl/tr in the GS block."""

    GS_LEFT  = '(GS (t 1.0)(pm PlayOn)(tl BlueTeam)(tr RedTeam)(sl 0)(sr 0))'
    GS_RIGHT = '(GS (t 1.0)(pm PlayOn)(tl RedTeam)(tr BlueTeam)(sl 0)(sr 0))'

    def test_team_side_left(self):
        c = _make_client('BlueTeam')
        gs = c._extract_gs(ThesisClient._parse_sexp_tree(self.GS_LEFT))
        if gs['tl'] == c._team:
            c._team_side = 'left'
        self.assertEqual(c._team_side, 'left')

    def test_team_side_right(self):
        c = _make_client('BlueTeam')
        gs = c._extract_gs(ThesisClient._parse_sexp_tree(self.GS_RIGHT))
        if gs['tr'] == c._team:
            c._team_side = 'right'
        self.assertEqual(c._team_side, 'right')


class TestExtractTorsoPos(unittest.TestCase):
    """_extract_torso_pos: RCSSServerMJ 0.2.0 format (p x y z)."""

    TORSO_MSG = '(pos (n torso_pos)(p 3.14 -1.57 0.85))'

    def setUp(self):
        self.c = _make_client()

    def test_torso_pos_values(self):
        tree = ThesisClient._parse_sexp_tree(self.TORSO_MSG)
        pos  = self.c._extract_torso_pos(tree)
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos[0],  3.14, places=5)
        self.assertAlmostEqual(pos[1], -1.57, places=5)
        self.assertAlmostEqual(pos[2],  0.85, places=5)

    def test_missing_torso_returns_none(self):
        tree = ThesisClient._parse_sexp_tree('(GS (t 1.0)(pm PlayOn))')
        pos  = self.c._extract_torso_pos(tree)
        self.assertIsNone(pos)

    def test_old_format_does_not_match(self):
        # The original (wrong) format: (pos (n torso_pos)(pos x y z)) — must return None.
        tree = ThesisClient._parse_sexp_tree(
            '(pos (n torso_pos)(pos 1.0 2.0 3.0))')
        pos  = self.c._extract_torso_pos(tree)
        self.assertIsNone(pos)


class TestParsePlayers(unittest.TestCase):
    """_parse_players_from_tree: player team/id extraction."""

    PLAYER_MSG = ('(P (team RedTeam)(id 2)'
                  '(head (pol 3.5 15.0 -5.0)))')

    TWO_PLAYER_MSG = ('(P (team RedTeam)(id 1)(head (pol 4.0 10.0 -3.0)))'
                      '(P (team BlueTeam)(id 3)(head (pol 2.0 -20.0 0.0)))')

    def setUp(self):
        self.c = _make_client('BlueTeam')

    def _players(self, msg):
        return self.c._parse_players_from_tree(ThesisClient._parse_sexp_tree(msg))

    def test_player_team(self):
        players = self._players(self.PLAYER_MSG)
        self.assertEqual(len(players), 1)
        self.assertEqual(players[0]['team'], 'RedTeam')

    def test_player_id(self):
        players = self._players(self.PLAYER_MSG)
        self.assertEqual(players[0]['id'], 2)

    def test_multiple_players_count(self):
        players = self._players(self.TWO_PLAYER_MSG)
        self.assertEqual(len(players), 2)

    def test_no_pol_player_skipped(self):
        # (P) block with no (pol ...) inside — must be skipped.
        players = self._players('(P (team Red)(id 9))')
        self.assertEqual(len(players), 0)


class TestEmptyAndEdgeCases(unittest.TestCase):
    """Edge cases for parser robustness."""

    def setUp(self):
        self.c = _make_client()

    def test_empty_message_returns_no_gs(self):
        gs = self.c._extract_gs(ThesisClient._parse_sexp_tree(''))
        self.assertIsNone(gs)

    def test_empty_message_returns_no_torso(self):
        pos = self.c._extract_torso_pos(ThesisClient._parse_sexp_tree(''))
        self.assertIsNone(pos)

    def test_no_spaces_between_subexprs(self):
        # GS fields without whitespace padding between sub-exprs.
        msg = '(GS(t 9.99)(pm PlayOn)(tl A)(tr B)(sl 2)(sr 1))'
        gs  = self.c._extract_gs(ThesisClient._parse_sexp_tree(msg))
        self.assertIsNotNone(gs)
        self.assertAlmostEqual(gs['t'], 9.99)


if __name__ == '__main__':
    unittest.main()
