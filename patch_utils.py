"""
patch_utils.py -- patch-awareness for the T1 win model
=======================================================
Shared by train_and_save_B.py and backtester.py so training, backtesting and
the live payload all use the SAME definition of "how old is this data".

THE PROBLEM THIS SOLVES
-----------------------
Champion win-rate features used to be all-time counts: a game on patch 16.18
counted exactly the same as a game from 14 months of patches ago, and nothing
knew that a tournament played on 16.18 is stale relative to a 16.20 Worlds.
Patch also decided the Tips-tab window by "last two labels in the data",
which breaks when one event (e.g. DCGI) carries a label no other event has.

WHAT IT DOES
------------
1. Every champion / role-champion / player-champion count is decayed by
   PATCH DISTANCE: a game played d patches before the reference patch counts
   PATCH_DECAY ** d as much as a game on the reference patch.
   - Training / backtest: the reference is each game's OWN patch, and only
     games BEFORE it are counted (walk-forward -> no self-leakage).
   - Live payload: the reference is the target patch (see below).
2. The target patch is max(latest patch in the data, TARGET_PATCH_OVERRIDE).
   Before Worlds has any 16.20 games, the override makes 16.18 (DCGI) count
   as 2 patches stale instead of "current".
3. The Tips-tab window is "every patch within RANK_PATCH_SPAN of the target",
   widened until it holds MIN_RANK_GAMES games.

Set PATCH_AWARE = False to get the old behaviour back. Set PATCH_DECAY = 1.0
for walk-forward all-time rates (a clean baseline for A/B testing).
"""

# ---- knobs -------------------------------------------------------------
PATCH_AWARE           = True
PATCH_DECAY           = 0.85   # weight lost per patch of age. UNTUNED -- sweep it
                               # with backtester.py (1.0 / 0.9 / 0.85 / 0.75 / 0.65).
                               # 0.85 -> 1 patch old = 85%, 2 = 72%, 5 = 44%, 10 = 20%.
TARGET_PATCH_OVERRIDE = 16.20  # Worlds 2026 patch (26.20 == OE 16.20). The target
                               # is max(this, newest patch in data), so it moves
                               # forward by itself once newer data exists and is
                               # harmless after Worlds. Set None to disable.
RANK_PATCH_SPAN       = 2      # Tips tab: patches within this many of the target
MIN_RANK_GAMES        = 100    # ...widened one patch at a time until this many games
MAX_RANK_SPAN         = 8
PATCHES_PER_YEAR      = 24     # only used to order patches across a season boundary


def patch_index(p):
    """OE patch label (16.18) -> integer patch number. None if missing/unparseable.

    OE labels are zero-padded 'major.minor' (16.01 ... 16.24). major*24+minor
    keeps 16.18 -> 16.20 two steps apart and orders 15.24 -> 16.01 correctly
    (approximately -- a season has ~24 patches, which is exact enough for decay).
    """
    try:
        x = float(p)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    major = int(x)
    minor = int(round((x - major) * 100))
    return major * PATCHES_PER_YEAR + minor


def target_patch_index(data_patch_indices):
    """Reference patch for the live payload / ranking window."""
    idxs = [i for i in data_patch_indices if i is not None]
    if TARGET_PATCH_OVERRIDE is not None:
        o = patch_index(TARGET_PATCH_OVERRIDE)
        if o is not None:
            idxs.append(o)
    return max(idxs) if idxs else None


class PatchDecayStats:
    """(wins, games) per key, exponentially decayed by patch distance.

    Stored as one accumulator per key expressed at the newest patch seen for
    that key, so reads and writes are O(1) regardless of history length.
    """

    def __init__(self, decay):
        self.decay = decay
        self.d = {}  # key -> [wins, games, ref_patch_idx]

    def add(self, key, win, pi):
        s = self.d.get(key)
        if s is None:
            self.d[key] = [float(win), 1.0, pi]
            return
        w, g, ref = s
        if ref is None:
            ref = pi
        if pi is None:
            pi = ref
        if pi is None or pi >= ref:
            f = 1.0 if pi is None else self.decay ** (pi - ref)
            s[0] = w * f + win
            s[1] = g * f + 1.0
            s[2] = pi
        else:  # a game labelled older than the newest we've seen for this key
            f = self.decay ** (ref - pi)
            s[0] = w + win * f
            s[1] = g + f

    def get(self, key, pi):
        """Decayed (wins, games) as seen from patch `pi`."""
        s = self.d.get(key)
        if s is None:
            return 0.0, 0.0
        w, g, ref = s
        if pi is not None and ref is not None and pi > ref:
            f = self.decay ** (pi - ref)
            return w * f, g * f
        return w, g  # same patch, or a read from "before" the newest label


def _shrunk(w, g, prior, k):
    return (w + k * prior) / (g + k)


class PatchAwareRates:
    """Hierarchical champion / role-champion / player-champion rates, patch-decayed.

    Same shrinkage hierarchy as the rest of the pipeline:
      player-champ -> role-champ -> champion -> 0.5
    """

    def __init__(self, decay, k_champ, k_role, k_pc, pc_weight, rc_weight, positions):
        self.k_champ, self.k_role, self.k_pc = k_champ, k_role, k_pc
        self.pc_weight, self.rc_weight = pc_weight, rc_weight
        self.positions = positions
        self.champ = PatchDecayStats(decay)
        self.role = PatchDecayStats(decay)
        self.pc = PatchDecayStats(decay)

    def _pos(self, i):
        return self.positions[i] if i < len(self.positions) else 'unknown'

    def champ_rate(self, c, pi):
        w, g = self.champ.get(c, pi)
        return _shrunk(w, g, 0.5, self.k_champ)

    def role_rate(self, pos, c, pi):
        w, g = self.role.get((pos, c), pi)
        return _shrunk(w, g, self.champ_rate(c, pi), self.k_role)

    def pc_rate(self, pl, pos, c, pi):
        w, g = self.pc.get((pl, c), pi)
        return _shrunk(w, g, self.role_rate(pos, c, pi), self.k_pc)

    def side_features(self, players, picks, pi):
        """-> (mean champion rate, mean blended player/role rate) for one side."""
        cw, bl = [], []
        for i, (pl, c) in enumerate(zip(players, picks)):
            pl, c, pos = pl.strip(), c.strip(), self._pos(i)
            cw.append(self.champ_rate(c, pi))
            bl.append(self.pc_weight * self.pc_rate(pl, pos, c, pi)
                      + self.rc_weight * self.role_rate(pos, c, pi))
        return sum(cw) / len(cw), sum(bl) / len(bl)

    def update(self, players, picks, won, pi):
        """Record one side's game. `won` is 1/0 for that side."""
        for i, (pl, c) in enumerate(zip(players, picks)):
            pl, c, pos = pl.strip(), c.strip(), self._pos(i)
            self.champ.add(c, won, pi)
            self.role.add((pos, c), won, pi)
            self.pc.add((pl, c), won, pi)

    def export(self, pi, pc_pos=None):
        """Final rate dicts (same shapes the app expects) as seen from patch `pi`."""
        pc_pos = pc_pos or {}
        win_champ_rate = {c: self.champ_rate(c, pi) for c in self.champ.d}
        role_champ_rate = {(pos, c): self.role_rate(pos, c, pi) for (pos, c) in self.role.d}
        pc_rate = {}
        for (pl, c) in self.pc.d:
            pos = pc_pos.get((pl, c), 'unknown')
            w, g = self.pc.get((pl, c), pi)
            prior = role_champ_rate.get((pos, c), win_champ_rate.get(c, 0.5))
            pc_rate[(pl, c)] = _shrunk(w, g, prior, self.k_pc)
        return win_champ_rate, role_champ_rate, pc_rate
