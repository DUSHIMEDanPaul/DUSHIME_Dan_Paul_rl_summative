"""
SafeBuild Inspector: a custom Gymnasium environment.

Mission
-------
Many homes in fast-growing African cities are built on river floodplains,
in wetlands or on steep, landslide-prone slopes, often without approved plans.
In this environment an inspection drone surveys a small district and must
decide, building by building, whether each structure is SAFE or must be
FLAGGED as violating building rules / sitting in a hazard zone.

The drone sees the terrain around it (like GIS hazard maps: flood risk and
slope), but some violations (no approved plan, setback breaches, unsafe
foundations) can only be found by INSPECTING the building on site.

Real-world mapping of actions
-----------------------------
0 MOVE_NORTH   - fly one cell north
1 MOVE_SOUTH   - fly one cell south
2 MOVE_EAST    - fly one cell east
3 MOVE_WEST    - fly one cell west
4 INSPECT      - close-range scan of the building below (reveals hidden
                 code violations such as missing permit / setback breach)
5 FLAG_UNSAFE  - report the building to the authority as unsafe / illegal
6 CLEAR_SAFE   - record the building as compliant and safe

A building is truly UNSAFE if ANY of these is true:
  * it sits in the river floodplain  (flood hazard)
  * it sits on a steep slope          (landslide hazard)
  * it has a hidden code violation    (only visible after INSPECT)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_NAMES = {
    0: "MOVE_NORTH",
    1: "MOVE_SOUTH",
    2: "MOVE_EAST",
    3: "MOVE_WEST",
    4: "INSPECT",
    5: "FLAG_UNSAFE",
    6: "CLEAR_SAFE",
}

# Reward structure (kept in one place so the report and code always agree)
REWARDS = {
    "step": -0.05,             # time / battery cost of every action
    "bump_wall": -0.5,         # trying to fly off the map
    "inspect_new": 0.5,        # useful inspection of an unclassified building
    "inspect_wasted": -0.2,    # inspecting empty ground or a re-inspection
    "flag_correct": 10.0,      # unsafe building correctly reported
    "flag_wrong": -6.0,        # safe building wrongly reported (false alarm)
    "clear_correct": 5.0,      # safe building correctly cleared
    "clear_wrong": -10.0,      # unsafe building missed (most costly error)
    "invalid_decision": -0.2,  # flag/clear on empty ground or a decided building
    "mission_complete": 15.0,  # every building classified
    "approach": 0.2,           # potential-based shaping per cell closer to nearest undecided building
}

WINDOW = 2  # local view radius -> 5x5 window


class SafeBuildInspectorEnv(gym.Env):
    """Grid-world inspection mission with hidden building-code violations."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(
        self,
        grid_size: int = 10,
        n_buildings: int = 8,
        max_steps: int = 150,
        hidden_violation_prob: float = 0.3,
        render_mode: str | None = None,
        renderer: str = "auto",
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.grid_size = grid_size
        self.n_buildings = n_buildings
        self.max_steps = max_steps
        self.hidden_violation_prob = hidden_violation_prob
        self.render_mode = render_mode
        self.renderer_choice = renderer
        self._renderer = None

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        win = 2 * WINDOW + 1
        # 4 global + 5 current-cell + 3 nearest-target + 3 x (5x5) local map
        self.obs_dim = 4 + 5 + 3 + 3 * win * win
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------ map
    def _generate_map(self):
        g, rng = self.grid_size, self.np_random

        # Smooth random elevation (0..1)
        raw = rng.random((g + 4, g + 4))
        k = np.ones((3, 3)) / 9.0
        smooth = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                smooth[i, j] = (raw[i:i + 3, j:j + 3] * k).sum()
        # add a hill on one side so steep slopes exist
        hx, hy = rng.integers(0, g), rng.integers(0, g)
        yy, xx = np.mgrid[0:g, 0:g]
        hill = np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * (g / 5) ** 2))
        elev = 0.5 * smooth + 0.9 * hill
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-8)

        # River: random walk from north to south, flows through low ground
        river = np.zeros((g, g), dtype=bool)
        x = int(rng.integers(2, g - 2))
        for y in range(g):
            river[y, x] = True
            x = int(np.clip(x + rng.integers(-1, 2), 1, g - 2))
        elev[river] = 0.0

        # Flood plain = low cells next to the river
        flood = np.zeros((g, g), dtype=np.float32)
        for y in range(g):
            for x in range(g):
                if river[y, x]:
                    continue
                d = np.min(np.abs(np.argwhere(river) - [y, x]).sum(axis=1))
                if d <= 2:
                    flood[y, x] = float(np.clip((3 - d) / 2 * (1.0 - elev[y, x]), 0, 1))
        # Slope = local gradient magnitude, normalised
        gy, gx = np.gradient(elev)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        slope = np.clip(slope / (slope.max() + 1e-8), 0, 1).astype(np.float32)

        self.elevation = elev.astype(np.float32)
        self.river = river
        self.flood_risk = flood
        self.slope = slope

        # Buildings: about half placed in hazard zones, half on safe ground
        free = [(y, x) for y in range(g) for x in range(g)
                if not river[y, x] and (y, x) != (0, 0)]
        hazard = [c for c in free if flood[c] > 0.5 or slope[c] > 0.6]
        safe = [c for c in free if c not in hazard]
        rng.shuffle(hazard)
        rng.shuffle(safe)
        n_haz = min(len(hazard), self.n_buildings // 2)
        cells = hazard[:n_haz] + safe[: self.n_buildings - n_haz]

        self.buildings = {}
        for c in cells:
            terrain_unsafe = bool(flood[c] > 0.5 or slope[c] > 0.6)
            hidden = bool(rng.random() < self.hidden_violation_prob)
            self.buildings[c] = {
                "terrain_unsafe": terrain_unsafe,
                "hidden_violation": hidden,
                "unsafe": terrain_unsafe or hidden,
                "inspected": False,
                "status": "unknown",  # unknown | flagged | cleared
            }

    # -------------------------------------------------------------- helpers
    def _unclassified(self):
        return [c for c, b in self.buildings.items() if b["status"] == "unknown"]

    def _nearest_target(self):
        targets = self._unclassified()
        if not targets:
            return 0.0, 0.0, 0.0
        ay, ax = self.agent
        d = [abs(y - ay) + abs(x - ax) for y, x in targets]
        y, x = targets[int(np.argmin(d))]
        g = self.grid_size - 1
        return (x - ax) / g, (y - ay) / g, min(d) / (2 * g)

    def _potential(self):
        """Phi(s) = -approach * distance to nearest undecided building (0 when none left).

        Shaping F = Phi(s') - Phi(s) is potential-based (Ng et al., 1999), so it guides the
        drone towards buildings without changing which policy is optimal.
        """
        targets = self._unclassified()
        if not targets:
            return 0.0
        ay, ax = self.agent
        return -REWARDS["approach"] * min(abs(y - ay) + abs(x - ax) for y, x in targets)

    def _get_obs(self):
        g = self.grid_size
        ay, ax = self.agent
        cell = self.buildings.get(self.agent)
        is_bld = 1.0 if cell and cell["status"] == "unknown" else 0.0
        inspected = 1.0 if cell and cell["inspected"] else 0.0
        revealed = 1.0 if cell and cell["inspected"] and cell["hidden_violation"] else 0.0

        glob = [
            ax / (g - 1),
            ay / (g - 1),
            1.0 - self.steps / self.max_steps,
            len(self._unclassified()) / max(1, self.n_buildings),
        ]
        current = [is_bld, float(self.flood_risk[ay, ax]), float(self.slope[ay, ax]),
                   inspected, revealed]
        target = list(self._nearest_target())

        win = 2 * WINDOW + 1
        bmap = -np.ones((win, win), dtype=np.float32)
        fmap = -np.ones((win, win), dtype=np.float32)
        smap = -np.ones((win, win), dtype=np.float32)
        for dy in range(-WINDOW, WINDOW + 1):
            for dx in range(-WINDOW, WINDOW + 1):
                y, x = ay + dy, ax + dx
                if 0 <= y < g and 0 <= x < g:
                    b = self.buildings.get((y, x))
                    bmap[dy + WINDOW, dx + WINDOW] = 1.0 if b and b["status"] == "unknown" else 0.0
                    fmap[dy + WINDOW, dx + WINDOW] = 1.0 if self.river[y, x] else self.flood_risk[y, x]
                    smap[dy + WINDOW, dx + WINDOW] = self.slope[y, x]

        obs = np.concatenate([glob, current, target,
                              bmap.ravel(), fmap.ravel(), smap.ravel()]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _info(self):
        flagged_ok = sum(1 for b in self.buildings.values() if b["status"] == "flagged" and b["unsafe"])
        false_alarms = sum(1 for b in self.buildings.values() if b["status"] == "flagged" and not b["unsafe"])
        cleared_ok = sum(1 for b in self.buildings.values() if b["status"] == "cleared" and not b["unsafe"])
        missed = sum(1 for b in self.buildings.values() if b["status"] == "cleared" and b["unsafe"])
        return {
            "agent_pos": tuple(self.agent),
            "battery": 1.0 - self.steps / self.max_steps,
            "remaining": len(self._unclassified()),
            "unsafe_total": sum(b["unsafe"] for b in self.buildings.values()),
            "correct_flags": flagged_ok,
            "false_alarms": false_alarms,
            "correct_clears": cleared_ok,
            "missed_unsafe": missed,
            "last_event": self.last_event,
        }

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._generate_map()
        self.agent = (0, 0)  # start state: drone at the depot (north-west corner)
        self.steps = 0
        self.last_action = None
        self.last_event = "mission start"
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._info()

    def step(self, action: int):
        action = int(action)
        r = REWARDS["step"]
        phi_before = self._potential()
        self.steps += 1
        self.last_action = action
        ay, ax = self.agent
        cell = self.buildings.get(self.agent)
        event = ACTION_NAMES[action]

        if action <= 3:  # movement
            dy, dx = [(-1, 0), (1, 0), (0, 1), (0, -1)][action]
            ny, nx = ay + dy, ax + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                self.agent = (ny, nx)
            else:
                r += REWARDS["bump_wall"]
                event = "hit map boundary"
        elif action == 4:  # inspect
            if cell and cell["status"] == "unknown" and not cell["inspected"]:
                cell["inspected"] = True
                r += REWARDS["inspect_new"]
                event = ("inspection: hidden violation found" if cell["hidden_violation"]
                         else "inspection: no hidden violation")
            else:
                r += REWARDS["inspect_wasted"]
                event = "wasted inspection"
        else:  # decisions
            if not cell or cell["status"] != "unknown":
                r += REWARDS["invalid_decision"]
                event = "invalid decision (no undecided building here)"
            elif action == 5:
                cell["status"] = "flagged"
                r += REWARDS["flag_correct"] if cell["unsafe"] else REWARDS["flag_wrong"]
                event = "FLAGGED correctly" if cell["unsafe"] else "FALSE ALARM (safe building flagged)"
            else:
                cell["status"] = "cleared"
                r += REWARDS["clear_wrong"] if cell["unsafe"] else REWARDS["clear_correct"]
                event = "MISSED HAZARD (unsafe building cleared)" if cell["unsafe"] else "cleared correctly"

        self.last_event = event
        r += self._potential() - phi_before  # potential-based shaping
        terminated = len(self._unclassified()) == 0
        if terminated:
            r += REWARDS["mission_complete"]
        truncated = (not terminated) and self.steps >= self.max_steps

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(r), terminated, truncated, self._info()

    # ------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import make_renderer
            self._renderer = make_renderer(self, self.render_mode, self.renderer_choice)
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
