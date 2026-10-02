"""Greet, then follow: a RULE wrapped around the shipped follower (`greet`).

The duck searches with the learned follower (`learned:follow-v4` by default),
and the first time the follower's own tracker CONFIRMS a person — two hits
on the same track, observation slot 79 — it asks the reflex tier for one
`bow` skill cycle, then hands the twist back to the follower for good. One
bow per episode: `reset()` is the only thing that re-arms it.

Why a rule and not a sixth follower: "exactly once, on first contact" is one
boolean here and an approximation after two million steps there. The follower
is not retrained, not re-exported and not touched — it keeps stepping
underneath the whole time (so its tracker and decision cadence stay warm),
and this class only decides WHOSE intent reaches the world.

The world side (`world/arena.py`): `Intent.skill="bow"` → `World.start_skill`
swaps `bow.onnx` in for `BOW_S` seconds, zeroes every command meanwhile and
swaps back; `Senses.skill` reads "bow" for exactly that window. A missing
bow policy makes `start_skill` return False silently, so this brain waits at
most `BOW_WAIT_S` for the world to pick the skill up and then follows anyway,
saying so in its note (and in record-world's events.txt).
"""

from __future__ import annotations

from .runtime import REGISTRY, Intent, Senses

OBS_CONFIRMED = 79          # brain_env.senses_to_obs: 1.0 once the track has confirm_hits
BOW_WAIT_S = 0.5            # how long the world gets to start the skill before we give up on it
SETTLE_SPEED = 0.05         # m/s: "standing still" for the bow to start from
SETTLE_MAX_S = 1.0          # give up waiting for a standstill after this and bow anyway
RISE_S = 0.5                # standing still after the bow before the follower turns the duck


class GreetFollow:
    kind = "greet"

    def __init__(self, follow: str = "learned:follow-v4", inner=None, skill: str = "bow"):
        # `inner` is for tests (a stub follower); `follow` is the registry
        # kind the world builds — any brain whose `last_obs` carries the
        # contract's confirmation slot, or failing that, whose `state` says
        # it is tracking.
        self.inner = inner if inner is not None else REGISTRY.make(follow)
        self.skill = skill
        self.reset()

    def reset(self) -> None:
        self.inner.reset()
        self.state = "search"
        self.bowed = False
        self._settle_t: float | None = None
        self._bow_t: float | None = None
        self._rise_t: float | None = None
        self._skill_seen = False
        self._senses: Senses | None = None

    # -- what counts as "a person" -------------------------------------------------
    def _person_confirmed(self) -> bool:
        o = getattr(self.inner, "last_obs", None)
        if o is not None:
            return bool(o[OBS_CONFIRMED] > 0.5)
        return getattr(self.inner, "state", "") in ("tracking", "approach", "hold")

    # -- the brain contract --------------------------------------------------------
    def step(self, senses: Senses) -> Intent:
        self._senses = senses
        inner = self.inner.step(senses)      # ALWAYS: keeps its tracker and cadence warm

        if self.state == "search":
            if not self.bowed and self._person_confirmed():
                self.state, self.bowed, self._settle_t = "settle", True, senses.t
            else:
                return Intent(twist=inner.twist, head=inner.head, note=f"search · {inner.note}")

        if self.state == "settle":
            # Stop BEFORE bowing. Measured (bow_matrix2, zero command after
            # the skill): a bow started at 0.24-0.35 m/s put the duck down at
            # the hand-back to the walker (t = 0.8-1.7 s) in 5 of 8 trials;
            # 0.5 s of zero twist first — the duck reads 0.00 m/s — 0 of 12.
            # Below ~0.2 m/s it never fell, so a slow duck bows at once.
            still = senses.speed is not None and abs(senses.speed) < SETTLE_SPEED
            if not still and senses.t - (self._settle_t or 0.0) < SETTLE_MAX_S:
                return Intent(note=f"person confirmed · settling ({(senses.speed or 0.0):+.2f} m/s)")
            self.state, self._bow_t = "bow", senses.t
            return Intent(skill=self.skill, note="bow")

        if self.state == "bow":
            running = senses.skill == self.skill
            self._skill_seen = self._skill_seen or running
            if running:
                return Intent(note="bowing")
            if not self._skill_seen and senses.t - (self._bow_t or 0.0) < BOW_WAIT_S:
                return Intent(note="bow requested")
            if not self._skill_seen:
                self.state = "follow"
                return Intent(twist=inner.twist, head=inner.head,
                              note=f"bow skipped (no {self.skill} policy) · {inner.note}")
            self.state, self._rise_t = "rise", senses.t

        if self.state == "rise":
            # The bow leaves the trunk pitched forward and the walker is
            # handed an all-zero command for the first time in 1.5 s. Measured
            # (greet_probe, seed 0): resuming the follower's twist on the very
            # next tick — (-0.2, -0.3, -1.0) — put the duck on the floor 0.5 s
            # later; follow-v4 alone in the same room never fell.
            if senses.t - (self._rise_t or 0.0) < RISE_S:
                return Intent(note="rising")
            self.state = "follow"

        return Intent(twist=inner.twist, head=inner.head, note=f"follow · {inner.note}")

    def inputs(self) -> dict:
        out = dict(self.inner.inputs()) if hasattr(self.inner, "inputs") else {}
        out["greet"] = {"state": self.state, "bowed": self.bowed,
                        "skill_running": None if self._senses is None else self._senses.skill}
        return out

    def view(self) -> dict | None:
        v = getattr(self.inner, "view", None)
        return v() if callable(v) else None


REGISTRY.register("greet", GreetFollow)

__all__ = ["BOW_WAIT_S", "GreetFollow"]
