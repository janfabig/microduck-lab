"""The greet brain (brain/greet.py): bow ONCE on the first confirmed person,
then hand the twist to the follower — the state machine on a stub follower,
and one end-to-end check that a real duck in the follow-me room actually
runs the bow skill and goes on following."""

import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.brain import REGISTRY, GreetFollow, Intent, Senses
from microduck_local.brain.greet import BOW_WAIT_S, OBS_CONFIRMED, RISE_S, SETTLE_MAX_S


class StubFollower:
    """A follower that walks forward and reports what we tell it to see."""
    kind = "stub"

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "lost"
        self.last_obs = None
        self.steps = 0

    def see(self, confirmed: bool):
        self.last_obs = np.zeros(80, np.float32)
        self.last_obs[65] = 1.0
        self.last_obs[OBS_CONFIRMED] = 1.0 if confirmed else 0.0
        self.state = "tracking"

    def step(self, senses):
        self.steps += 1
        return Intent(twist=(0.1, 0.0, 0.05), note="inner")

    def inputs(self):
        return {"stub": True}


def test_greet_is_registered():
    assert "greet" in REGISTRY.available()


def test_passes_the_follower_through_until_a_person_is_confirmed():
    inner = StubFollower()
    b = GreetFollow(inner=inner)
    i = b.step(Senses(t=0.0))
    assert b.state == "search" and i.skill is None and i.twist == (0.1, 0.0, 0.05)
    inner.see(confirmed=False)                       # one hit is not a person yet
    i = b.step(Senses(t=0.1))
    assert b.state == "search" and i.skill is None


def test_bows_once_then_follows_for_good():
    inner = StubFollower()
    b = GreetFollow(inner=inner)
    inner.see(confirmed=True)
    i = b.step(Senses(t=1.0, speed=0.3))              # walking: stop first
    assert i.skill is None and i.twist == (0.0, 0.0, 0.0) and b.state == "settle"
    i = b.step(Senses(t=1.1, speed=0.0))              # standing: now bow
    assert i.skill == "bow" and i.twist == (0.0, 0.0, 0.0) and b.state == "bow"
    # the world picked it up: Senses.skill reads "bow" for the cycle
    for t in (1.12, 1.6, 2.1):
        i = b.step(Senses(t=t, skill="bow", speed=0.0))
        assert i.skill is None and i.twist == (0.0, 0.0, 0.0) and b.state == "bow"
    i = b.step(Senses(t=2.62, skill=None, speed=0.0))  # cycle over: stand up straight first
    assert b.state == "rise" and i.twist == (0.0, 0.0, 0.0)
    i = b.step(Senses(t=2.62 + RISE_S, skill=None, speed=0.0))
    assert b.state == "follow" and i.twist == (0.1, 0.0, 0.05) and i.note.startswith("follow")
    # a second confirmation later changes nothing — one bow per episode
    inner.see(confirmed=True)
    i = b.step(Senses(t=9.0, speed=0.0))
    assert i.skill is None and b.state == "follow"
    assert inner.steps == 8                           # the follower stepped on EVERY tick, bow included
    # reset re-arms it
    b.reset()
    assert b.state == "search" and not b.bowed


def test_a_missing_bow_policy_does_not_strand_the_duck():
    inner = StubFollower()
    b = GreetFollow(inner=inner)
    inner.see(confirmed=True)
    assert b.step(Senses(t=0.0, speed=0.0)).skill == "bow"
    i = b.step(Senses(t=0.1, skill=None, speed=0.0))  # the world never started it
    assert b.state == "bow" and i.twist == (0.0, 0.0, 0.0)
    i = b.step(Senses(t=BOW_WAIT_S + 0.05, skill=None, speed=0.0))
    assert b.state == "follow" and "skipped" in i.note and i.twist == (0.1, 0.0, 0.05)


def test_a_duck_that_will_not_stop_bows_anyway_after_the_settle_limit():
    inner = StubFollower()
    b = GreetFollow(inner=inner)
    inner.see(confirmed=True)
    assert b.step(Senses(t=0.0, speed=0.4)).skill is None and b.state == "settle"
    assert b.step(Senses(t=0.5, speed=0.4)).skill is None and b.state == "settle"
    i = b.step(Senses(t=SETTLE_MAX_S + 0.02, speed=0.4))
    assert i.skill == "bow" and b.state == "bow"


def test_inputs_carry_the_greet_state():
    b = GreetFollow(inner=StubFollower())
    b.step(Senses(t=0.0, skill=None))
    assert b.inputs() == {"stub": True, "greet": {"state": "search", "bowed": False, "skill_running": None}}


def test_duck_in_the_room_bows_at_the_person_then_follows():
    """End to end on the real walker, the real follower and the real bow
    policy: the skill must actually run in the world, and the duck must be
    following (not fallen, not frozen) afterwards."""
    import onnxruntime as ort

    from microduck_local.brain.learned import brains_dir
    from microduck_local.world import Duck, Person, Scenario, World
    from microduck_local.world.arena import World as _W
    walker = C.MICRODUCK_RL_DIR.parent / "microduck" / "policies" / "alpha_walking.onnx"
    if not walker.exists():
        pytest.skip("upstream policies not checked out")
    if not (brains_dir() / "follow-v4" / "brain.onnx").exists():
        pytest.skip("brains/follow-v4 not exported")
    bow = _W.skill_path("bow")
    if bow is None or not bow.exists():
        pytest.skip("no bow policy under policies/bow/")
    sess = ort.InferenceSession(str(walker))
    nm = sess.get_inputs()[0].name
    sc = Scenario(name="greet", floor=(6.0, 6.0),
                  ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "ideal", "ideal")],
                  persons=[Person("p0", (1.2, 0.0), 1.57, path=[(1.2, 1.2), (-1.2, 1.2)], speed=0.2)])
    world = World(sc, infer_for={"d0": lambda o: sess.run(None, {nm: o[None]})[0][0].astype(np.float32)})
    d = world.ducks["d0"]
    brain = GreetFollow()
    skill_ticks, states = 0, []
    for _ in range(int(14.0 / C.CTRL_DT)):
        tof, det = d.tof.last, d.detector.last
        senses = Senses(t=world.t, tof=tof, tof_age=None if tof is None else world.t - tof.t,
                        det=det, det_age=None if det is None else world.t - det.t,
                        speed=d.heading_speed(world.data), odom=world.odom(d), skill=d.skill)
        intent = brain.step(senses)
        world.apply_intent(d, intent)
        d.set_cmd(world.data, intent.twist)
        world.step()
        skill_ticks += d.skill == "bow"
        if not states or states[-1] != brain.state:
            states.append(brain.state)
    assert states == ["search", "settle", "bow", "rise", "follow"], states
    assert skill_ticks >= int(1.0 / C.CTRL_DT), skill_ticks      # the bow really ran (BOW_S = 1.5 s)
    assert d.falls == 0
    assert brain.inner.state == "tracking"                          # and the follower has the person
