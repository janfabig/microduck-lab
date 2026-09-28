"""MOSS's model of the room (`brain/moss_search.py`, `TidyMoss.object_memory`
and `.patrol`): remember what was seen, never chase a one-frame phantom, and
go looking when nothing is in view."""
import dataclasses
import math
import random
from types import SimpleNamespace

from microduck_local.brain.moss_search import ObjectMemory, Patrol, area_from_world
from microduck_local.sensors.detector import Detection, DetectionFrame
from microduck_local.world.scenario import Wall


def test_a_far_object_is_one_memory_not_a_streak():
    """A can at 1.5 m ranged from its apparent width (~10% noise) is ONE
    remembered object near where it is — the first cut's flat 0.15 m gate
    made a streak of five along the line of sight."""
    random.seed(0)
    m = ObjectMemory()
    for k in range(40):
        r = 1.5 * (1 + random.gauss(0, 0.1))
        m.observe([(r, random.gauss(0, 0.02), 0.115, r)], 0.1 * k)
    confirmed = [e for e in m.items if m.confirmed(e)]
    assert len(confirmed) == 1
    assert abs(confirmed[0].x - 1.5) < 0.08


def test_a_phantom_is_never_confirmed_and_is_forgotten():
    m = ObjectMemory()
    m.observe([(0.3, 0.9, 0.05, 0.9)], 0.0)
    assert not m.confirmed(m.items[0])
    m.looked([], set(), m.tentative_s + 0.1)
    assert m.items == []


def test_looking_right_at_a_remembered_spot_and_not_seeing_it_forgets_it():
    m = ObjectMemory()
    for k in range(3):
        m.observe([(0.6, 0.0, 0.1, 0.6)], 0.1 * k)
    e = m.items[0]
    for _ in range(m.miss_frames - 1):
        m.looked([e], set(), 1.0)
    assert m.items == [e]                      # a few missed frames: still there
    m.looked([e], set(), 1.0)
    assert m.items == []                       # looked long enough: gone


def test_the_rim_patrol_is_inset_from_the_yard_walls_and_goes_round():
    yard = SimpleNamespace(scenario=SimpleNamespace(walls=[
        Wall((-1.5, -1.2), (1.5, -1.2)), Wall((1.5, -1.2), (1.5, 1.2)),
        Wall((1.5, 1.2), (-1.5, 1.2)), Wall((-1.5, 1.2), (-1.5, -1.2))]))
    area = area_from_world(yard)
    assert area == (-1.5, 1.5, -1.2, 1.2)
    p = Patrol(area, inset_m=0.45)
    assert all(-1.05 - 1e-9 <= x <= 1.05 + 1e-9 and -0.75 - 1e-9 <= y <= 0.75 + 1e-9
               for x, y in p.waypoints)
    first = p.next_waypoint(-0.9, 0.0)
    assert first == (-1.05, 0.0)               # the nearest, from the spawn
    seen = {first} | {p.next_waypoint(0, 0) for _ in range(7)}
    assert len(seen) == 8                      # every waypoint once round the loop
    assert area_from_world(SimpleNamespace(scenario=SimpleNamespace(walls=[]))) is None


def _frame(t, bearing, rng, name="can0", radius=0.0575):
    w = 2.0 * math.atan(radius / rng)
    return DetectionFrame(t=t, detections=[Detection("toy", name, bearing, -0.05, w, rng, 1.0)])


def _brain(**kw):
    from microduck_local.brain.tidy_moss import TidyMoss
    b = TidyMoss()
    b.p = dataclasses.replace(b.p, **kw)
    return b


def test_with_memory_a_one_frame_phantom_is_not_a_target():
    from microduck_local.brain.runtime import Senses
    b = _brain(object_memory=True)
    b._to("search", 0.0)
    s = Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.0, 0.2, 0.9, name=""))
    b.step(s)
    assert b.state == "search"                 # seen once: could be a phantom
    b.step(Senses(t=0.04, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.03, 0.2, 0.9, name="")))
    assert b.state == "approach"               # seen again in the same place: real
    # ...and without memory the first frame is enough (what chased phantoms).
    b = _brain(object_memory=False)
    b._to("search", 0.0)
    b.step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, det=_frame(0.0, 0.2, 0.9, name="")))
    assert b.state == "approach"


def test_with_nothing_in_view_the_patrol_drives_to_the_rim_after_a_full_turn():
    from microduck_local.brain.runtime import Senses
    yard = SimpleNamespace(scenario=SimpleNamespace(walls=[
        Wall((-1.5, -1.2), (1.5, -1.2)), Wall((1.5, -1.2), (1.5, 1.2)),
        Wall((1.5, 1.2), (-1.5, 1.2)), Wall((-1.5, 1.2), (-1.5, -1.2))]))
    b = _brain(object_memory=True, patrol=True)
    b.world = yard
    b._to("search", 0.0)
    x, y, yaw, t, dt = 0.0, 0.0, 0.0, 0.0, 0.02
    drove = False
    for _ in range(int(20 / dt)):
        t += dt
        it = b.step(Senses(t=t, odom=(x, y, yaw), speed=0.0))
        vx, _vy, wz = it.twist
        yaw += wz * dt
        x += vx * math.cos(yaw) * dt
        y += vx * math.sin(yaw) * dt
        if vx > 0.05:
            drove = True
            assert t > 2 * math.pi / b.p.search_wz - 0.5    # only after a full turn
            break
    assert drove
    assert b._srch["kind"] == "rim"
