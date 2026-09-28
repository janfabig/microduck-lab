"""MOSS's model of the room: what it has seen, where it has looked, and where
to go when nothing is in view (2026-09-28).

Asked on /sim with one bottle cap left in `moss-yard`: "didn't it see that
cap before? why doesn't it go over there? do we need a map?" MOSS had no
memory of objects — only the one it was going for — and when nothing was in
view it spun on the spot, forever, unless a one-frame phantom detection (the
detector's modelled false positive, ~1 per 20 s) sent it across the yard.
Measured over two 15-minute runs: once only small objects were left, 10 of
the approaches were at phantoms, and the only way it ever reached the cap was
by chasing them.

Three pieces, all in the brain's ODOMETRY frame and built only from what the
robot has (its detections and its own pose):

* `ObjectMemory` — every toy seen, kept after it leaves the view. A sighting
  is CONFIRMED on its second detection within `gate_m`, which is also what
  keeps a phantom out: it never recurs in the same place. An entry is
  forgotten when the camera looks right at where it should be, from close
  enough that it would be found on every frame, and finds nothing.
* `Coverage` — when each patch of floor was last in the camera's view,
  within `cover_m`. For the map on /sim, and for choosing where to look.
* `Patrol` — when a full turn on the spot shows nothing: drive to the
  nearest remembered object and look at it from close up; with nothing
  remembered, drive the rim of the work area, `inset_m` in from its walls,
  turning to face the middle at every waypoint.

The WORK AREA is declared, not sensed: MOSS has no range sensor in this sim
(its RealSense's depth is not modelled), so it cannot find the walls itself.
It is the yard's wall rectangle from the scenario — what a person marks in an
app — and the place a mapped wall line would plug in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Seen:
    """One remembered object, odometry frame."""
    x: float
    y: float
    size: float
    hits: int
    first: float
    last: float
    misses: int = 0
    id: int = 0
    #: Sum of the sightings' weights (1 / sigma^2): a close look outweighs
    #: many distant ones.
    wsum: float = 0.0


class ObjectMemory:
    """Toys seen, kept after they leave the view."""

    #: A sighting's position error, m, at range r: the detector ranges an
    #: object from its apparent width, which is ~10% noisy, so the error is
    #: mostly ALONG the line of sight and grows with range. The first cut used
    #: a flat 0.15 m gate and a plain mean, and one can at 1.5 m became a
    #: streak of five "remembered objects" on the /sim map.
    SIGMA0, SIGMA_PER_M = 0.02, 0.08
    #: Weight cap, so a pushed object's entry can still follow it.
    WSUM_CAP = 5.0 / (0.02 + 0.08 * 0.3) ** 2

    def __init__(self, gate_m: float = 0.10, confirm: int = 2,
                 tentative_s: float = 8.0, miss_frames: int = 5,
                 gate_per_m: float = 0.20):
        self.gate_m = gate_m
        self.gate_per_m = gate_per_m
        self.confirm = confirm
        self.tentative_s = tentative_s
        self.miss_frames = miss_frames
        self.items: list[Seen] = []
        self._next = 1

    def observe(self, pts: list[tuple[float, float, float, float]], t: float) -> list[Seen]:
        """Fold one frame's sightings (x, y, size, range) in; returns the
        entry each one matched or made, in order."""
        out: list[Seen] = []
        taken: set[int] = set()
        for x, y, size, rng in pts:
            wi = 1.0 / (self.SIGMA0 + self.SIGMA_PER_M * rng) ** 2
            best, bd = None, self.gate_m + self.gate_per_m * rng
            for e in self.items:
                if e.id in taken:
                    continue
                d = math.hypot(e.x - x, e.y - y)
                if d < bd:
                    best, bd = e, d
            if best is None:
                best = Seen(x, y, size, 0, t, t, id=self._next, wsum=wi)
                self._next += 1
                self.items.append(best)
            else:
                k = wi / (best.wsum + wi)
                best.x += k * (x - best.x)
                best.y += k * (y - best.y)
                best.size += k * (size - best.size)
                best.wsum = min(best.wsum + wi, self.WSUM_CAP)
            best.hits += 1
            best.last = t
            best.misses = 0
            taken.add(best.id)
            out.append(best)
        self._merge()
        return out

    def _merge(self) -> None:
        """Two entries that have converged on one spot are one object."""
        i = 0
        while i < len(self.items):
            a = self.items[i]
            j = i + 1
            while j < len(self.items):
                b = self.items[j]
                if math.hypot(a.x - b.x, a.y - b.y) < self.gate_m:
                    w = a.wsum + b.wsum
                    a.x = (a.x * a.wsum + b.x * b.wsum) / w
                    a.y = (a.y * a.wsum + b.y * b.wsum) / w
                    a.size = (a.size * a.wsum + b.size * b.wsum) / w
                    a.wsum = min(w, self.WSUM_CAP)
                    a.hits += b.hits
                    a.first, a.last = min(a.first, b.first), max(a.last, b.last)
                    del self.items[j]
                else:
                    j += 1
            i += 1

    def confirmed(self, e: Seen) -> bool:
        return e.hits >= self.confirm

    def near(self, x: float, y: float, gate: float | None = None) -> Seen | None:
        g = self.gate_m if gate is None else gate
        best, bd = None, g
        for e in self.items:
            d = math.hypot(e.x - x, e.y - y)
            if d < bd:
                best, bd = e, d
        return best

    def forget_near(self, x: float, y: float, r: float) -> None:
        self.items = [e for e in self.items if math.hypot(e.x - x, e.y - y) >= r]

    def looked(self, expected: list[Seen], seen_ids: set[int], t: float) -> None:
        """Entries the camera should have found this frame (in view and close
        enough for their size) and did not: after `miss_frames` of those in a
        row the thing is not there any more."""
        for e in expected:
            if e.id not in seen_ids:
                e.misses += 1
        self.items = [e for e in self.items if e.misses < self.miss_frames
                      and (self.confirmed(e) or t - e.last < self.tentative_s)]


@dataclass
class Coverage:
    """When each cell of the work area was last in the camera's view."""
    area: tuple[float, float, float, float]      # xmin, xmax, ymin, ymax
    cell_m: float = 0.2
    last: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.nx = max(1, int(round((self.area[1] - self.area[0]) / self.cell_m)))
        self.ny = max(1, int(round((self.area[3] - self.area[2]) / self.cell_m)))
        self.last = [-1e9] * (self.nx * self.ny)

    def centre(self, i: int, j: int) -> tuple[float, float]:
        return (self.area[0] + (i + 0.5) * (self.area[1] - self.area[0]) / self.nx,
                self.area[2] + (j + 0.5) * (self.area[3] - self.area[2]) / self.ny)

    def mark(self, cam_xy: tuple[float, float], yaw: float, half_fov: float,
             near_m: float, far_m: float, t: float) -> None:
        for j in range(self.ny):
            for i in range(self.nx):
                cx, cy = self.centre(i, j)
                dx, dy = cx - cam_xy[0], cy - cam_xy[1]
                r = math.hypot(dx, dy)
                if not near_m <= r <= far_m:
                    continue
                b = math.atan2(dy, dx) - yaw
                b = math.atan2(math.sin(b), math.cos(b))
                if abs(b) <= half_fov:
                    self.last[j * self.nx + i] = t

    def ages(self, t: float, cap_s: float = 99.0) -> list[int]:
        """Seconds since each cell was seen, capped, row-major from ymin."""
        return [int(min(t - v, cap_s)) for v in self.last]


class Patrol:
    """Where to go when a full turn shows nothing."""

    def __init__(self, area: tuple[float, float, float, float] | None,
                 inset_m: float = 0.45):
        self.area = area
        self.waypoints: list[tuple[float, float]] = []
        if area is not None:
            x0, x1, y0, y1 = area
            x0, x1, y0, y1 = x0 + inset_m, x1 - inset_m, y0 + inset_m, y1 - inset_m
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            # Around the rim, counter-clockwise: corners and edge midpoints.
            self.waypoints = [(x0, y0), (xm, y0), (x1, y0), (x1, ym),
                              (x1, y1), (xm, y1), (x0, y1), (x0, ym)]
        self.i: int | None = None

    @property
    def centre(self) -> tuple[float, float] | None:
        if self.area is None:
            return None
        return ((self.area[0] + self.area[1]) / 2, (self.area[2] + self.area[3]) / 2)

    def next_waypoint(self, x: float, y: float) -> tuple[float, float] | None:
        """The next rim waypoint: the nearest one the first time, then on
        round the loop."""
        if not self.waypoints:
            return None
        if self.i is None:
            self.i = min(range(len(self.waypoints)),
                         key=lambda k: math.hypot(self.waypoints[k][0] - x,
                                                  self.waypoints[k][1] - y))
        else:
            self.i = (self.i + 1) % len(self.waypoints)
        return self.waypoints[self.i]


def area_from_world(world) -> tuple[float, float, float, float] | None:
    """The work area: the scenario's wall rectangle (their bounding box, less
    half a wall), or None when the room has no walls."""
    sc = getattr(world, "scenario", None)
    walls = getattr(sc, "walls", None) or []
    xs, ys = [], []
    for w in walls:
        a, b = getattr(w, "a", None), getattr(w, "b", None)
        if a is None:
            a, b = getattr(w, "start", None), getattr(w, "end", None)
        if a is None or b is None:
            continue
        xs += [float(a[0]), float(b[0])]
        ys += [float(a[1]), float(b[1])]
    if len(xs) < 4:
        return None
    return (min(xs), max(xs), min(ys), max(ys))
