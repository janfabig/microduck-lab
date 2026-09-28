# MOSS pick retrain from yard handover states — results (2026-09-27)

* **8956a1** = 478dad + `handover_states` (771 yard creep states, seeds 13-72) +
  `publish_grip` (3-D grip fix in slots 6/13) + `deep_grip_m` 0.035, 3M steps.
  Held-out yard states (seeds 1-12, 150) in the env: picked 97 v 478dad 86; shallow settled
  grips 6/38 v 11/31; balls 9/18 v 5/18, cans 25/31 v 22/31. moss-yard: seeds 1-24 100 v 94,
  25-48 93 v 96 — **48 seeds 193 v 190, paired +0.06 +- 0.30: null in the bin.** Its first
  24 seeds kept more carries (101 v 87; balls 20 v 13) and it did not last. Not shipped.
* **f726a3** = 478dad + `sphere_centre_m` 0.016 + `ball_frac` 0.5, 2M steps. Env ball centre
  from the pad midpoint median 17.6 mm v 14.1, over 20 mm 24/60 v 9/54 — WORSE; yard 88 v 94.
  Not shipped.
* **Card "visibility"** was not the problem: with the current brain and mount the card is
  detected 93-886 frames per 300 s and targeted 6-21 s; it sits in a corner the search reaches
  at 130-200 s. The problem is small-object PICKS in the yard: attempts reaching a lift — cap
  0%, paper 0%, butt 0%, card 6%, block 10% (can 79%) — while the env, from the SAME yard
  states, picks cap 88%, block 79%, card 43%. `min_grasp_m` (20 mm, a can's number) is one
  cause: a floor of 0 for objects sized under 8 cm lifted far more small objects (carries
  125 -> 161) but the bin did not follow — 103 v 94 on 478dad (paired +0.38 +- 0.46, halves
  disagree), +0.12 +- 0.12 on 8956a1. Reverted (patch `min_grasp_small.patch` in scratch).
* Reading: better picks and more carries have now failed to move the bin count four times.
  The count is limited somewhere else — next is a TIME budget of a 300 s run (search,
  approach, pick attempts, carry, release, fold) to see where the seconds go.

## Where the 300 s go, and keeping the arm out between retries (2026-09-27)

Time budget, 478dad, 12 moss-yard runs: deploy 79.5 s/run (26%, x13.3, 6.0 s each), tuck
76.3 s (25%, x12.9, 5.9 s), creep 70.7 s (24%, x13.3, 5.3 s), stow 38.4 s (13%, x5.4), the
rest 12% — ~13 pick attempts at ~17 s each for 3.4 deliveries. This is why better picks,
more carries and fewer drops have not moved the bin count: each attempt pays 12 s of arm
travel whatever its outcome.
* KEEP THE ARM OUT between retries on the same object (`retry_deployed`: a timed-out pick
  with its target still in sight goes back to `deploy`, not `tuck`): 48 seeds 163 v 190,
  paired -0.56 +- 0.28, both halves. Tuck 76 -> 51 s, but deploy did not get cheaper (87 s,
  x17.2 — re-aligning into the band costs ~5 s from the pick pose too), pick attempts rose
  13.3 -> 16.8 and deliveries fell 3.42 -> 3.17: a pick that failed once fails again from
  the same spot, and the tuck-and-approach cycle came back from a different angle. Reverted
  (patch `retry_deployed.patch` in scratch).
* WHERE DEPLOY'S TIME REALLY GOES: the arm reaches the grasp pose in 1.2 s (15 of 20
  deploys); the rest is the base creeping into the handover band at 0.06 m/s, and 7 of 20
  deploys ran to the 9 s cap. `band_mps` 0.12: **48 seeds 225 v 190 in the bin, paired
  +0.73 +- 0.29 (+0.96 / +0.50 per half)** — the first bin gain since the 25 Hz fix. 0.18 is
  no better (-0.12 +- 0.30 v 0.12); `band_timeout_s` 1.0 instead is worse (77 v 94 on 24).
  Now the default, with a test that fails at 0.06.

## Shipped 2026-09-28: the small-object grasp fix and the depth-camera pick

48 moss-yard seeds x 300 s, paired on the same seeds, objects in the bin:
478dad 190 -> + `band_mps` 0.12: 225 -> + `min_grasp_small_m` 0 (a small object may be lifted
with the jaws below 20 mm; paper 0 -> 28 kept, caps 5, blocks 28): **281** (+1.17 +- 0.23) ->
+ `teach-moss_pick_grip-3d2aa6-s6`: **358** (+1.60 +- 0.27 v 281, +1.67 / +1.54 per half).
The new pick (MOSS_PICK_GRIP, from scratch, six gated stages: 20/20/20/20/19 of 20) sees the
object in its jaws (grip_xyz), trains base-free with 15 s to re-grasp, from yard handover
states, deep grips only. Held-out yard states in the env: 121/150 v 478dad 86. Balls kept 47
(3 lost) v 28 (27 lost); caps 28, butts 24. Weak spot: cans lost in the lift (74 of 200).
Stage-1 lesson: from scratch, 478dad's locked base and randomised wrist stop the first grab
(8/20 and 14/20 v 19/20 plain; 4/20 with all of them; every penalty on: 0/20).
Keep-the-arm-out between retries lost again with the faster alignment (216 v 225): removed.
* Litter fine-tune fa3707 (s6 + litter + handover_frac 0.5): env butts 11/30 v 4/30, others
  unchanged; yard 48 seeds 362 v 358 (+0.08 +- 0.21) — a trade (paper 43 v 26 kept, butts 15
  v 24, balls 37 v 47). Not shipped. The shipped pick's card and butt problem in the yard is
  the CARRY: card 19 carries -> 12 lost in the stow swing, 6 in the lift, 1 kept; butt 44
  carries -> 20 lost. Next: trace a card carry (squeeze, lift ramp, swing) as the ball's was.
* THE CARD SLIPS IN THE CARRY, not the grab: traced, it touches only the two pads for the
  whole carry and leaves them in one tick at a jerk (the lift ending, a leg change in the
  swing) — a 4 mm card is pinched by its edges. `carry_ease` (smoothstep on every lift/stow
  leg): 48 seeds **382 v 358** (paired +0.50 +- 0.22, halves +0.54 / +0.46); paper kept 37 v
  26, cards lost in the lift 1 v 6 — the card is still not delivered (8 lost in the stow).
  Carrying small objects twice as slowly lost (333; 337 with easing). Easing is on.

## The scripted top-down pinch (2026-09-28): works mechanically, costs too much time

`pinch_small` (off): drive in on a LOCKED target (the head camera's fix jumps to another toy
as a small thing leaves its view — 9/12 butt pinches drove off toward one), hover over it,
look down with the wrist camera (aimed at the locked target, not the nearest object), roll
the wrist, correct for arm sag, descend straight, close slowly, rise straight 6 cm, lift.
`robots/moss_pinch.py` is the arm's own FK/IK in the base frame (frame check: 0.0 mm).
Env prototype: butts 24/30 along / 11/30 across (pads stand 8 mm apart shut; a butt is 7-9
mm) v 4/30 for the learned pick, which shoves the half-gram butt away (22/30 moved > 3 cm).
Yard, 48 seeds, with the other session's lift gate and the eased carry: pinch OFF 390; ON
(< 8 cm) 345 — it also took the cans; ON (< 4.5 cm) 369 (paired -0.44 +- 0.24): caps 38 v 31,
butts 8 v 4, but cans 106 v 120, balls 29 v 39 — each pinch is 8-10 s of approach, look,
hover, look, align, descend, close, rise, often ending in a fallback. Off. When it completes
on a butt it holds it (6/7 in a trace). Next if pursued: a faster pinch (shorter phases), or
use it as demonstrations for the learned pick. Also: with the lift gate, butts kept fell 21
-> 4 (end-to-end grips sit beyond its 3 cm) — reported to the gate's session.
* Slow swing for EVERY carry under 8 cm (not just pinched ones), 96 seeds: 874 v 878 (-0.04
  +- 0.12) — null; the card stays at 0 delivered: held by its 4 mm edges it slips on any
  swing. Not landed. State at c650a4f: 878 over 96 seeds (9.1 per 5-min run; 4.0 at the start
  of 2026-09-27). Remaining losses: cans (48 lift + 60 stow of 324), paper in the lift (45),
  the card (never delivered). Next levers are no longer brain knobs: a pick that scores the
  SETTLED DEEP GRIP the brain lifts on (the gate session's suggestion), and the fold.
* SETTLED-GRIP RETRAIN (50e9a1 = s6 + success on "both pads held 0.35 s, and within 3 cm for
  objects >= 8 cm", the lift gate's handover), 2M steps: 96 seeds 882 v 878 (+0.04 +- 0.14) —
  null. Cans kept 242 v 224 and lift losses 114 v 128, balls 83 v 88. The gate already refuses
  anything but a deep grip at the handover, so scoring it in the env adds little. Not shipped;
  the `settled_grip` knob removed. The yard stays at c650a4f: 878 / 96 seeds.

## The fold: a shorter route, not a retrain (2026-09-28)

* `tuck` was 88 s of a 300 s run (29%, ~12 folds). The learned fold (81875e) moves at its
  0.75 rad/s command cap THE WHOLE WAY — 7.8 s on every delivery — so it is not slow, its
  route is long: paid to pass `moss.RETRACT_WAYPOINT` (chosen to serve 14 different delivery
  poses) it travels 3.0 rad out and 3.2 back from a release pose 2.0 rad from home. This brain
  releases from ONE pose (12/15 folds). A planner on the robot's own collision model found a
  waypoint from it whose route costs 2.0 rad of the slowest joint (the roll) — 2.5 s.
* `fold_route` (scripted, rate-capped, leashed, only from the release pose), moss-yard 96
  seeds: **876 v 848 (+0.29 +- 0.16)**, tuck 86 -> 50 s/run, 0 bin contacts on 888 route
  folds. At 1.2 rad/s: +0.15 +- 0.15 and more contact — the cap stays at 0.75.
* The rise in contact was NOT the route: it was the learned fold handed an arm out in FRONT
  after a missed pick (it trained from over-the-bin starts only) — 7/15 such folds scraped,
  up to the 12 s budget. `missed_pick_straight` sends those home by the drop path: **893 v
  848 (+0.47 +- 0.15, halves +0.46/+0.48); +0.18 +- 0.10 over the route alone.** Both ON.
* Baseline note: the same 96 seeds scored 848 today against 878 at c650a4f — the live tree
  carries another session's uncommitted `moss_env.py`/`vec_env.py` edits; both arms of every
  pair ran on the same tree.
* Open: straight-home paths (missed picks, and the older drop path from lift/stow) still
  brush the hull, ~26-58 ticks a fold. Planned waypoints are start-specific (one miss pose had
  no single-waypoint route), so the fix is a two-stage route through a raised pose. No fold
  retrain is needed for this gain.

## Seeing, remembering, searching: the end game (2026-09-28)

Asked on /sim with one bottle cap left: "why can't it see that? didn't it see it before? do we
need SLAM?" Three findings, three changes.

* **The cap is below the detector's pixel gate.** A 15 mm cap is found on every frame only
  inside 0.31 m and never past 1.22 m (0.70 / 2.81 deg at 640 px over 87 deg). Nothing on the
  page said so. `Detector.explain` (same geometry as `measure`, no noise, no RNG draws) now
  labels every object on /sim: "MOSS sees ...", "can't see cap0: too small from 1.7 m (needs
  < 1.2 m)", "can't see ...: its own arm is in the way". MOSS/MARS frames only (`det.why`).
  The front RealSense's DEPTH is not modelled at all — only its colour detector.
* **No object memory.** The brain kept only its current target, so anything seen while busy
  was forgotten, and with nothing in view it spun on the spot. Late in 15-min runs 10 of the
  last 11 approaches (two seeds) were at PHANTOM detections (the detector's modelled false
  positives) — the only thing that ever moved it somewhere new.
* **`object_memory` + `patrol` (brain/moss_search.py), both ON.** Remember every toy seen,
  in the odometry frame, weighted by sighting precision (range-scaled gate — a flat 0.15 m
  gate turned one far can into a streak of five); approach only after two sightings in one
  place; forget a spot when the camera looks straight at it from close enough and sees
  nothing, or when a lift starts there. With nothing in view after a full turn: go back to
  the nearest remembered object, else drive the rim of the work area (0.45 m in from the
  walls) facing the middle at each waypoint. 48 seeds x 15 min, objects TRULY in the bin:
  **486 v 469 (+0.35 +- 0.10, halves +0.46/+0.25)**; all ten grippable objects binned in
  48/48 runs (39/48 before; the cap was left 11 times); phantom approaches 188 -> 0; search
  464 -> 258 s/run. At 5 min 444 v 439 (+0.10 +- 0.17) — the benchmark rarely reaches the
  end game, so it neither gains nor loses. The card is the only thing left (42/48).
* The work AREA is declared (the scenario's wall rectangle via `runtime.attach_world`), not
  sensed: MOSS has no range sensor here. Modelling the RealSense depth would let it map the
  walls itself — the depth/SLAM session's thread.
* /sim: "MOSS's map · what it believes" (map toggle) draws the memory, the floor it has
  looked at, the rim route and its current leg, beside the camera overlay's explanation of
  what it can really see.
* Counting note: the brain's own `picked` and the objects truly in the bin differ (a
  "dropped" pinch that lands in the bin anyway) — new yard numbers here count the bin.
