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
