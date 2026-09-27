# MOSS stow: the seat hands back dead starts — the aim-chain bisect, 2026-09-26

**Every varied-shape stow number before this is partly a pick number.** The
stow env's reset seats the object on the floor, closes the jaw and raises it
to the rung's start pose. For six shapes the seat FAILS on ~7 of 8 draws —
ball, card, most off-size cans — and the reset fell through silently, handing
back the object LYING ON THE FLOOR under an open jaw at the grasp pose, at
every rung. `_carrying()` even reported True for it (inside the radius, within
the debounce of `_last_contact = 0`).

* rung 0, six shapes: held at reset **8/60** (ball 0/15, card 0/11, tall 0/3,
  can-family 1/13, block 6/11); the reference can alone: 60/60.
* rung 2 likewise ~5/40 lifted. The shipped stow's ~58% at rung 2 is partly
  the policy RE-PICKING the object from the failed seat, which rung 2's open
  jaws allow; at rung 0 that is the whole job, so the aim chain's stage 1
  (from scratch, rung 0) collapsed to 3/40 in 10-step episodes.

The bisect (rung 0, from scratch, one knob per arm, 34-83k steps): the can-only
control scored +62.6 per episode in 13-step deliveries; every six-shape arm
(-28 to -40) failed whatever else it added — so clutter, the drop point and the
release bonus were never tested by the aim chain at all.

Fix: `valid_start` (`MICRODUCK_MOSS_VALID_START`) — a failed seat-and-carry
redraws the shape (up to 6) then falls back to the reference can: held AND
lifted 59-60/60 at rungs 0 and 2, with and without clutter; ~a quarter to a
third fall back; cards never seat (the separate grasp problem). Default OFF so
old runs replay identically; `test_a_valid_start_stow_episode_begins_...`
fails on the planted regression. Round 2 of the bisect runs on it.

## Round 2 — valid starts (rung 0, from scratch, 250k steps, 60 unseen seeds)

| arm | delivered (spot real / zeroed) | landing err | perched |
|---|---|---|---|
| V0 six shapes | 59/60 | 38 mm | 0 |
| V1 + drop point | 60 / 59 | 40 / 38 mm | 0 / 0 |
| V2 + release bonus 0.6 | 59 / 59 | 36 / 37 mm | 0 / 0 |
| V3 + 0-6 clutter | 52 / 53 | 43 / 46 mm | 26 / 28 |

* The seat WAS the collapse: with valid starts every arm delivers (3/40 -> 52-60/60).
* No arm USES the spot: zeroing slots 28-29 changes nothing. Only V3 tests it (an
  empty bin's spot is always the centre); its +0.3 slope survives zeroing — physics
  deflecting objects toward open space, not aim. The policy lets go in 10-12 steps.
* Aiming would matter: V3, 180 deliveries, perched by release distance from the
  spot — 0-2 cm 5/12, 2-4 16/50, 4-6 11/41, 6-20 cm 56/77 (73%).

## Round 3 — scripted aim, and the FILL test (the metric that matters)

Scripted aim: IK the jaws over the chosen spot (`scratchpad` prototype), 0.03 rad/tick,
open. Findings on the way:
* **Reach**: the jaws only get over the +y ~58% of the bin (pan stops at -1.92 rad; same
  at every height). `data/moss_bin_reach.npz`; `choose_drop_point` now masks by it (1-cell
  margin). Unmasked, a spot past it was 45 mm out of reach and landed ON THE RIM.
* **"Perched" is fill, not aim**: single drops, high/centre, perched by items already in
  the bin: 0:0/28 1:0/22 2:7/21 3:15/29 4:19/29 5:26/30 6:21/23. Resting on other litter in a
  full bin is how a bin fills — a bad success metric.
* Single drops, 200 seeds: low release (2 cm under the rim) wedges on pads/clutter (75%
  delivered); high release (as the learned policy: 11 cm above) 87-91%, aim makes no
  paired difference (10 v 10 perches).

FILL TEST (items delivered one after another into ONE bin, each left where it landed; 10
attempts per bin; same shape sequence for every controller):

| release | aim at spot | aim at centre | sequences |
|---|---|---|---|
| high, items in bin | 6.06 | 6.08 | 36 |
| high, items before 1st miss | 3.72 | 5.00 | 36 |
| mid (+5 cm, lift away) | 5.92 | 5.58 | 12 |
| low (-2 cm, lift away) | 4.58 | 3.00 | 12 (10-0 paired, p~0.002) |

**Verdict: in this sim, aiming does not fill the bin fuller.** It helps only when forced to
release low, and releasing low fills worse. At the best height the centre is as good and
misses LATER (spot aims go toward the edges; the centre leaves bounce room). The bin holds
~6 mixed items. Caveat: placed items are welded where they land (static clutter), so a new
drop cannot shove them — a real bin settles; that could favour either side.

## The "cheap release fix" — a null, and what it says about where the yard loses objects

`STOW_RELEASE_HIGH` (tool point over the bin centre, 11 cm above the rim, carry wrist)
in place of the descent to `STOW_INSIDE`: moss-yard, 6 seeds x 300 s, paired —
**14/36 objects in the bin against 13/36** (per seed +0 +1 +1 +1 -2 +0). In the world
~93% of RELEASES already land in the bin (13 of 14). An env replica of the brain's
carry lost 32/72 at release — the replica was wrong, which is why it was not trained
on. The yard's losses are upstream: block 0/6 and ball 0/6 are almost never gripped,
and fingertip grips (>45 mm from the tool point) never survive the carry (0/7).
`release_high` kept, default OFF. Also found: `_held()`'s 2.0 mm servo-stall test
reads a physically carried can as dropped at the brain's 27 mm carry squeeze.

## Pick retrain, step 1 — the pick policies scored IN THE YARD (6 seeds x 300 s, 6 objects)

| pick | in bin | attempts | lifts | kept past lift | reached release |
|---|---|---|---|---|---|
| teach-moss_pick-5e9df7 (brain's) | 13 | 77 | 39 | 21 | 14 |
| moss-pick-v1 | 12 | 78 | 34 | 19 | 15 |
| teach-moss_pick-d879c3 | 11 | 83 | 44 | 22 | 16 |
| teach-moss_pick-ad9876 (record.json pick) | 4 | 90 | 32 | 18 | 14 |

* The three can-only picks are indistinguishable at 6 seeds (+-1): the yard cannot rank them
  at this sample, so no env criterion can be validated against a ranking among them.
* Funnel (pooled): attempt->lift ~49%, lift->kept ~54%, kept->release ~71%, release->bin ~80%.
  Ball: 0 lifts in any run. Block: 1-3 lifts, 0 kept. Squat: about half.
* ad9876 — the only six-shape pick (88% in its env) — trained with publish_attitude,
  publish_size, publish_proximity AND base_lock. tidy_moss publishes none of them and does
  not lock the base; the world has no wrist camera at all (head-camera detections only).
  Its 4/36 is a deployment mismatch, not its skill: 14 releases, 4 in the bin at the end.

## ad9876 given what it trained with — the wrist camera in the yard (robots/moss_wrist.py)

The world now produces MOSS's wrist/front-camera readings of the object nearest the jaws
by calling the pick env's OWN `_sense_arm` / `_front_attitude` / `_true_attitude`
(`Senses.target_obs`), and tidy_moss fills exactly the slots each leg trained with and
zeroes the base twist for a leg trained base-locked (read from its run.json). Slot diff
yard vs env: all alive; the yard's wrist range is 0.27 m vs 0.06 m in the env (the yard
hands over FARTHER), and yard objects stand (uprightness 0.96 vs 0.38).

ad9876 in the yard, 6 seeds x 300 s (in bin / lifts / kept / releases):
no inputs, base free 4/32/18/14; inputs, base free 5/23/16/14; no inputs, base locked
1/25/12/7; inputs + lock 0/28/7/1 — against 5e9df7's 13/39/21/14. The inputs were not
the gap (+1); the base lock is fatal because the yard hands over beyond the arm's reach;
and it never lifts block/ball/squat in the yard. Its env skill does not transfer: the
handover situations differ. Also open: only 5 of its 14 releases end in the bin.
Default brain unchanged (5e9df7 has no extra inputs).
