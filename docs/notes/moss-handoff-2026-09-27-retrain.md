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
