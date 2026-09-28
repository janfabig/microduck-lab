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
