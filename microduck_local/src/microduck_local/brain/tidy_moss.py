"""The tidy brain for MOSS: drive to a can, scoop it, stow it in its own bin.

    search ──▶ approach ──▶ deploy ──▶ creep ──▶ close ──▶ lift ──▶ stow ──▶ release ──▶ tuck ─┐
      ▲        (tucked,     (jaws      (the      (grip)   (up     (over its   (open)    (fold) │
      │         arm inside   open at    base                0.25   own bin)                    │
      │         the shell)   can        drives                m)                               │
      └───────────────────── height)    the can ─────────────────────────────────────────────  ┘
                                        IN)

**A sibling of `brain/tidy_arm.py`, not a subclass** — the repo's own rule for
a whole brain whose vocabulary differs. Three things are structurally
different from MARS's, and each one REMOVES machinery rather than adding it:

* **The bin rides on the robot.** MARS drives to a basket, servos onto its
  rim and places over it; `tidy_arm` has `carry`, `deliver` and `place`
  states for that leg and reads the basket's rim height out of the scenario.
  MOSS carries its bin behind the arm, so the delivery target is a FIXED arm
  pose (`moss.DROP_POSE`) and the whole leg collapses into one swing. There
  is nothing to re-find, nothing to line up, and nothing to drop short of.
* **No run-time IK.** The grasp point is fixed in the robot's own frame, so
  the four poses are constants measured once (`robots/moss.py`) and what
  varies is where the BASE is — which this brain was steering anyway.
* **The base does the last centimetres, not the arm.** MEASURED, and it is
  the whole reason this brain is shaped the way it is: his palm sits ~60 mm
  up the approach axis and his can is 115 mm tall, so an arm descending onto
  a standing can lands on its lid and stops 35 mm short; and swinging the arm
  out of the tuck with a can in front sweeps it 20 cm away inside 0.4 s. So
  the jaws go to can height while the can is still 0.55 m off, and then the
  robot drives it in. A litter-picker's own order.

**What it is worth, MEASURED, and the two numbers disagree on purpose.**

* SCRIPTED, in MOSS's own scene with the can placed by hand
  (`scripts/probe_moss_grasp.py`'s sibling): **5/5 lifted, 5/5 stowed**.
* THIS BRAIN, in `moss-yard` off its own camera, 300 s, seeds 0/1/2:
  **2/9 cans stowed** (1, 1, 0), from 11 grasp attempts.

The gap is the brain's, not the arm's, and the failure is always the same:
the can is knocked aside or toppled out of line during the creep instead of
entering the jaws, so the machine closes on air and stows nothing. A 25 mm
belief error at the moment of closing (measured, below) against a grasp band
of +-30 mm is most of it. Worth quoting as 0.67 cans per 5 minutes, and worth
nobody's confidence until the creep is re-measured.

The can TOPPLES as it enters the jaws and is carried lying down; nothing here
grasps an upright can.

**Untested on hardware**, like everything else about this body.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass

import numpy as np

from pathlib import Path

from ..robots import moss
from ..robots import moss_env as ME
from .runtime import REGISTRY, Brain, Intent, Senses  # noqa: F401


#: The run whose `policy.onnx` is the shipped pickup skill.
#:
#: **A TIE BROKEN ON PROVENANCE, not a measured win** (2026-09-24). This was
#: `moss-pick-v1` — an RL leg whose chain ran through the lab but whose final
#: fine-tune was a CLI run. `teach-moss_pick-5e9df7` is the same leg
#: continued through `/teach` at rung 2 for 500k, so it is lab-produced end
#: to end, which is what this workspace asks of a trained leg.
#:
#: They cannot be told apart on the number that matters. Cans delivered over
#: 48 seeds of `moss-yard`, 144 cans an arm:
#:
#:     moss-pick-v1             46/144  (32%)
#:     teach-moss_pick-5e9df7   39/144  (27%)      0.93 se apart
#:
#: and the gap SHRANK as power went up — 1.6 se at 72 cans, 0.93 at 144 —
#: which is what noise does. At this n the smallest detectable difference is
#: about 15 points and the observed one is 5; separating them honestly needs
#: roughly 1300 cans an arm, which is not worth buying for a leg that is not
#: the bottleneck. On instant picks they are exactly tied, 45/48 each.
#:
#: The incumbent keeps the better point estimate, so this is reversible by
#: design: put `moss-pick-v1` back here and nothing else changes. What is NOT
#: a reason to prefer either is `5e9df7`'s better sustained-grip score (81%
#: against 62%) — that is the criterion it was trained against, and the same
#: criterion measured WORSE delivery under optimisation, so crediting it
#: would be crediting the proxy that failed.
#:
#: 2026-09-26: SWITCHED to `teach-moss_pick-ad9876`, the six-shape pick, once
#: the brain could run it as it trained (wrist-camera slots, base locked —
#: read off its run.json by `_flags`) and hand it objects where it trained
#: (per-object range, size-relative carry squeeze). moss-yard, 12 seeds x
#: 300 s, cans + block + squat + ball: ad9876 38/72, 5e9df7 37/72 (paired
#: +0.08 +- 0.42 per seed — a tie on the total); ad9876 delivers blocks 6/12
#: and balls 5/12 against 1/12 and 2/12, 5e9df7 squat cans 12/12 against 7/12.
#: Chosen for the shapes. Reversible: put "teach-moss_pick-5e9df7" back.
#: 2026-09-27: -> `teach-moss_pick-478dad`, ad9876 fine-tuned for objects
#: CLOSE IN (spawn 0.22-0.34 m) on the reward ad9876 actually trained under
#: (`gap_from_tcp` — unrecorded until now; three fine-tunes without it
#: learned to DRAG objects in). Deep picks in its env 81/78/90% at
#: 0.22-0.28/0.28-0.34/0.36-0.42 m against ad9876's 65/75/93%, card close in
#: 8/11 against 2/11, no dragging (-5.9 cm). moss-yard, 6 seeds x 300 s,
#: handover 0.41 m: 19/48 against 16/48. Reversible: "teach-moss_pick-ad9876".
SHIPPED_RUN = "teach-moss_pick-478dad"
#: One trained policy per LEG of the loop, each under `moss.CONTRACT_ID` —
#: the three envs share one 32-slot observation and one 8-action layout, so a
#: session is interchangeable and only the state that drives it differs. A leg
#: with no run on disk keeps its scripted behaviour, which is what every one
#: of them had before there was anything trained to replace it.
SHIPPED_RUNS: dict[str, str] = {
    "pick": SHIPPED_RUN,
    "approach": "moss-approach-v1",
    "stow": "moss-stow-v1",
    # The learned FOLD (2026-09-26): out of the bin and home, from wherever
    # this brain's tuck state really begins — after a delivery, a failed
    # stow or a failed pick (trained on 92 such poses from moss-yard seeds
    # 10-21). MEASURED in the yard, seeds 0-2, 22 tucks: home 19/22, arm-bin
    # contact 1.2%, the fold's own worst joint speed 1.56 rad/s — against the
    # scripted tuck's 21/22, 4.0% and 11.5-11.9 rad/s on EVERY tuck.
    "fold": "teach-moss_fold-81875e",
}
#: Per-leg overrides, for A/Bing a fresh run against the shipped one.
POLICY_ENV_VARS: dict[str, str] = {
    "pick": "MICRODUCK_MOSS_POLICY",
    "approach": "MICRODUCK_MOSS_APPROACH_POLICY",
    "stow": "MICRODUCK_MOSS_STOW_POLICY",
    "fold": "MICRODUCK_MOSS_FOLD_POLICY",
}
#: Where each leg's policy expects the arm to BE when it takes over — the
#: pose its env resets to. Getting this wrong is not a detail: the pickup
#: policy handed control in the wrong pose scored 0/3 in the room against
#: 4/12 in its own env, which is what `deploy`'s arrival test exists for.
POLICY_START_POSE: dict[str, tuple[float, ...]] = {
    "pick": moss.GRASP_POSE,
    "approach": moss.tuck_pose(),
    "stow": moss.GRASP_POSE,
}


@dataclass(frozen=True)
class TidyMossParams:
    #: Where the can must be, in the base frame, before the arm comes out.
    #: Far enough that the deploy sweep misses it (`moss.DEPLOY_STANDOFF_M`).
    deploy_at: float = moss.DEPLOY_STANDOFF_M
    #: Where it must be for the jaws to be around it.
    grasp_at: float = moss.GRASP_STANDOFF_M
    #: Approach and creep speeds, m/s. The creep is slow because it is the
    #: motion that puts a 18 g can between two pads.
    approach_mps: float = 0.30
    creep_mps: float = 0.08
    #: Yaw gain on the bearing, per state. The creep's is lower: a hard turn
    #: at 0.08 m/s sweeps the jaws sideways THROUGH the can.
    approach_kp: float = 2.0
    creep_kp: float = 1.2
    #: Turn rate while searching, rad/s.
    search_wz: float = 0.8
    #: How long each arm move is given before the next state, in seconds.
    #: The servo is his (kp 70 / kv 3), and these are what the scripted probe
    #: used; a shorter deploy drops the can, a shorter stow drops it outside.
    deploy_s: float = 2.0
    #: How close every arm joint has to be to `moss.GRASP_POSE` before the
    #: learned skill takes over, and how long to wait before giving up on it.
    deploy_tol_rad: float = 0.05
    deploy_timeout_s: float = 6.0
    #: Hand the can to the pickup policy only once it is inside the box that
    #: policy trained in (`moss.PICK_HANDOVER_BOX`), nudging the base until
    #: it is. Set False to hand over on arm arrival alone, which is what this
    #: did before 2026-09-24 and what put 72% of handovers out of band.
    band_handover: bool = True
    #: WITH THE BASE LOCKED, hand over nearer: close in until the object is
    #: at most this far (base frame), not merely inside the box's far edge
    #: (0.47). The arm reaches 0.457 m at floor height (measured, joint-space
    #: sweep), so a pick that cannot drive (ad9876 trained base-locked) was
    #: handed objects at the edge of its reach and nudged them along instead
    #: of grasping. 0.41 is the middle of the box it trained on (0.36-0.47).
    band_far_locked_m: float = 0.41
    #: SMALL THINGS, STRAIGHT DOWN (2026-09-27). The jaws can point straight
    #: down at floor height from 0.10 to 0.38 m (joint sweep; 9 deg at 0.40,
    #: 18 at 0.42), and the pick USES that close in: at 0.24-0.30 m it shuts
    #: on a card at 4 deg from vertical (7/7 picked), a block at 5 (12/12) —
    #: at 0.38-0.42 m everything is taken 20-27 deg off vertical, card 5/7.
    #: Big objects stay at `band_far_locked_m`: from a close handover a can or
    #: tall can was lost on the lift. An object smaller than
    #: `small_object_m` (detected size) is handed over at `band_far_small_m`.
    size_aware_handover: bool = True
    small_object_m: float = 0.08
    band_far_small_m: float = 0.28
    #: How long to spend nudging into the band before handing over anyway. A
    #: cap, not a target: a can the base cannot line up is still worth an
    #: out-of-band attempt, because the alternative is no attempt at all.
    band_timeout_s: float = 3.0
    #: Closing speed while lining up, m/s. Forward only — arriving NEAR
    #: converts at 73% against 38% inside the trained box, so there is
    #: nothing here worth backing away from.
    #: 0.12, not 0.06 (2026-09-27). A time budget of moss-yard runs put
    #: deploy at 26% of every 300 s, and the arm was not what took it: it
    #: reaches the grasp pose in 1.2 s; the base then crept into the band at
    #: 0.06 m/s and 7 deploys in 20 ran to the 9 s cap. 48 seeds x 300 s with
    #: 478dad: 225 v 190 in the bin, paired +0.73 +- 0.29 per seed (+0.96 /
    #: +0.50 per half). 0.18 is no better (-0.12 +- 0.30 v 0.12), and cutting
    #: the band short instead (band_timeout_s 1.0) is worse (77 v 94).
    band_mps: float = 0.12
    close_s: float = 1.0
    lift_s: float = 1.8
    stow_s: float = 3.4
    #: How long the LEARNED stow is given before the loop falls back to
    #: opening the jaws. Its env runs 20 s episodes and it delivers in a
    #: median of ~120 control steps, so this is generous on purpose.
    stow_policy_s: float = 20.0
    #: The three legs of the scripted route, in seconds.
    stow_high_s: float = 2.0
    stow_turn_s: float = 4.0
    stow_down_s: float = 3.0
    #: The stow releases when the arm has ARRIVED over the bin, not when its
    #: ramp's clock runs out — the same distinction `deploy` already makes,
    #: and for the same reason. MEASURED 2026-09-24: releasing on the clock
    #: opened the jaws with the carried can still at x = +0.151 m in the base
    #: frame, out in FRONT, when the bin's mouth is x = -0.087. The ramp
    #: commands the pose; the arm is still travelling to it.
    #: COMMAND THE JAW EVERY TICK OF THE CARRY, at the measured hold rather
    #: than at whatever the pickup policy last said. `WorldRobot.set_arm` is
    #: absolute — "a brain that wants two joints held says both every tick" —
    #: and `lift`/`stow` send only the five arm joints, so the gripper keeps
    #: the creep's final command for the whole carry. MEASURED 2026-09-24:
    #: that command is 0.0 m, a full squeeze, because the learned pickup
    #: closes hard to grip and the scripted `close` state (the only place
    #: `GRASP_JAW_CTRL_M` is ever commanded) is BYPASSED when a pick policy
    #: is loaded. Traced through a carry the jaws crush from 36.8 mm to
    #: 28.8 mm on a 66 mm can and it squirts out: 47 of 79 grip losses happen
    #: in `lift`, with the can still touching the floor.
    hold_jaw: bool = True
    #: Start the lift's ramp from where the arm IS, not from the pickup
    #: policy's last COMMAND. `_ramp`'s own docstring promises the former and
    #: the code did the latter; a servo under load lags, so the first tick of
    #: the lift was a step of that lag. MEASURED over 63 handovers: worst
    #: joint 0.062 rad median, but `shoulder_lift` reaches 0.38 rad at p95 and
    #: 0.60 rad at worst, and a step input is what the same docstring records
    #: as shearing the can out of the jaws.
    ramp_from_achieved: bool = True
    #: What to hold it at — `moss.GRASP_JAW_CTRL_M`, the gentlest setting the
    #: grasp sweep measured that still holds (6 mm of interference).
    carry_jaw_m: float = moss.GRASP_JAW_CTRL_M
    #: ...as INTERFERENCE, not a position: carry at the jaw's ACHIEVED
    #: position when the lift starts minus this. 27 mm is 6 mm of squeeze on
    #: a 66 mm can but WIDER than a 40 mm block or a 50 mm ball, so the lift
    #: opened the jaws on them — MEASURED in moss-yard (2026-09-26), blocks
    #: lifted 21 times and balls 16 and not one survived the lift.
    #: 10 mm, not 6 (2026-09-27): the carry's first swing (up to STOW_HIGH)
    #: rotates the wrist a long way and a close-in grip slid out there — at
    #: a 0.30 m handover 9 losses on that leg in 4 yard seeds, 4 at 10 mm, and
    #: objects in the bin 6 -> 13/32. At the default 0.41 m handover, 8 seeds:
    #: 33/64 against 28/64 (paired +0.62 +- 0.68 per seed — not resolved alone).
    carry_interference_m: float = 0.010
    carry_jaw_relative: bool = True
    #: ABANDON THE CARRY WHEN THE CAN IS GONE. The scripted stow never asked
    #: whether it was still holding anything: it ran all three ramps and
    #: opened the jaws over the bin regardless. MEASURED, only 31% of lifts
    #: still hold the can when the release fires, so about four stows in ten
    #: are nine seconds of theatre — in a 180 s run that is most of a minute
    #: spent posting nothing, while the can it dropped sits on the floor
    #: behind it. Watched in the lab, which is where this was spotted.
    #: STOP THE TRACKS ONCE THE ARM CAN REACH IT. The pickup policy drives
    #: the base flat out for the whole grab: measured over 29,165 creep ticks,
    #: it commands motion on 100% of them, |vx| saturated at the contract's
    #: 0.200 m/s and |wz| at a median 0.808 rad/s with 78% of ticks turning
    #: harder than 0.3 rad/s. That is not a policy converging on a can, it is
    #: a policy that was paid to close distance and never learned that the
    #: last 30 cm belong to the arm.
    #:
    #: It is also where the cans go: 158 of 434 can-touches in a run are the
    #: TRACK and the HULL, and a can knocked under the chassis is a can this
    #: loop then chases in circles. Spotted by a human watching the lab —
    #: "it gets stuck in the track and then it's spinning around trying to
    #: hit it... a lot of movement we don't really need from the tracks."
    #: RAMP THE TRANSITIONS, don't step them. `lift`, `stow` and `release`
    #: already ramp and are calm — max 2.00, 2.37 and 0.67 rad/s. `deploy`
    #: and `tuck` commanded their pose outright and are not: p95 9.47 and
    #: 7.94 rad/s, peaks of 12.9 and 13.5. An XL330's practical no-load speed
    #: is about 5.6 rad/s, so those peaks are not merely harsh, they are
    #: unreachable on the real arm — the sim only gets there because a
    #: position servo with no rate limit will slam. Spotted by a human
    #: watching the lab: "it's going really fast and kind of violent, I'm
    #: worried it might break the robot."
    deploy_ramp_s: float = 1.2
    #: The retract out of the bin, which is the one that can sweep delivered
    #: trash back out on its way past the rim.
    tuck_ramp_s: float = 1.5
    still_when_in_reach: bool = False
    #: Inside this range the base holds still and the ARM closes the gap.
    #: `moss.GRASP_STANDOFF_M` is 0.26 — where the grasp pose actually
    #: reaches — so this is that plus a little, not a guess at the reach.
    reach_stop_m: float = 0.30
    #: How much of the policy's commanded twist survives inside that range.
    #: MEASURED, and the answer is that ALL of it is needed by this policy:
    #:
    #:     scale   0.00   0.25   0.50   0.75   1.00
    #:     cans    9/36  14/36   8/36  15/36  22/36
    #:
    #: So the thrashing is not excess the deploy side can trim. The policy
    #: was trained with the base in its action vector and steers with it;
    #: taking it away puts it off its own manifold. `still_when_in_reach` is
    #: therefore OFF, and the fix belongs in the ENV
    #: (`moss_env.PICK_BASE_REACH_SCALE`), where a policy can learn that the
    #: last 30 cm are the arm's job instead of being told so at deploy time.
    reach_twist_scale: float = 1.0
    stow_abort_on_drop: bool = True
    #: Check the grip during LIFT as well. MEASURED (moss-yard, seeds 0-2):
    #: 8 of 10 drops happen at the very start of the carry — the grip never
    #: really held — and the check only ran in `stow`, so the brain noticed
    #: 2.8 s late. `lift_drop_grace_s` lets the jaw settle first.
    lift_abort_on_drop: bool = True
    #: LIFT ONLY ON A CONTACT THAT IS HAPPENING NOW. The creep->lift gate
    #: reads the DEBOUNCED grip (held within `grip_debounce_s`), so a contact
    #: that had just ended still read as held and the lift began with the
    #: pads touching nothing: MEASURED in moss-yard, 8 of 10 empty-handed
    #: carries had the pads on the object 0% of ticks from the first.
    lift_needs_live_contact: bool = True
    #: THE WRIST CAMERA SAYS WHETHER IT IS IN THE HAND, not just the pads.
    #: Both pads can touch an object still on the floor (pinched at an edge,
    #: pushed), and `Senses.holding` then says "held": MEASURED in moss-yard,
    #: a tall can carried 7.2 s with the pads on it 89% of ticks and in the
    #: hand 40%. The depth wrist camera gives the object's HEIGHT
    #: (`Senses.target_obs["z"]`). Calibrated on 6 yard seeds: in LIFT from
    #: 0.6 s a held object has risen (p5 -1.2 cm) and one left behind has not
    #: (median -14 cm); in STOW a held object never reads below 0.238 m.
    #: OFF: MEASURED three ways (6 yard seeds each) it cut short MORE real
    #: carries than it saved (21/30, 14/23 against 8/20 without) and put
    #: fewer objects in the bin (12/48 against 16/48) — first because the
    #: wrist target was a binned object, then with that fixed still.
    #: Kept, with `target_obs["z"]`, for a better-calibrated attempt.
    camera_hold_check: bool = False
    lift_rise_after_s: float = 0.6
    lift_rise_until_s: float = 0.9
    lift_min_rise_m: float = -0.02
    stow_min_z_m: float = 0.15
    lift_drop_grace_s: float = 0.3
    #: AFTER A DROP, straight home at this joint rate (rad/s), not the
    #: learned fold: the arm is not over the bin, so there is no bin to route
    #: around, and the fold's waypoint route took 7-12 s every time. 0.75 is
    #: the rate a command moves a joint at (0.03 rad per 25 Hz tick) — the
    #: same cap the fold obeys, so nothing about this is fast.
    drop_tuck_rate: float = 0.75
    drop_tuck_max_s: float = 6.0
    #: How long the grip has to read EMPTY before believing it. The pads lose
    #: and regain contact during the swing, so a single tick means nothing;
    #: `_gripped` is already debounced and this is on top of it.
    stow_drop_grace_s: float = 0.5
    stow_tol_rad: float = 0.12
    #: How long to keep holding the final pose while it settles before
    #: letting go anyway. A cap: a jam that never arrives still has to end,
    #: and dropping the can beside the bin beats carrying it forever.
    stow_settle_s: float = 3.0
    #: Let go from `moss_env.STOW_RELEASE_HIGH` (over the bin centre, 11 cm
    #: above the rim) instead of descending to `STOW_INSIDE`. ON, on the
    #: MECHANISM: at the end of each release in moss-yard (6 seeds), the
    #: descent left the can wedged in the open jaws 3 times in 12 — then the
    #: fold carried it off — and the high release 0 in 13. The end-of-run
    #: count moved 11 -> 13 (paired +1 0 -1 +1 +2 -1: inside the noise on its
    #: own); the fold still gets home (60/72 against 66/78) and over-speeds
    #: less (2 against 6). A first A/B that only counted objects at the end
    #: (14 v 13) could not see the wedge and was read as a null.
    release_high: bool = True
    #: How long a grip may READ as lost before the loop believes it. The
    #: pads break and remake contact constantly while the arm accelerates:
    #: traced in the room, `holding` flickered holds/released/holds/released
    #: inside 0.12 s right after a good grip, and read raw that ended the
    #: stow leg ONE TICK after it began (t=29.82 -> 29.84). The env has the
    #: same debounce on its own carry test for the same reason.
    grip_debounce_s: float = 0.25
    #: How long a grip has to HOLD before the pickup hands over. The loop
    #: used to leave on the first contact frame, and traced in the room that
    #: frame is routinely a flicker: `holds` at t=27.98, handed over at
    #: 28.00, `released` at 28.00. The scripted lift then raised an empty
    #: gripper and the stow policy — 9/12 into the bin in its own env — was
    #: handed nothing, 0 of 9 times. The pickup's OWN success test is
    #: stronger than one contact (held AND lifted clear), and this is the
    #: cheapest way to ask for the same thing from the room.
    grip_settle_s: float = 0.35
    #: How far apart the jaws must be for the grip to be a GRASP rather than
    #: a pinch. His can is 66 mm across, so jaws round it rest near 29 mm;
    #: MEASURED in the room, the pickup routinely ends with `holding` true at
    #: 7 mm, which is the jaws shut PAST the can with it wedged against the
    #: palm and the pads grazing it. That is a real contact and a useless
    #: grasp: handed on, the stow policy — which only ever trained on jaws at
    #: 27-29 mm — sees a finger position it has never seen and opens within
    #: 0.1 s, which is 0 of 9 cans in the room against 9/12 in its own env.
    min_grasp_m: float = 0.020
    #: How far a detection may sit from the can being tracked and still be
    #: believed to BE it. Wide enough for a fix that has drifted while the
    #: robot drove, narrow enough to reject a different can — the ones that
    #: caused this are 0.6 m and more away.
    retarget_gate_m: float = 0.30
    release_s: float = 1.8
    tuck_s: float = 2.4
    #: THE LEARNED FOLD's budget. It takes ~190 control ticks (7.6 s) in its
    #: env; past this it hands on to search wherever the arm is.
    fold_policy_s: float = 12.0
    #: THE COMMAND LEASH for the fold, rad: each arm joint's goal is kept
    #: within this of its MEASURED position (goal = present +- leash, which
    #: is also what the brain writes on the robot). Without it a jaw pinned
    #: by the dropped object stores the advancing command and releases it at
    #: 10-14 rad/s when it slips free — learned fold and scripted alike. The
    #: fold trained under exactly this value (`moss_env.CMD_LEASH_RAD`).
    fold_leash_rad: float = 0.08
    #: A detection older than this is not a fix any more.
    stale_s: float = 1.0
    #: Only cans IN FRONT are targets: a bearing past this is something the
    #: robot would have to turn for, and a "can" nearer than `min_x` is
    #: either under the chassis or the one already in the bin.
    #: What the detector models an upright can as: `Prop.radius()` is the
    #: largest half-extent, so for a 66 x 115 mm can it is 0.0575 m.
    can_radius_m: float = 0.0575
    #: Take range from the DETECTOR's per-object estimate (`range_est`), not
    #: from a can's radius. MEASURED in moss-yard (2026-09-26): with the can's
    #: radius, every non-can object read ~0.2 m FARTHER than it stood — block
    #: 0.52 m v 0.26 true, ball 0.53 v 0.30, squat 0.51 v 0.31, cans +-0.01 —
    #: so the robot drove 20 cm too close (the arm unfolded into the object:
    #: 16-30% of deploys timed out) and the pick was told to reach 20 cm past
    #: it (0 ball lifts, block never kept). `range_est` has been per-object
    #: since the arena passed each prop's own size to the detector; the can
    #: radius was the workaround for the class-constant bug that preceded it.
    range_from_detector: bool = True
    max_bearing: float = 1.05          # ~60 deg
    min_x: float = 0.30
    #: How many times a can may be attempted before it is written off, and
    #: how close two fixes have to be to count as the same can. A can against
    #: a wall cannot be approached — the robot would have to get behind it —
    #: and without this the machine retries it for the rest of the run.
    #: `brain/tidy_arm.py` carries the same pair for the same reason.
    max_retries: int = 3
    same_can_m: float = 0.22
    #: How long a written-off can stays written off. Not forever: the robot
    #: moves, and a can that was unreachable from one side may not be from
    #: another — which is the one advantage a mobile base has over an arm.
    give_up_s: float = 45.0
    #: Give up creeping after this long and go back to searching — a can that
    #: never arrives is a can that was knocked away, and standing still with
    #: the jaws open is the one state this machine could sit in forever.
    creep_timeout_s: float = 12.0
    #: How long the LEARNED skill gets before the machine takes back over and
    #: re-deploys. It is the trained episode length (`moss_env.EPISODE_S`),
    #: not a bigger number: past its own horizon the policy is extrapolating,
    #: and a fresh deploy puts it back at the state its episodes start from.
    #: Measured over a 12 s window the gripper action drifted OPEN (+0.129
    #: against +0.015 in the env at matched can distance) and the jaw
    #: oscillated 29 -> 34 -> 29 mm without ever committing.
    learned_window_s: float = 8.0
    #: HAND THE PICK A CLEAN START (2026-09-26). Diffing the pick's first
    #: observation in moss-yard against its env's showed the last-action slots
    #: carrying the previous leg's action (up to 680 sd off the env's zeros)
    #: and a stale pick command carried between attempts. Entering `creep`
    #: now zeroes the last action and re-seeds the command from `GRASP_POSE`,
    #: as the pick env's reset does — the same fix the stow and fold legs got.
    pick_clean_start: bool = True
    #: Run the pick at its trained 25 Hz, holding the command between control
    #: ticks (per world tick it integrates at twice its trained rate — arm
    #: and jaw at double speed). An earlier measurement (8 -> 4 of 18 cans,
    #: ad9876) could not separate the two. ON since 2026-09-27, found by
    #: diffing the yard's pick against the env's from one snapshot (the yard
    #: stepped every 0.02 s): 478dad in moss-yard, 24 seeds x 300 s, 99 v 71
    #: in the bin (paired +1.17 +- 0.36 per seed, +1.17 in each half), lift
    #: losses 39 -> 24, cans kept 20 -> 45.
    pick_at_control_hz: bool = True


def _shipped_policy() -> str | None:
    """The trained pickup this repo ships, if it is on disk.

    Found by PATH rather than by an environment variable, so the skill is
    live the moment the lab starts: a variable set in one shell does not
    survive `restart.sh` launching the backend in its own session, and a
    brain that silently falls back to the scripted pickup looks identical
    from the outside. `MICRODUCK_MOSS_POLICY` still overrides, for A/Bing a
    new run against this one.
    """
    here = Path(__file__).resolve().parents[3] / "runs" / SHIPPED_RUN / "policy.onnx"
    return str(here) if here.is_file() else None


def _shipped_policies() -> dict[str, str]:
    """Every leg's trained policy that is actually on disk, by task.

    Same PATH-not-variable rule as `_shipped_policy`, for the same reason: a
    variable exported in one shell does not survive `restart.sh` launching
    the backend in its own session, and a brain that quietly falls back to
    the scripted leg looks identical from the outside.
    """
    runs = Path(__file__).resolve().parents[3] / "runs"
    found: dict[str, str] = {}
    for task, run in SHIPPED_RUNS.items():
        override = os.environ.get(POLICY_ENV_VARS[task])
        cand = Path(override) if override else runs / run / "policy.onnx"
        if cand.is_file():
            found[task] = str(cand)
    return found


class TidyMoss:
    """MOSS's litter loop. `brain/runtime.Brain`'s protocol."""

    kind = "tidy_moss"

    def __init__(self, params: TidyMossParams | None = None,
                 policy: str | None = None, **_ignored):
        self.p = params or TidyMossParams()
        #: The LEARNED half. `policy` is a path to an ONNX exported by
        #: `export-walk` under `moss.CONTRACT_ID`; given one, the `creep` and
        #: `close` states hand control to it and this class keeps doing what
        #: a state machine is good at — searching, approaching, stowing,
        #: folding. That is the split this whole brain exists to make: the
        #: contact-rich 30 cm is learned, the mission logic is code.
        self._sess = None
        self._in = None
        paths = _shipped_policies()
        if policy:                      # an explicit path is the PICKUP skill
            paths["pick"] = policy
        self._policy_paths = paths
        self._sessions: dict[str, tuple[object, str]] = {}
        if paths:
            import onnxruntime as ort
            for task, path in paths.items():
                sess = ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"])
                self._sessions[task] = (sess, sess.get_inputs()[0].name)
        #: What each loaded leg TRAINED with, read off its own run.json: the
        #: extra observation slots it expects filled, and whether its base was
        #: locked. `teach-moss_pick-ad9876` trained with attitude, proximity,
        #: size AND a locked base; handed zeros and a live base it put 4
        #: objects in the bin over 6 yard seeds (2026-09-26).
        self._flags: dict[str, dict] = {}
        for task, path in paths.items():
            fl = dict(ME.obs_env_kwargs(path))
            fl["base_lock"] = bool(ME.eval_env_kwargs(path).get("base_lock"))
            msg = ME.wrist_mount_mismatch(path, f"the {task} leg ({path})")
            if msg:
                warnings.warn(msg)
            self._flags[task] = fl
        # `_sess`/`_in` stay the PICKUP session: the creep and close states
        # read them directly and every existing test names them.
        if "pick" in self._sessions:
            self._sess, self._in = self._sessions["pick"]
        #: THE LEARNED STOW IS OFF BY DEFAULT, and the reason is a number.
        #: It scores 9/12 into the bin in its own env and 0 of 9 cans in the
        #: room, against 2 of 9 for the scripted path it would replace — so
        #: wiring it in makes the mission loop WORSE, and shipping it because
        #: it looks finished would be shipping a regression.
        #:
        #: What is known about the gap, all measured: the observation now
        #: matches its env at the handover (last_action, target_base and
        #: target_seen all zero-diff, joint_pos within 0.03; only joint_vel
        #: still differs, 0.65 max, because the env reads `qvel` and a brain
        #: finite-differences). The physics match (identical can, identical
        #: pad priority and friction, the room carries the corrected bin and
        #: arm geometry). What does NOT match is the GRASP: the pickup leaves
        #: the jaws at 7 mm in the room — shut past a 66 mm can, wedged
        #: against the palm — where the stow env always begins at 27-29 mm.
        #: Six separate fixes narrowed that without moving the room number,
        #: which says the remaining cause is not yet understood rather than
        #: nearly fixed.
        #:
        #: Set MICRODUCK_MOSS_LEARNED_STOW=1 to use it anyway, which is what
        #: the next person debugging this will want.
        self._use_learned_stow = (
            "stow" in self._sessions
            and os.environ.get("MICRODUCK_MOSS_LEARNED_STOW") == "1")
        self._policy_path = paths.get("pick")
        self.reset()

    def reset(self) -> None:
        self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
        self._prev_arm: dict[str, float] | None = None
        self._prev_t = 0.0
        self._prev_yaw: float | None = None
        self._policy_cmd: dict[str, float] | None = None
        self._carry_from: dict[str, float] | None = None
        self._track_fix = None
        self._track_t = -1e9
        self._prev_grip_cmd = 0.041
        self._grip_seen_t = -1e9
        self._last_obs: np.ndarray | None = None
        self._last_policy_t = -1e9
        self._held_twist = (0.0, 0.0, 0.0)
        self._leg_from: dict[str, float] | None = None
        #: When the carry first read empty, for `stow_drop_grace_s`.
        self._empty_since: float | None = None
        #: Snapshot for `_ramp_from_here`, cleared on every transition.
        self._ramp_from: dict[str, float] | None = None
        self._leg_i: int = -1
        self._fold_seeded = False
        self._dropped = False
        self._drop_from: dict[str, float] | None = None
        self._carry_jaw: float | None = None
        self._fix_size: float | None = None
        self._lift_z0: float | None = None
        self._low_since: float | None = None
        self._grip_from_t = 1e9
        self._grip_ticks = 0
        self.state = "search"
        self._t0 = 0.0                      # when the current state began
        self._fix: tuple[float, float] | None = None   # last can fix, base frame
        self._fix_t = -1e9
        self._odom: tuple[float, float, float] | None = None
        self.picked = 0
        #: Cans that have been tried and not collected: world-frame (x, y, t)
        #: of the attempt. World frame, because the robot moves and the point
        #: of writing one off is that it is still there when we come back.
        self._written_off: list[tuple[float, float, float]] = []
        self._attempts = 0
        self._target_world: tuple[float, float] | None = None

    # ------------------------------------------------------------- sensing

    def _see(self, senses: Senses) -> tuple[float, float] | None:
        """The nearest toy, as (x, y) in the BASE frame.

        The detector reports a bearing and a range from the CAMERA, which sits
        `moss.CAMERA_POS` ahead of the base origin — his CAD number, so the
        offset is added here rather than absorbed into the standoff.
        """
        frame = senses.det
        if frame is None or not frame.detections:
            return None
        out = []
        for det in frame.detections:
            if det.cls != "toy":
                continue
            r = self._range(det)
            x = moss.CAMERA_POS[0] + r * math.cos(det.bearing)
            y = moss.CAMERA_POS[1] + r * math.sin(det.bearing)
            # Three filters, and every one of them is a bug this brain had.
            # A can already IN THE BIN is still a can to a detector, and it
            # sits behind the camera, so its bearing folds back to a small
            # or negative x and the machine went straight from `search` to
            # `close` on it, over and over (seen on the 90 s contact sheet:
            # one can picked, then 80 s of cycling on nothing).
            if abs(det.bearing) > self.p.max_bearing:
                continue                       # not in front
            if x < self.p.min_x:
                continue                       # under the robot, or behind
            if (moss.BIN_INTERIOR_X[0] - 0.05 < x < moss.BIN_INTERIOR_X[1] + 0.05
                    and moss.BIN_INTERIOR_Y[0] < y < moss.BIN_INTERIOR_Y[1]):
                continue                       # its own bin's contents
            if self._is_written_off(x, y):
                continue                       # tried, could not collect
            # its physical SIZE, from the same detection: the detector ranges
            # each object by its own size, so range x angular width is it
            size = 2.0 * r * math.tan(max(float(det.width), 1e-4) / 2.0)
            out.append((r, x, y, size))
        if not out:
            return None
        _r, x, y, size = min(out)
        self._fix_size = float(size)
        return (x, y)

    def _range(self, det) -> float:
        """Range from the APPARENT WIDTH and the can's own radius.

        NOT `Detection.range_est`, and this is the single most expensive bug
        in this brain's history: `range_est` is derived from a per-CLASS
        nominal radius, and a 66 x 115 mm drinks can is not the playroom
        block that class was sized on. MEASURED in the room — the brain shut
        its jaws believing the can was 0.259 m ahead while it truly stood at
        **0.491 m**, and did that seven times in 180 s while never touching
        a can (0/3 stowed).

        A sphere of radius R at range r subtends `2*atan(R/r)`, so the range
        is `R / tan(width/2)`. `can_radius_m` is what the DETECTOR models the
        prop as — `world/scenario.Prop.radius()` takes the largest half
        extent, which for an upright can is half its HEIGHT, not its own
        radius. On hardware this number is the detector's own calibration for
        the class it is reporting, which is why it is a parameter and not a
        constant taken from the MJCF.
        """
        if self.p.range_from_detector:
            return float(det.range_est)
        half = max(float(det.width) / 2.0, 1e-4)
        return self.p.can_radius_m / math.tan(half)

    def _world(self, base_xy) -> tuple[float, float] | None:
        """A base-frame point in world coordinates, using `odom`."""
        od = self._odom
        if od is None:
            return None
        c, s_ = math.cos(od[2]), math.sin(od[2])
        return (od[0] + base_xy[0] * c - base_xy[1] * s_,
                od[1] + base_xy[0] * s_ + base_xy[1] * c)

    def _is_written_off(self, x: float, y: float) -> bool:
        w = self._world((x, y))
        if w is None:
            return False
        return any(math.hypot(w[0] - wx, w[1] - wy) < self.p.same_can_m
                   for wx, wy, _t in self._written_off)

    def _give_up(self, t: float) -> None:
        """Write the current target off, so `_see` stops offering it."""
        if self._target_world is not None:
            self._written_off.append((*self._target_world, t))
        self._attempts = 0
        self._target_world = None
        self._fix = None

    def _expire(self, t: float) -> None:
        self._written_off = [w for w in self._written_off
                             if t - w[2] < self.p.give_up_s]

    def _carry_fix(self, senses: Senses) -> None:
        """Move the remembered fix with the robot.

        The can is INVISIBLE for the last few centimetres — it goes under the
        camera's near edge — so the creep runs on dead reckoning. `odom` is
        the body's own (x, y, yaw), and the fix is rotated and translated by
        whatever moved since the last tick.
        """
        od = senses.odom
        if od is None or self._fix is None:
            self._odom = od
            return
        if self._odom is not None:
            dx, dy = od[0] - self._odom[0], od[1] - self._odom[1]
            dyaw = od[2] - self._odom[2]
            c, s = math.cos(-self._odom[2]), math.sin(-self._odom[2])
            fx, fy = self._fix
            # into the previous base frame's world offset, then re-rotate
            bx, by = fx - (dx * c - dy * s), fy - (dx * s + dy * c)
            cy_, sy = math.cos(-dyaw), math.sin(-dyaw)
            self._fix = (bx * cy_ - by * sy, bx * sy + by * cy_)
        self._odom = od

    # ------------------------------------------------------------ the machine

    def _track(self, senses: Senses) -> None:
        """The fix the POLICY is shown — kept separately from the one the
        state machine steers by.

        `_see` filters for TARGET SELECTION: it drops anything behind, under
        the chassis, or already in the bin, so the machine cannot chase its
        own load. Those filters are wrong for tracking a can the robot has
        already committed to: they blank it at exactly grasp range, and the
        policy then reads `target_seen` 0 on 55% of creep steps against the
        2% it saw in training. So this keeps the nearest toy with no filter
        but the field of view.
        """
        frame = senses.det
        if frame is None:
            return
        toys = [d for d in frame.detections if d.cls == "toy"]
        if not toys:
            return
        # ONCE COMMITTED, STAY ON THE SAME CAN. Taking the nearest DETECTED
        # toy is right while hunting and wrong the moment the robot has
        # chosen one: the committed can drops under the camera's near edge at
        # almost exactly grasp range, and the nearest thing still visible is
        # then a DIFFERENT can, further away. MEASURED at a creep handover —
        # can1 truly at 0.504 m, can2 at 1.174, can0 at 1.502 — the brain
        # handed the pickup policy a target at 1.117 m, i.e. can2, twice the
        # furthest distance that policy ever trained on (its rung 2 tops out
        # at 0.55). It converted 3 of 9 attempts in the room against 12/12 in
        # its own env.
        #
        # KEPT ON A NULL. A/B'd over 6 seeds x 180 s: 8 cans of 18 into the
        # bin with this gate and 8 of 18 without it. So the retarget is a
        # real defect and demonstrably not what limits the loop — at this
        # sample size, which for 18 events is a weak instrument. It stays
        # because pointing a policy at an object it never trained on is
        # wrong whether or not this particular room punishes it, and because
        # the next person measuring the pickup should not have to rediscover
        # that the target can change under it mid-grasp.
        #
        # So while the loop has committed, the fix is the detection nearest
        # the one being tracked, and only if it is plausibly the same object.
        # Nothing within the gate means the can is out of sight, which the
        # policy is trained to handle — `target_seen` goes to 0 and it acts
        # on the last fix. Being blind is recoverable; being confidently
        # pointed at the wrong can is not.
        committed = self.state in ("deploy", "creep", "close")
        if committed and self._track_fix is not None:
            def gap(det):
                rr = self._range(det)
                px = moss.CAMERA_POS[0] + rr * math.cos(det.bearing)
                py = moss.CAMERA_POS[1] + rr * math.sin(det.bearing)
                return math.hypot(px - self._track_fix[0],
                                  py - self._track_fix[1])
            best = min(toys, key=gap)
            if gap(best) > self.p.retarget_gate_m:
                return                      # none of these is our can
        else:
            best = min(toys, key=lambda d: d.range_est)
        r = self._range(best)
        self._track_fix = np.array([
            moss.CAMERA_POS[0] + r * math.cos(best.bearing),
            moss.CAMERA_POS[1] + r * math.sin(best.bearing)])
        self._track_t = senses.t

    def _policy_obs(self, senses: Senses, fix,
                    task: str = "pick") -> np.ndarray | None:
        """`moss.CONTRACT_ID`'s 32 floats, built from what a BRAIN is handed.

        The env builds this from its own `MjData`; here it comes off
        `Senses`, which is the same information a code skill would have on
        the robot — achieved joint positions, odometry, the detector's fix.
        Joint velocities are finite-differenced because `Senses` carries no
        velocity channel, which is also what the Jetson would do.
        """
        arm = senses.arm
        if arm is None:
            return None
        dt = max(senses.t - self._prev_t, 1e-3)
        pos = np.array([float(arm.get(j, moss.ARM_HOME[j]))
                        for j in moss.JOINT_NAMES], np.float32)
        if self._prev_arm is None:
            vel = np.zeros_like(pos)
        else:
            prev = np.array([float(self._prev_arm.get(j, moss.ARM_HOME[j]))
                             for j in moss.JOINT_NAMES], np.float32)
            vel = (pos - prev) / dt
        wz = 0.0
        if senses.odom is not None and self._prev_yaw is not None:
            d = senses.odom[2] - self._prev_yaw
            wz = math.atan2(math.sin(d), math.cos(d)) / dt
        o = np.zeros(moss.OBS_DIM, np.float32)
        o[moss.OBS_JOINT_POS] = pos - np.asarray(moss.DEFAULT_POSE)
        o[moss.OBS_JOINT_VEL] = vel
        o[moss.OBS_LAST_ACTION] = self._act
        o[moss.OBS_BASE_TWIST] = (float(senses.speed or 0.0), wz)
        # NO FIX MEANS ZEROS, exactly as the env writes it ("if self._fix is
        # None: target = np.zeros(3)"). This used to fall back to the last
        # known position and publish it regardless, which is a train/deploy
        # mismatch for any leg whose env runs without a fix — and the STOW
        # leg is precisely that: the can is inside the jaws, occluded and far
        # nearer than the depth minimum, so its env publishes zeros for the
        # whole carry. Handed a real (x, y) there instead, a policy measured
        # at 9/12 into the bin in its own env delivered 0 of 9 in the room.
        tf = self._track_fix if self._track_fix is not None else fix
        if tf is None:
            o[moss.OBS_TARGET_BASE] = (0.0, 0.0, 0.0)
        else:
            o[moss.OBS_TARGET_BASE] = (tf[0], tf[1], 0.0575)
        # FRESHNESS, not a constant 1.0 — which is what this read for one
        # run. The can drops under the camera's near edge at almost exactly
        # grasp range, so `target_seen` going to 0 is not an edge case here,
        # it is the last second of every pickup. The env trains through it
        # (its own slot flips) and the policy learns to act blind; telling it
        # "I can see it" in the one phase it was trained to handle without
        # sight is a train/deploy mismatch in the only 30 cm that matter.
        o[moss.OBS_TARGET_SEEN] = (
            1.0 if (senses.t - self._track_t) < 0.6 else 0.0)
        # THE SLOTS THIS LEG TRAINED WITH, from the wrist/front cameras
        # (`Senses.target_obs`), exactly as its env fills them — and left at
        # zero for a leg that trained with them dead (a live value there
        # would clip against a normaliser whose var is 3e-10).
        fl = self._flags.get(task, {})
        tob = senses.target_obs or {}
        if fl.get("publish_attitude") and tob.get("att") is not None:
            o[moss.OBS_TARGET_AXIS] = tob["att"][:2]
            o[moss.OBS_TARGET_UPRIGHT] = tob["att"][2]
        if fl.get("publish_proximity"):
            o[moss.OBS_TARGET_BASE.start + 2] = float(tob.get("range") or 0.0)
        if (fl.get("publish_size") and o[moss.OBS_TARGET_SEEN] > 0
                and tob.get("top") is not None):
            o[moss.OBS_SPARE] = float(tob["top"])
        if fl.get("publish_grip"):
            # the wrist depth camera's grip fix, as `MossPickEnv._grip_obs`
            # publishes it (`moss.OBS_GRIP`); `read()` already applied the
            # freshness gate
            g = tob.get("grip")
            o[list(moss.OBS_GRIP)] = (
                (float(np.linalg.norm(g)), 1.0) if g is not None
                else (ME.GRIP_UNSEEN_M, 0.0))
        #: The last vector actually published to a policy. Kept because the
        #: only way to find a train/deploy gap is to diff this against what
        #: the env hands the same policy at its reset, slot by slot — every
        #: integration bug in this loop has been found that way and none of
        #: them by reasoning about it.
        self._last_obs = o
        return o

    def _band_error(self, fix: tuple[float, float] | None) -> float:
        """How far out of the pickup policy's reach is the can, to close in?

        Returns 0.0 when the handover should go ahead, and a positive metre
        count when the base should still close in. It never returns negative:
        backing off is not something this should do, and that is MEASURED
        rather than assumed.

        The first version of this drove the can to the middle of the box the
        policy trains in (`moss.PICK_HANDOVER_BOX`), on the argument that a
        policy is competent where it was trained. It doubled the in-band
        share, 28% -> 67%, and delivered one FEWER can. Tagging every
        handover with where it started and whether it reached a lift says
        why — the box is not the ranking:

            too NEAR  x < 0.36      8/11   73%
            IN BAND   0.36-0.55     5/13   38%
            too FAR   x > 0.55     10/18   56%
            too WIDE  |y| > 0.12    1/5    20%

        Nearer is better, not worse, so parking at the band's middle threw
        away the best cell it had. What actually loses attempts is the
        LATERAL offset: the policy's env spawns |y| <= 0.12 and the room
        hands it cans up to 0.297 m off to one side, and those convert at
        20%. So the gate closes in when the can is beyond the box and turns
        to centre it when it is off to the side, and otherwise gets out of
        the way. (11, 13, 18 and 5 attempts: the wide and near cells are
        thin, and it is the DIRECTION that is being acted on here.)
        """
        if fix is None:
            return 0.0
        lo, hi, ymax = moss.PICK_HANDOVER_BOX
        if self._flags.get("pick", {}).get("base_lock"):
            hi = min(hi, self.p.band_far_locked_m)
            if (self.p.size_aware_handover and self._fix_size is not None
                    and self._fix_size < self.p.small_object_m):
                hi = min(hi, self.p.band_far_small_m)
        x, y = float(fix[0]), float(fix[1])
        if abs(y) > ymax:
            # Turning is the caller's yaw term; keep a little forward creep
            # on so a can that is wide AND far still gets closed down.
            return max(x - hi, 0.02)
        return max(x - hi, 0.0)

    def _gripped_raw(self, senses: Senses) -> bool:
        """Is the can in the jaws RIGHT NOW? `Senses.holding`, undebounced.

        The world fills that from `WorldRobot.held_body`, which the driver
        answers by asking whether BOTH pads are touching the same foreign
        body (`robots/moss_drive.held_body`). That is the only test here that
        distinguishes a carry from a crush: a jaw squeezing a can flat
        against the floor stalls its servo exactly like one holding it, and
        reading the servo alone had this loop lift, stow and release nothing
        24 times in 300 s.

        The servo stall stays as a FALLBACK for a world that answers nothing
        — a bare env, or a body whose driver has no `held_body` — with the
        settle guard that keeps a closing transient from reading as a stall.
        """
        # ONLY the physical signal. There was a servo-stall fallback here
        # for one run and it fired constantly: the policy drives the gripper
        # command to 0, the jaw reaches 0 on air, and any transient reads as
        # a stall — 24 lift/stow/release cycles in 300 s, every one of them
        # carrying nothing. A brain runs in a WORLD, and a world always
        # answers this channel; a body whose driver cannot say, says no.
        return bool(senses.holding)

    def _gripped(self, senses: Senses) -> bool:
        """`_gripped_raw`, held for `grip_debounce_s` after the last yes.

        The pads break and remake contact constantly while the arm moves, so
        the raw answer stutters. Traced in the room it went holds/released/
        holds/released inside 0.12 s of a good grip, and the stow leg — which
        exits when the can is gone — ended one tick after it started.
        """
        if self._gripped_raw(senses):
            if (senses.t - self._grip_seen_t) > self.p.grip_debounce_s:
                self._grip_from_t = senses.t     # a fresh grip starts here
            self._grip_seen_t = senses.t
            return True
        return (senses.t - self._grip_seen_t) < self.p.grip_debounce_s

    def _grip_held_for(self, senses: Senses) -> bool:
        """Has the current grip lasted `grip_settle_s`? See that parameter."""
        return (senses.t - self._grip_from_t) >= self.p.grip_settle_s

    def _grip_is_a_grasp(self, senses: Senses) -> bool:
        """Are the jaws AROUND the can, or shut past it? See `min_grasp_m`."""
        arm = senses.arm or {}
        jaw = arm.get(moss.GRIPPER_JOINT)
        if jaw is None:
            return True                  # no channel to judge by; allow it
        return float(jaw) >= self.p.min_grasp_m

    def _policy_step(self, senses: Senses, fix, task: str = "pick"):
        """Run one leg's learned skill, AT ITS OWN CONTROL RATE."""
        # THE POLICY RUNS AT 25 Hz, the world ticks at 50. Every MOSS policy
        # is trained inside an env that decimates to `moss.CONTROL_HZ` and
        # applies one action per control period; called every world tick it
        # gets TWICE the actions per second, so its integrated commands move
        # at twice the rate it ever experienced — the arm at 1.5 rad/s
        # instead of 0.75, the jaw at 0.2 m/s instead of 0.1.
        #
        # MEASURED before this: the brain's observation dt was 0.0200 s
        # against a 0.0400 s control tick, its |joint_vel| ran to 4.96, and a
        # traced pickup sat at 0.28-0.36 m oscillating the jaw
        # 0 -> 35 -> 41 -> 0 mm without ever closing, then gave up. The
        # pickup converted about a third of its attempts in the room against
        # 12/12 in its own env, and this is the difference between the two.
        #
        # Between control ticks the last command is HELD, which is what the
        # env does during its decimation.
        # ...AND YET RUNNING IT AT 25 Hz MEASURED WORSE. Holding the command
        # between control ticks, the way the env does, took the room from 8
        # cans of 18 to 4. So the mismatch is real and correcting it alone is
        # not the fix — 18 events cannot separate 4 from 8 with any
        # confidence either, so read this as "tried, did not help" rather
        # than "harmful". THE PICK is now held to 25 Hz by its caller
        # (`pick_at_control_hz`, on since 2026-09-27: 99 v 71 in 24 yard
        # seeds with 478dad); this method itself still runs per call.
        return self._policy_tick(senses, fix, task)

    def _policy_tick(self, senses: Senses, fix, task: str = "pick"):
        """Run one leg's learned skill for one tick -> (twist, arm).

        The deltas below are the ENV's, not this file's opinion:
        `ARM_DELTA_RAD` 0.03, `JAW_DELTA_M` 0.004, `MAX_VX` 0.20,
        `MAX_WZ` 1.0. A policy is a function of the action scaling it was
        trained under, so these two have to agree or the same network does
        something else on the robot than it did in the env.
        """
        obs = self._policy_obs(senses, fix, task)
        if obs is None:
            return (0.0, 0.0, 0.0), None
        sess, in_name = self._sessions[task]
        a = sess.run(None, {in_name: obs.reshape(1, -1)})[0][0]
        self._act = np.clip(np.asarray(a, np.float32), -1.0, 1.0)
        if self._policy_cmd is None:
            self._policy_cmd = {**_pose(POLICY_START_POSE[task]),
                                moss.GRIPPER_JOINT: 0.041}
        cmd = self._policy_cmd
        for i, j in enumerate(moss.ARM_JOINTS):
            cmd[j] = float(cmd[j] + self._act[i] * 0.03)
        cmd[moss.GRIPPER_JOINT] = float(np.clip(
            cmd[moss.GRIPPER_JOINT] + self._act[5] * 0.004, 0.0, 0.041))
        twist = (float(self._act[6]) * 0.20, 0.0, float(self._act[7]) * 1.0)
        if self._flags.get(task, {}).get("base_lock"):
            # it trained with the base LOCKED: its base outputs were no-ops
            # and are unconstrained, so they must not drive the robot
            twist = (0.0, 0.0, 0.0)
        self._held_twist = twist
        return twist, dict(cmd)

    def _ramp_from_here(self, target, since: float, seconds: float,
                        senses: Senses) -> dict[str, float]:
        """Ramp to `target` from wherever the arm WAS when this state began.

        `_ramp` interpolates from `_carry_from`, which the pickup handover
        seeds; this is for the transitions that have no handover to inherit —
        `deploy` and `tuck` — and it snapshots the achieved pose on the first
        tick of the state rather than trusting a command.
        """
        if self._ramp_from is None:
            if senses.arm is None:
                return _pose(target)
            self._ramp_from = {j: float(senses.arm.get(j, v))
                               for j, v in zip(moss.ARM_JOINTS, target)}
        k = min(1.0, max(0.0, since / max(seconds, 1e-3)))
        return {j: (1.0 - k) * self._ramp_from[j] + k * v
                for j, v in zip(moss.ARM_JOINTS, target)}

    def _ramp(self, target, since: float, seconds: float) -> dict[str, float]:
        """Move the arm from where it IS to `target`, over `seconds`.

        The states after a grip used to command the endpoint outright, which
        is a step input to a position servo: MEASURED, the brain reported
        `holding=can1` at the grip and `holding=None` 1.9 s later, having
        dropped the can during the lift. It matters more with a LEARNED
        pickup than with the scripted one, because the policy finishes in
        whatever pose it likes and the jump from there is arbitrary.

        The gripper is deliberately NOT in the ramp: whatever closed on the
        can stays closed until `release` opens it.
        """
        if self._carry_from is None:
            self._carry_from = {j: float((self._policy_cmd or {}).get(j, v))
                                for j, v in zip(moss.ARM_JOINTS, moss.GRASP_POSE)}
        k = min(1.0, max(0.0, since / max(seconds, 1e-3)))
        return {j: (1.0 - k) * self._carry_from[j] + k * v
                for j, v in zip(moss.ARM_JOINTS, target)}

    def _abort_drop(self, t: float, where: str, direct: bool) -> Intent:
        """The jaws are empty mid-carry: count the attempt and go home for
        another go — STRAIGHT (see `drop_tuck_rate`) when the arm is not yet
        over the bin, by the fold when it is."""
        self._empty_since = None
        self._carry_from = None
        self._leg_from = None
        self._policy_cmd = None
        self._attempts += 1
        if self._attempts >= self.p.max_retries:
            self._give_up(t)
        self._to("tuck", t)
        self._dropped = bool(direct)
        self._drop_from = None
        return Intent(twist=(0.0, 0.0, 0.0), arm=None,
                      note=f"{where}: dropped it — going back for it")

    def _folded(self, senses: Senses) -> bool:
        """Is the arm home — within the fold's own tolerance of the tuck on
        every joint that shapes the arm (`moss_env.RETRACT_JOINTS`)?"""
        from ..robots.moss_env import RETRACT_JOINTS, RETRACT_TOL_RAD
        arm = senses.arm or {}
        tuck = dict(zip(moss.ARM_JOINTS, moss.tuck_pose()))
        return all(j in arm and abs(float(arm[j]) - tuck[j]) < RETRACT_TOL_RAD
                   for j in RETRACT_JOINTS)

    def _to(self, state: str, t: float) -> None:
        if (state == "creep" and self.state != "creep"
                and self.p.pick_clean_start):
            self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
            self._policy_cmd = None
            self._last_policy_t = -1e9
        if state not in ("lift", "stow", "release"):
            self._carry_jaw = None
        if state == "lift":
            self._lift_z0 = None
        self._low_since = None
        self.state, self._t0 = state, t
        self._fold_seeded = False
        # A new state ramps from where the arm IS now, not from the last
        # state's starting pose. Cleared here so `_ramp_from_here` snapshots
        # again on its first tick.
        self._ramp_from = None

    def step(self, senses: Senses) -> Intent:
        p, t = self.p, senses.t
        self._expire(t)
        self._carry_fix(senses)
        self._track(senses)
        seen = self._see(senses)
        if seen is not None:
            self._fix, self._fix_t = seen, t
        # STALE means stale. This read `1e9` for one run, so a fix never
        # expired and the machine ran the whole loop on a memory of a can it
        # had already binned.
        fix = self._fix if (t - self._fix_t) < self.p.stale_s else None
        since = t - self._t0
        arm: dict[str, float] | None = None
        twist = (0.0, 0.0, 0.0)
        note = self.state

        if self.state == "search":
            arm = self._ramp_from_here(moss.tuck_pose(), since,
                                       p.tuck_ramp_s, senses)
            if fix is not None and seen is not None:
                self._policy_cmd = None     # the next leg seeds its own pose
                self._to("approach", t)
            else:
                twist = (0.0, 0.0, p.search_wz)
                note = "search: turning for a can"

        elif self.state == "approach":
            if fix is None:
                self._to("search", t)
            else:
                x, y = fix
                if x <= p.deploy_at:
                    self._to("deploy", t)
                elif "approach" in self._sessions:
                    # THE LEARNED DRIVE. It holds the arm as well as the
                    # base, deliberately: the arm does not reach `TUCK_POSE`
                    # — it jams on the hull 0.74 rad short and rides 75 mm
                    # proud of the chassis — so folding it in is something
                    # the policy is paid for rather than a pose asserted here.
                    twist, arm = self._policy_step(senses, fix, "approach")
                    note = f"approach: can {x:.2f} m ahead, learned"
                else:
                    twist = (p.approach_mps, 0.0,
                             p.approach_kp * math.atan2(y, x))
                    note = f"approach: can {x:.2f} m ahead, arm in"

        elif self.state == "deploy":
            arm = {**self._ramp_from_here(moss.GRASP_POSE, since,
                                          p.deploy_ramp_s, senses),
                   moss.GRIPPER_JOINT: 0.041}
            note = "deploy: jaws open at can height"
            self._policy_cmd = None
            # NOT the last action, though the symmetry with the stow
            # handover says it should be: the stow leg gains from starting
            # with `o[OBS_LAST_ACTION]` at zero the way its env does, and
            # the pickup leg LOSES by it. MEASURED over the same three
            # seeds: clearing it here took the room from 4 cans in the bin
            # to 0. `deploy` holds for several ticks while the arm travels,
            # so zeroing here also zeroes it on every one of them, and the
            # pickup evidently reads its own recent history through that
            # slot in a way the stow policy does not. Left alone on the
            # evidence rather than on the argument.
            # Hand over when the arm has ARRIVED, not after a fixed wait.
            # MEASURED at the handover tick: on a 2 s timer the wrist was
            # still 0.708 rad from the grasp pose — most of the way back at
            # HOME — because the travel out of the TUCK is longer than the
            # travel the env settles from. The policy then opened every
            # episode in a pose it had never trained in, and the loop stowed
            # 0/3 while the same policy scored 4/12 in its own env.
            arrived = False
            if senses.arm is not None:
                arrived = max(
                    abs(float(senses.arm.get(j, 0.0)) - v)
                    for j, v in zip(moss.ARM_JOINTS, moss.GRASP_POSE)
                ) < p.deploy_tol_rad
            if arrived or since >= p.deploy_timeout_s:
                # ARRIVING IS NOT THE SAME AS BEING IN RANGE. This tested
                # the arm's pose alone and handed the policy whatever range
                # the can had drifted to while the arm travelled — 13 of 47
                # handovers (28%) inside the box it trains in, 38% too far,
                # 23% too near. The policy is 12/12 when its env starts it
                # in the box, so most of the room's failures were states it
                # had never been shown rather than a skill it lacked.
                band = self._band_error(fix) if p.band_handover else 0.0
                if band == 0.0 or since >= p.deploy_timeout_s + p.band_timeout_s:
                    self._to("creep", t)
                else:
                    # Close the loop on the ONE variable the next leg's
                    # competence depends on. Backing off is as necessary as
                    # closing in: a quarter of these are already too near.
                    x, y = fix if fix is not None else (0.0, 0.0)
                    twist = (p.band_mps, 0.0,
                             p.creep_kp * math.atan2(y, max(x, 1e-3)))
                    note = (f"deploy: lining up, can {x:.2f} m "
                            f"-> [{moss.PICK_HANDOVER_BOX[0]:.2f}, "
                            f"{moss.PICK_HANDOVER_BOX[1]:.2f}]")

        elif self.state == "creep" and self._sess is not None:
            # THE LEARNED SKILL drives here: base and arm together, until it
            # reports a grip or the window runs out. `fix` going stale is NOT
            # a failure here the way it is for the scripted creep — the can
            # drops under the camera's near edge at exactly this range, and
            # the policy was trained through that (its `target_seen` slot
            # goes to zero and it keeps acting). So the learned branch runs
            # on the last fix and only gives up on the clock.
            if self._fix is None or since > p.learned_window_s:
                self._attempts += 1
                if self._attempts >= p.max_retries:
                    self._give_up(t)
                self._to("tuck", t)
            elif (self._gripped(senses) and self._grip_held_for(senses)
                  and self._grip_is_a_grasp(senses)
                  and (not p.lift_needs_live_contact
                       or self._gripped_raw(senses))):
                self._carry_from = (
                    {j: float(senses.arm.get(j, 0.0)) for j in moss.ARM_JOINTS}
                    if (p.ramp_from_achieved and senses.arm is not None)
                    else dict(self._policy_cmd or {}))
                self._to("lift", t)
            else:
                if (p.pick_at_control_hz and t - self._last_policy_t
                        < 1.0 / moss.CONTROL_HZ - 1e-6
                        and self._policy_cmd is not None):
                    twist, arm = self._held_twist, dict(self._policy_cmd)
                else:
                    self._last_policy_t = t
                    twist, arm = self._policy_step(senses, self._fix)
                note = f"creep: learned pickup, can {self._fix[0]:.2f} m"
                # THE LAST 30 CM BELONG TO THE ARM. The policy keeps its arm
                # channels either way — only the tracks are silenced, and
                # only once the can is inside the reach the grasp pose was
                # measured at.
                if (p.still_when_in_reach
                        and float(self._fix[0]) <= p.reach_stop_m):
                    k = p.reach_twist_scale
                    twist = (twist[0] * k, twist[1] * k, twist[2] * k)
                    note = (f"creep: in reach at {self._fix[0]:.2f} m — "
                            f"tracks at {k:.0%}, arm closing")

        elif self.state == "creep":
            if fix is None or since > p.creep_timeout_s:
                # The can did not arrive: it was knocked aside, or it is
                # somewhere this base cannot drive to. Count the attempt.
                self._attempts += 1
                if self._attempts >= p.max_retries:
                    self._give_up(t)
                    note = "gave up on this one"
                self._to("tuck", t)
            else:
                x, y = fix
                self._target_world = self._world((x, y)) or self._target_world
                if x <= p.grasp_at:
                    self._to("close", t)
                else:
                    twist = (p.creep_mps, 0.0, p.creep_kp * math.atan2(y, x))
                    note = f"creep: driving the can in, {x:.2f} m"

        elif self.state == "close":
            arm = {moss.GRIPPER_JOINT: moss.GRASP_JAW_CTRL_M}
            note = "close: gripping"
            if since >= p.close_s:
                self._to("lift", t)

        elif self.state == "lift":
            arm = self._ramp(moss.LIFT_POSE, since, p.lift_s)
            note = "lift"
            z = (senses.target_obs or {}).get("z") if p.camera_hold_check else None
            if z is not None and self._lift_z0 is None:
                self._lift_z0 = float(z)
            if (z is not None and self._lift_z0 is not None
                    and p.lift_rise_after_s <= since <= p.lift_rise_until_s
                    and float(z) - self._lift_z0 < p.lift_min_rise_m):
                return self._abort_drop(t, "lift (camera: not rising)",
                                        direct=True)
            if p.lift_abort_on_drop and since > p.lift_drop_grace_s:
                if not self._gripped(senses):
                    self._empty_since = self._empty_since or t
                    if t - self._empty_since >= p.stow_drop_grace_s:
                        return self._abort_drop(t, "lift", direct=True)
                else:
                    self._empty_since = None
            if since >= p.lift_s:
                self._carry_from = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
                # Hand the learned stow the pose its env starts from: the
                # lift pose with the jaws shut on the can. `_policy_step`
                # only seeds `_policy_cmd` when it is None, and coming out of
                # the pickup it still holds whatever THAT policy ended on —
                # which is not where the stow policy was trained to begin.
                if self._use_learned_stow:
                    # KEEP THE PICKUP'S JAW COMMAND. Seeding this from
                    # `_prev_grip_cmd` handed the stow policy 41 mm — the
                    # initial wide-open value, never updated — while the
                    # fingers were physically at 8 mm, shut on the can. The
                    # policy inherits the command, so its first tick drove
                    # the jaws open and the can fell: measured, the stow leg
                    # lasted 0.28 s and the loop delivered 0 of 9 while the
                    # same policy scored 9/12 in its own env, whose reset
                    # hands it 27 mm. Only the ARM moves to the lift pose;
                    # whatever closed on the can stays closed.
                    # AND CLEAR THE LAST ACTION. `o[OBS_LAST_ACTION]` is
                    # eight of the thirty-two slots, and every stow episode
                    # its env ever ran began with them at ZERO. Handing the
                    # stow policy the PICKUP's last action — a saturated
                    # vector from a different task — is eight slots of
                    # out-of-distribution input on its very first tick, and
                    # it answers the way a policy does off its manifold:
                    # measured in the room, every channel saturated at +/-1
                    # and the jaws went 0 -> 20 -> 40 mm in 0.2 s, dropping
                    # the can before the leg had begun.
                    self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
                    held_jaw = (self._policy_cmd or {}).get(
                        moss.GRIPPER_JOINT, moss.GRASP_JAW_CTRL_M)
                    self._policy_cmd = {
                        **_pose(moss.LIFT_POSE),
                        moss.GRIPPER_JOINT: held_jaw}
                    # And FORGET the can. Its env resets the tracker at the
                    # start of every stow episode and the detector never
                    # re-acquires a can that is inside the jaws, so the
                    # policy learned the whole carry with those slots at
                    # zero. Carrying the approach's fix into the stow leg
                    # hands it an observation it has never seen.
                    self._track_fix = None
                    self._track_t = -1e9
                    self._fix = None
                self._to("stow", t)

        elif self.state == "stow" and self._use_learned_stow:
            # THE LEARNED STOW drives here, arm and jaws together, until it
            # lets go. The scripted version below ramps to `moss.DROP_POSE`,
            # which is a pose the arm CANNOT REACH while carrying: it stalls
            # 0.379 rad short against the bin's own front wall, and opening
            # the jaws from that jam flings the can clear of the bin. That
            # branch stays only for a lab with no stow policy on disk.
            twist, arm = self._policy_step(senses, fix, "stow")
            note = "stow: learned"
            if not self._gripped(senses):
                # It opened the jaws — the delivery is its own business now.
                self._carry_from = None
                self._policy_cmd = None
                self.picked += 1
                self._attempts = 0
                self._target_world = None
                self._fix = None
                self._to("tuck", t)
            elif since >= p.stow_policy_s:
                self._to("release", t)

        elif self.state == "stow":
            # UP, ROUND, DOWN — not a straight ramp to `moss.DROP_POSE`.
            # That pose cannot be reached while carrying: the arm stalls
            # 0.379 rad short against the bin's own front wall and opening
            # the jaws from that jam flings the can clear of the bin. The
            # rotation has to happen ABOVE the bin. Ramped along these three
            # waypoints and opened at the end, this delivers 6 of 6 in the
            # stow env, where the straight ramp delivers 0 of 10.
            drop_pose = ME.STOW_RELEASE_HIGH if p.release_high else ME.STOW_INSIDE
            legs = ((ME.STOW_HIGH, p.stow_high_s),
                    (ME.STOW_TURNED, p.stow_turn_s),
                    (drop_pose, p.stow_down_s))
            t_end = 0.0
            arm = None
            for pose, dur in legs:
                if since < t_end + dur:
                    if self._leg_from is None or self._leg_i != legs.index(
                            (pose, dur)):
                        self._leg_i = legs.index((pose, dur))
                        self._leg_from = dict(self._carry_from or {})
                    arm = {j: (1.0 - min(1.0, (since - t_end) / dur))
                           * self._leg_from.get(j, v) + min(
                               1.0, (since - t_end) / dur) * v
                           for j, v in zip(moss.ARM_JOINTS, pose)}
                    break
                t_end += dur
                self._carry_from = dict(zip(moss.ARM_JOINTS, pose))
                self._leg_from = None
            note = "stow: up, round, down (scripted)"
            # THE CAMERA'S OPINION: an object the wrist camera sees down near
            # the floor for `stow_drop_grace_s` is not in the hand, whatever
            # the pads report.
            zc = (senses.target_obs or {}).get("z") if p.camera_hold_check else None
            if zc is not None and float(zc) < p.stow_min_z_m and since > p.stow_drop_grace_s:
                self._low_since = self._low_since if self._low_since is not None else t
                if t - self._low_since >= 0.3:
                    self._low_since = None
                    return self._abort_drop(t, "stow (camera: on the floor)",
                                            direct=self._leg_i in (-1, 0))
            else:
                self._low_since = None
            # IS THERE STILL A CAN? Asked every tick of the carry, because
            # the alternative is finishing the delivery with empty jaws and
            # then driving off as though it had worked.
            if p.stow_abort_on_drop and since > p.stow_drop_grace_s:
                if not self._gripped(senses):
                    self._empty_since = getattr(self, "_empty_since", None) or t
                    if t - self._empty_since >= p.stow_drop_grace_s:
                        self._empty_since = None
                        self._carry_from = None
                        self._leg_from = None
                        self._policy_cmd = None
                        # Past the first leg the arm is turning over (or into)
                        # the bin: the straight path home runs through it —
                        # MEASURED, 305 of 504 ticks against the bin and a
                        # 5.8 rad/s whip. The learned fold routes around it.
                        return self._abort_drop(
                            t, "stow", direct=self._leg_i in (-1, 0))
                else:
                    self._empty_since = None
            if arm is None:
                # THE CLOCK IS NOT THE ARM. This released as soon as the
                # three ramps had run, which commands the pose but does not
                # wait for it: following the carried can showed it at
                # x = +0.151 +- 0.319 m when the jaws opened, still out in
                # front, with the bin's mouth behind at x = -0.087. Hold the
                # final pose and let go once the arm is actually there.
                arm = _pose(drop_pose)
                reached = False
                if senses.arm is not None:
                    reached = max(
                        abs(float(senses.arm.get(j, 0.0)) - v)
                        for j, v in zip(moss.ARM_JOINTS, drop_pose)
                    ) < p.stow_tol_rad
                note = "stow: holding over the bin"
                if reached or since >= t_end + p.stow_settle_s:
                    self._to("release", t)

        elif self.state == "release":
            arm = {moss.GRIPPER_JOINT: 0.041}
            note = "release: into the bin"
            if since >= p.release_s:
                self._carry_from = None
                self.picked += 1
                self._attempts = 0
                self._target_world = None
                self._fix = None
                self._to("tuck", t)

        elif self.state == "tuck" and self._dropped:
            # STRAIGHT HOME after a drop, at the commanded joint rate, from
            # wherever the arm IS (snapshotted on the first tick).
            if self._drop_from is None:
                arm_now = senses.arm or {}
                self._drop_from = {j: float(arm_now.get(j, v)) for j, v in
                                   zip(moss.ARM_JOINTS, moss.tuck_pose())}
            tuck = dict(zip(moss.ARM_JOINTS, moss.tuck_pose()))
            step = p.drop_tuck_rate * since
            arm = {j: (tuck[j] if abs(tuck[j] - q0) <= step
                       else q0 + math.copysign(step, tuck[j] - q0))
                   for j, q0 in self._drop_from.items()}
            # The same leash as the fold: a blocked joint must not store the
            # ramp's advance and release it as a whip.
            arm_now = senses.arm or {}
            for j in list(arm):
                if j in arm_now:
                    q = float(arm_now[j])
                    arm[j] = float(np.clip(arm[j], q - p.fold_leash_rad,
                                           q + p.fold_leash_rad))
            arm[moss.GRIPPER_JOINT] = 0.041
            note = "tuck: dropped it — straight home"
            if self._folded(senses) or since >= p.drop_tuck_max_s:
                self._dropped = False
                self._drop_from = None
                self._to("search", t)

        elif self.state == "tuck" and "fold" in self._sessions:
            # THE LEARNED FOLD. Seeded the way its env starts a fold: command
            # = the MEASURED arm pose (not a nominal start pose — it takes
            # over from wherever the delivery left the arm), jaws open, last
            # action zero, no target fix.
            if not self._fold_seeded:
                self._fold_seeded = True
                arm_now = senses.arm or {}
                self._policy_cmd = {
                    **{j: float(arm_now.get(j, moss.ARM_HOME[j]))
                       for j in moss.ARM_JOINTS},
                    moss.GRIPPER_JOINT: moss.MISSION_OPEN_M}
                self._act = np.zeros(moss.NUM_ACTIONS, np.float32)
                self._track_fix = None
                self._track_t = -1e9
                self._last_policy_t = -1e9
            # AT ITS OWN 25 Hz. Run every 50 Hz world tick, its integrated
            # command would move at 1.5 rad/s instead of the 0.75 it trained
            # at — the fast motion this leg exists to avoid.
            if t - self._last_policy_t >= 1.0 / moss.CONTROL_HZ - 1e-6:
                self._last_policy_t = t
                _tw, arm = self._policy_step(senses, fix, "fold")
                arm_now = senses.arm or {}
                for j in moss.ARM_JOINTS:
                    if j in arm_now:
                        q = float(arm_now[j])
                        arm[j] = float(np.clip(arm[j], q - p.fold_leash_rad,
                                               q + p.fold_leash_rad))
                self._policy_cmd.update(
                    {j: arm[j] for j in moss.ARM_JOINTS})
            arm = dict(self._policy_cmd)
            twist = (0.0, 0.0, 0.0)          # the fold does not drive
            note = "tuck: learned fold"
            if self._folded(senses) or since >= p.fold_policy_s:
                self._policy_cmd = None
                self._to("search", t)

        elif self.state == "tuck":
            # VIA THE WAYPOINT. Folding straight to the tuck pose jams the arm
            # on the bin — this state's own note has recorded that for months
            # ("0.74 rad short and riding 75 mm proud"), and it is not a
            # tuning failure: the bin sits between the two poses, so no
            # monotone path exists. `moss.RETRACT_WAYPOINT` is a planner's
            # answer that connects 14/14 real post-delivery poses to the tuck,
            # and routing through it folds the arm home on 14/24 against 0/24
            # direct.
            _half = max(p.tuck_s * 0.45, 0.6)
            _goal = (moss.RETRACT_WAYPOINT if since < _half
                     else moss.tuck_pose())
            arm = self._ramp_from_here(_goal, since, p.tuck_ramp_s, senses)
            # NOT "arm inside the shell", which is what this said and what
            # the contact sheets then reported: MEASURED, the arm jams on the
            # hull 0.74 rad short of `TUCK_POSE` and rides ~75 mm proud of
            # the chassis. A note is read as an observation, so it may not
            # claim a pose the robot never reaches.
            note = ("tuck: swinging clear of the bin" if since < _half
                    else "tuck: folding the arm home")
            if since >= p.tuck_s:
                self._to("search", t)

        # The PREVIOUS values update LAST, after the observation has used
        # them. Updating them at the top — which this did for one run — makes
        # `prev` equal to `now` by the time `_policy_obs` differences them,
        # so all seven joint velocities and the yaw rate were identically
        # zero in the brain while the env fed the policy real ones (std 0.11
        # to 0.28). Measured by diffing the two published observations slot
        # by slot; nothing in the code reads wrong until you do that.
        if senses.arm is not None:
            self._prev_arm = dict(senses.arm)
        if senses.odom is not None:
            self._prev_yaw = senses.odom[2]
        self._prev_t = t
        # THE CARRY HAS TO KEEP SAYING IT. `WorldRobot.set_arm` replaces the
        # whole mapping every tick, so a state that names only the arm hands
        # the gripper back to whatever the driver last stored — which coming
        # out of a learned creep is a full squeeze. A learned stow commands
        # the jaw itself and is left alone.
        if (p.hold_jaw and self.state in ("lift", "stow")
                and isinstance(arm, dict)
                and moss.GRIPPER_JOINT not in arm):
            jaw = p.carry_jaw_m
            if p.carry_jaw_relative:
                if self._carry_jaw is None and senses.arm is not None:
                    got = float(senses.arm.get(moss.GRIPPER_JOINT,
                                               p.carry_jaw_m))
                    self._carry_jaw = max(0.0, got - p.carry_interference_m)
                if self._carry_jaw is not None:
                    jaw = self._carry_jaw
            arm = {**arm, moss.GRIPPER_JOINT: jaw}
        return Intent(twist=twist, arm=arm, note=note)

    def inputs(self) -> dict:
        """What the inspector shows: the fix this brain is acting on."""
        return {
            "can": {"value": list(self._fix) if self._fix else None,
                    "age": None if self._fix is None else 0.0},
            "picked": {"value": self.picked, "age": 0.0},
            "written_off": {"value": len(self._written_off), "age": 0.0},
            "attempts": {"value": self._attempts, "age": 0.0},
        }


def _pose(q) -> dict[str, float]:
    """A joint-name mapping from one of `robots/moss.py`'s measured poses."""
    return dict(zip(moss.ARM_JOINTS, q))


REGISTRY.register("tidy_moss", TidyMoss)
