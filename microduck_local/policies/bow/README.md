# bow — the greeting skill (CPU, this harness)

`bow.onnx` is the `bow` behavior (`behaviors/poses.py`) trained in this
harness: 61-obs / 14-action, normalizer baked in, deterministic mean. The
world runs it as a SKILL — `World.start_skill("bow")` swaps it in for
`BOW_S` seconds under an all-zero command (`world/arena.py`), the way the
kicks run — and `brain/greet.py` is the brain that asks for it.

Lineage: `teach-bow-25bdeb` (from scratch, 1.0M steps), then
`teach-bow-ad9e2d`, a 2.0M-step fine-tune warm-started from it
(`behavior.json` here is that run's recipe). Measured on the trunk-pitch
probe: the bow reaches ~17 deg and holds there; see `docs/roadmap.md` for
the open question of why it stops short. A prototype policy, not a robot
one — "sim2real honesty" in `../../AGENTS.md`.
