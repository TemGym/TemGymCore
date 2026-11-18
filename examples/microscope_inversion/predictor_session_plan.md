# Predictor session plan (SightX only)

Goal: one session of SightX frames that fits a non-dimensional TemGym NEOARM which predicts the beam on the camera
for MAG 8k–500k, DIFF 8–200 cm and spots 0–4, and can be inverted to steer the beam.
Built from `console_tables.json` (read 2026-11-17).

## What the console tables told us

| | Rows | Lenses | Console presets |
|---|---|---|---|
| **Branch A**: MAG 8k–25k | 8k, 10k, 12k, 15k, 20k, 25k | PL1 moves 58059 → 65535, IL3 ~44–49k | IS1 31744/32024, IS2 33280/33536, PLA 33968/33200, OLs 31974/28384 |
| **Branch B**: MAG 30k–800k | 30k … 500k (… 800k) | PL1 fixed 64000, IL3 falls 39733 → 12904 at 500k | IS1 32768/32768, IS2 32988/36240, PLA 32768/32768, OLs 31990/26784 |
| **Branch C**: DIFF 8–200 cm | 15 rows | IL1 ~20500 (nearly fixed), IL2/IL3 do the work, PL1 64000 | As B, except PLA 32720/32448 |

- OLf is 39269 in all three branches: the objective is static. CM is 58624 at alpha 0.
- Not used: MAG 25k (PL1 at its maximum, 65535), ≥1M (OLf changes), DIFF 4–80 m (objective off).
- Spots 0–4: CL1 = 26112, 28672, 31232, 33792, 36320 (steps of 2560, the last 2528). The console also changes CL3 (to keep the brightness) and SpotA.
- Alpha 1 changes CM (26624) and CLA1/CLA2. **Alpha stays 0 all session.**
- OLs and ILs are stigmators: they change the beam's shape, not its centre. Recorded, not modelled at first.

## Probe corrector (between CLA2 and CM)

- In the model: its round optics only, one static block (start: a transfer doublet turning the beam over, as in
  `pyjem_simulator/condenser.py`). The hexapoles only add aberrations, so they are left out.
- With CM and the objective also static, everything from CLA2 to the specimen is one fixed transfer: only that
  combination is fitted, not the corrector's insides.
- Its internal alignment deflector is ignored: it is on a different API (the corrector software shows it in mA), not
  PyJEM. **Do not touch the corrector during the session.** Write down its deflector mA at the start and the end, as a
  check that it did not move.
- If it has to be realigned mid-session: take a 40k check before and after, and give the blocks after it a new
  incoming beam (a change at one deflector looks like a new specimen shift and tilt).
- Later, to predict its deflector: kick it ±x, ±y at 40k (shift) and DIFF 40 cm (tilt), like the IS and PLA kicks.

## Model inputs (read back with every frame)

CL1, CL3, CM, OLf, IL1, IL2, IL3, PL1, SpotA, CLA1, CLA2, IS1, IS2, PLA (and OLs, ILs, CLs for the record),
spot, alpha, mode, row, HT, stage.

## Blocks

**Every block** (about 20 frames), at one row, reached by the approach rule:

1. Centre the beam on the SightX; 2 reference frames.
2. CLA1 ±x, ±y; CLA2 ±x, ±y (8 frames).
3. IL1, IL2, IL3, PL1 at −w and +w, both reached from below (8 frames). w = 1000, except:
   - IL3 at 300k–500k: w = 500 (IL3 is only 12904–19368 there);
   - IL2 at DIFF 200 cm: −w only (IL2 is 65520).
4. One ruler frame: the grating in MAG, the gold rings in DIFF.

**Branch kicks**, at 2 rows per branch (12 extra frames each): IS1, IS2, PLA ±x, ±y by 1000, back to the preset after each.

| Branch | Fitted blocks | Branch kicks at | Kept back (predicted, then measured) |
|---|---|---|---|
| A | 8k, 12k, 20k | 8k, 20k | 10k, 15k |
| B | 30k, 40k, 80k, 150k, 300k, 500k | 40k, 300k | 60k, 200k, 400k |
| C | 8, 15, 40, 80, 150 cm | 40 cm, 15 cm | 25 cm, 100 cm |

**Condenser steps** (inside the blocks named; the console sets CL3 and SpotA, we record them):

| Where | What | Why there |
|---|---|---|
| 300k and 500k | spots 0–4; CL3 through crossover | small beam is easy to resolve |
| 40k | spots 0–4; a few CL3 steps | middle of range, ties to the checks |
| 8k | CL3 spread | large beam fits on the camera |
| DIFF 40 cm, 15 cm | spots 0–4 | disk size gives angles |

**40k check** (2 frames, CLA kicks off): same setting every time; catches drift and hysteresis between groups.

## Session order

| Step | What | Frames |
|---|---|---|
| 0 | Console tables: **done** (`console_tables.json`) | – |
| 1 | JEOL lens relax, once | – |
| 2 | Find the beam at 40k, spot 4, alpha 0; drift test: 40k check 3 times, 5 min apart | 6 |
| 3 | Branch B blocks, low to high magnification; 40k check after 80k and after 500k | ~170 |
| 4 | 40k check → branch A blocks → 40k check | ~90 |
| 5 | Branch C (DIFF) blocks → 40k check | ~130 |
| 6 | Kept-back rows: predictions saved first, then reference + CLA kicks measured | ~70 |
| 7 | Final 40k check; read back everything | 2 |

About 470 frames, about 3 hours at ~20 s per frame.

## Fit order (after the session)

1. Projection lenses and deflectors (IL1–PL1, IS1, IS2, PLA) with SightX rotation and pixel/L, from the reference frames,
   lens wobbles and branch kicks across all rows.
2. CLA1/CLA2 and the incoming beam per block.
3. Condenser and source (CL1, CL3, CM, SpotA, spot steps).
4. A family of fits; judged by the kept-back rows.
