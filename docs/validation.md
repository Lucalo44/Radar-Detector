# Validation

To make sure the physics is behaving, the solver is checked against closed-form answers. To run every test:

```bash
python -m pytest tests
```

All of them should pass in about 5 seconds.

## Against closed-form answers

These seven are in [tests/test_rcs.py](../tests/test_rcs.py). All of them use a radar 2km away, so the target sits in the far field where the textbook formulas apply.

1. **Triangle phase integral.** The exact integral over 20 random triangles, compared against brute-force midpoint quadrature on a 600×600 grid. Tolerance: 0.2% of the triangle's area.
2. **Flat plate at normal incidence.** A 10cm metal plate facing the radar, against σ = 4π·a⁴/λ². Tolerance: 1%.
3. **Reflectance scales power.** The same plate with R = 0.25 has to come out at exactly a quarter of the metal plate. Tolerance: 1%.
4. **Sphere.** A 10cm-radius metal sphere, against the optical-region value σ = π·r². Tolerance: 35%, since a faceted sphere is a cluster of small flat mirrors and its return depends on how they line up.
5. **Trihedral corner reflector.** A 10cm corner with multi-bounce on, against σ = 4π·a⁴/(3λ²), which is +14.4dBsm. Tolerance: 1.5dB. The multi-bounce part also has to be more than 10× the single-bounce part, which checks that the right mechanism is doing the work.
6. **Absorber.** A plate of radar absorber (R = 0) has to return under 10⁻¹² m².
7. **Shadowing.** A metal plate behind a larger absorber plate has to return under 10⁻⁹ m², which checks that the shadow rays block it.

## Against the previous solver

When I split the solver into its frequency-independent and per-frequency stages, it had to give the same answers as before. [tests/legacy_rcs.py](../tests/legacy_rcs.py) is a frozen copy of the solver from commit 69677f2, and [tests/test_solver_split.py](../tests/test_solver_split.py) checks the new one against it:

1. The plate, sphere and trihedral cases, plus the full baseline biker from behind, at 60° and head-on. Every total, breakdown, energy share, per-face value and note has to match to a relative tolerance of 10⁻¹².
2. The cheaper shadow test has to give the same lit faces, with the same transmission, as the full multi-hit version, on the baseline biker from three directions.

The per-object values are the one thing that changed on purpose (see [How the solver works](solver.md)). Two tests cover that: a lone corner reflector's per-object value has to equal the scene total, and with a corner behind the saddle, the corner has to top the table.

## Band averaging

1. A flat plate's RCS barely changes over 1GHz, so its band average has to sit within 0.1dB of the single-frequency value.
2. Two small plates 20cm apart in depth beat against each other across the band (by more than 3× between frequencies). Their band average has to land between the single-frequency extremes.
3. With the bandwidth set to 0, the band solve has to equal a plain single-frequency solve exactly.

## Retroreflective materials

These are in [tests/test_retro.py](../tests/test_retro.py):

1. An ideal retro square (R = 1, q = 1) facing the radar has to match a metal plate of the same size, σ = 4π·A²/λ². Tolerance: 1%.
2. Turned 15°, 30° and 45° in bearing, it has to give σ = 4π·(A·cosθ)²/λ² within 0.1dB, while a metal plate of the same size has to drop more than 20dB below that.
3. Tilted 25° in elevation, it has to stay coherent across its faces, within 0.1dB of the same formula.
4. Past its coverage angle plus the 10° roll-off, and from behind, it has to return nothing.
5. The fitted metalens preset on a 53mm square has to reproduce the paper's measurements: 3.54dBsm at broadside (±0.3dB) and a −16.76dBsm median over ±40° in 1° steps (±1dB).
6. Two separate tags have to be found as two separate apertures.
7. An ideal tag behind the saddle has to more than triple the biker's return from 20° off the rear, and add nothing from the front.

## The rest

1. [tests/test_compare.py](../tests/test_compare.py) checks the change vs. biker alone, the ratio wording, the equivalent-sphere diameter and that the cached biker-alone result is thrown out when a material changes.
2. [tests/test_docs.py](../tests/test_docs.py) checks that every page of these docs loads, starts with a heading and only links to files that exist.
3. [tests/test_project.py](../tests/test_project.py) checks that a project saves and reopens with every setting, material and face assignment intact.
4. The ray-engine check has to report a name without raising, whether or not Embree is installed.
