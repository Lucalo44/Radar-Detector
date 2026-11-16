# Materials

Each material is described by three power fractions at normal incidence, plus how its reflected power is split:

1. **Reflectance R**: the fraction of incident power that's reflected.
2. **Absorptance A**: the fraction that's absorbed.
3. **Transmittance T = 1 − R − A**: the fraction that passes through. It's worked out, not set.
4. **Roughness s**: the fraction of the *reflected* power that scatters diffusely instead of like a mirror.
5. **Retro share q**: the fraction of the *reflected* power sent straight back toward the radar, as long as the radar is within ±the coverage half-angle of the face normal. Whatever is left, 1 − s − q, reflects like a mirror. Most materials have q = 0.

## Presets

These are the 17 presets in [radar_detector/materials.py](../radar_detector/materials.py). Every value is an engineering estimate at 77GHz (λ ≈ 3.9mm), derived from typical complex permittivities. For example, skin at ε ≈ 6.5 − j8 gives a power reflection of about 0.32. They're a reasonable starting point, not lab data, which is why the app says "estimate" under the material picker.

| Material | R | A | T | Roughness s | Retro q | Note |
|---|---|---|---|---|---|---|
| Aluminum | 0.99 | 0.01 | 0.00 | 0.02 | 0 | Bike frames, rims, brackets. Near-perfect conductor. |
| Steel | 0.98 | 0.02 | 0.00 | 0.03 | 0 | Spokes, chain, hardware. |
| Carbon fiber composite | 0.85 | 0.15 | 0.00 | 0.05 | 0 | Conductive fibres reflect most of the energy. |
| Clothed body | 0.33 | 0.67 | 0.00 | 0.50 | 0 | Skin under thin, mostly transparent jersey fabric. |
| Skin | 0.35 | 0.65 | 0.00 | 0.30 | 0 | High water content; lossy dielectric. |
| Tire rubber | 0.25 | 0.70 | 0.05 | 0.30 | 0 | Carbon-black loaded rubber. |
| Saddle foam / leather | 0.08 | 0.30 | 0.62 | 0.30 | 0 | |
| ABS / hard plastic | 0.05 | 0.03 | 0.92 | 0.05 | 0 | Low-loss dielectric; mostly transparent. |
| Polycarbonate | 0.06 | 0.02 | 0.92 | 0.02 | 0 | Lens and light housings. |
| Nylon fabric | 0.02 | 0.05 | 0.93 | 0.30 | 0 | Bags and panniers (dry). |
| Metallized fabric | 0.90 | 0.08 | 0.02 | 0.30 | 0 | Conductive textile. |
| Wood (dry) | 0.04 | 0.30 | 0.66 | 0.10 | 0 | |
| Glass | 0.18 | 0.20 | 0.62 | 0.02 | 0 | |
| Water / wet surface | 0.60 | 0.40 | 0.00 | 0.10 | 0 | |
| Radar absorber (RAM) | 0.01 | 0.99 | 0.00 | 0.00 | 0 | Carbon-loaded absorber foam. |
| Retroreflector (ideal) | 0.99 | 0.01 | 0.00 | 0.00 | 1 (±60°) | Upper bound: every reflected watt goes back to the radar within ±60°. |
| Metalens retro tag (measured) | 0.28 | 0.72 | 0.00 | 0.00 | 0.013 (±40°) | Fitted to Ghasemi et al. 2026: 3.54 dBsm broadside, −16.8 dBsm median over ±40° for 53 mm. |

Of note, most plastics are close to invisible at 77GHz. An ABS shell passes 92% of the power, which is why radar sensors can sit behind plastic bumpers. If you want a plastic part to show up, it needs metal in or on it: a foil layer, a metallized coating or a metal insert.

## Retroreflective materials

The last two presets stand in for retrodirective tags: parts built to send radar straight back where it came from, over a wide range of angles. Examples are metalens tags, Van Atta arrays and Luneburg lenses. They're modelled by what they do, not by what's inside them, so any face painted with one acts as a phase-conjugating surface (see [How the solver works](solver.md)).

1. **Retroreflector (ideal)** is the upper bound. Every reflected watt goes back to the radar out to ±60°, then the return rolls off over the next 10°. A flat 53mm square of it returns +8.2dBsm head-on, as much as a metal plate of the same size, and still +6.9dBsm at 30°, where the metal plate has dropped by 34dB to −26.3dBsm.
2. **Metalens retro tag (measured)** is fitted to the 53×53mm tag in Ghasemi, Hester and Eid, "A metalens-based bicycle safety reflector for autonomous vehicle radars", *Scientific Reports* (2026), [doi:10.1038/s41598-026-70859-x](https://doi.org/10.1038/s41598-026-70859-x). I matched two of their measured numbers on a flat 53mm square: 3.54dBsm at broadside and a median of −16.76dBsm over ±40° in 1° steps. The fit gives R = 0.28, q = 0.013 and ±40° coverage. Most of the broadside peak is the tag's metal ground plane reflecting like a mirror; the retro part is what holds up off-axis.

Of note, the measured tag returns far less than the ideal one. Its median over ±40° is about 25dB below a metal plate of the same size facing the radar. The value of the tag is that its return stays roughly level across 80°, where a plate collapses a few degrees off-axis.

Two things to keep in mind with these:

1. Retro faces are one-sided. They only work from the side their face normal points to, which for the samples is backwards, toward a radar behind the rider. The back of the tag reflects like an ordinary surface.
2. The paper reports its gains per range-azimuth cell of a radar with 3.6cm range resolution, so its radar separates the tag from the rider. This tool reports the total RCS of the whole scene. A small tag next to a whole rider shows a smaller change here than in the paper, but the per-object table shows the tag's own return.

To make your own, switch the Face materials box to Custom and set the "Retroreflective share" and its coverage angle. The retro share plus the roughness can't exceed 1.

## Assigning materials

In Simple mode, the material drop-down under "Selected object" sets one material for the whole object. Imported parts start as ABS. The metal samples start as aluminum, and the two retro tags start as their own retro presets.

In Advanced mode, the "Face materials" box lets you pick surfaces in the 3D view and assign materials to just those faces:

1. Click a surface to select it. The selection grows across neighbouring faces until it hits an edge sharper than the "Surface edge angle" (20° by default).
2. Shift+click adds or removes a surface. Option/Alt+click toggles a single triangle.
3. Pick a preset and press "Assign to selected faces", or "Assign to whole object".

## Custom materials

Switch the material source to "Custom" to set your own R, A and roughness. T updates as you type, and A is capped so R + A never exceeds 1.

Assigning a custom material saves it as a new entry in the library. It never changes a preset, so the baseline biker always means the same thing. The new material then shows up in every material list, and it's saved with the project. Assigning the same name, colour and values again reuses that entry instead of piling up copies. A different material with a name already in use gets a number, like "Foil (2)".

Changing any material's values, or any of the biker's face materials, re-solves the biker alone too. The comparison is always against the biker with the current library.
