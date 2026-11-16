# What the numbers mean

The question I built this tool to answer is simple: compared to a plain cyclist, does a part make the rider easier or harder for a car's radar to see, and by how much? The Results tab is laid out in that order. This page explains each number on it.

## Change vs. biker alone

This is the headline. It's the total radar cross section (RCS) of the scene, divided by the RCS of the baseline biker with no attachments, in dB:

Δ = 10·log₁₀(σ_with / σ_biker)

Both solves use the same radar settings and the same aim point (the centre of the biker), so the only difference between them is your parts. The colour tells you which way it went:

1. **Green** means more visible, by more than 1dB.
2. **Grey** means within ±1dB. That's inside the solver's own speckle, so I treat it as "no real change".
3. **Amber** means less visible, by more than 1dB. Of note, a part can do this by shadowing a reflective bit of the bike or rider.

Under it, the same change is written as a ratio of returned power ("about 2.2× the return"). Every 3dB is roughly a doubling, and every 10dB is a factor of ten.

The biker-alone result is solved once and cached. It's only solved again when something it depends on changes: the radar settings, the biker's face materials, a material's values, or the aim point.

## Total radar cross section (dBsm and m²)

RCS σ is the area of a perfect isotropic reflector that would send the same power back to the radar. It's measured in m², or in dB relative to 1m² (dBsm):

σ_dBsm = 10·log₁₀(σ / 1m²)

So 0dBsm is 1m², +10dBsm is 10m² and −10dBsm is 0.1m². The units drop-down switches the tab between the two.

The coloured bar under the number places it against rough typical values at 77GHz:

1. Pedestrian: about −10 to 0dBsm
2. Cyclist: about 0 to +10dBsm
3. Car: about +10 to +20dBsm

These are typical ranges from literature, and they're only for orientation. Measured values swing by 10dB or more with aspect, posture and the radar itself. I still need to check these exact ranges against published 77GHz measurements of pedestrians, cyclists and cars, and I'll cite them here once I have. Until then, treat the bar as a sanity check, not a reference.

## Equivalent sphere

A metal sphere much larger than the wavelength has an RCS of πr², whatever direction you look at it from. Turning that around gives the diameter of a sphere with the same RCS as the scene:

d = 2·√(σ / π)

"Equivalent to a 1.2m metal sphere" means something to me in a way that 1.15m² doesn't. It's a comparison, not a physical claim. A 10cm corner reflector behind the saddle reads as a 6m sphere, which is the whole point of corner reflectors.

## Returned power

This is the share of transmitted power that makes it back to the radar, from the monostatic radar equation:

P_returned / P_transmitted = G²·λ²·σ / ((4π)³·R⁴)

It's in dB re transmitted, so −60dB is one millionth. I moved this down to a secondary line, because it's dominated by the 1/R⁴ term. Moving the radar from 2m to 50m drops it by about 50dB, while a reflector changes it by 10 to 15dB. Distance isn't something a designer controls; the change vs. biker alone is.

## Why values flicker, and band averaging

At 77GHz the wavelength is 3.89mm. The solver adds the reflections from every face as complex fields, so they can reinforce or cancel. Moving a part by 1mm changes a round-trip path by 2mm, which is half a wavelength. That can swing the total by several dB.

This was not a major success at first. I'd nudge a slider and watch the number jump, and I couldn't tell real changes from noise. The fix comes from how the radar itself works. Automotive radars are FMCW: they sweep their frequency over 1 to 4GHz around 77GHz, so what they see is already an average over a band.

The app now does the same. By default it solves at 9 frequencies from 76.5 to 77.5GHz and averages the RCS (as power) across them. The line under the headline says what the number was averaged over, and by how much it moved across those frequencies. For the baseline biker seen from behind at 10m, a single frequency gives +1.3dBsm, the band average gives +0.6dBsm, and the spread across the band is a whopping 9.6dB. That spread is the flicker you'd otherwise be looking at.

Band averaging is cheap. Finding which faces are lit and tracing every multi-bounce ray doesn't depend on frequency, so the solver does that once and only redoes the field sums per frequency (see [How the solver works](solver.md)). Nine frequencies cost about 10% more than one.

In Advanced mode, "Solver settings" lets you change the bandwidth and the number of frequencies, or set the bandwidth to 0 for a single 77GHz solve. There's also an option to average over ±1° of radar bearing (three bearings, so three times the solve time). It's off by default.

## The per-object table

The table lists each object's own RCS, largest first, with:

1. **Share**: its value as a percentage of the sum of all the objects' values. The objects add coherently inside themselves but not with each other, so these shares sum to 100% while the coherent total can be higher or lower than the sum.
2. **vs. biker alone** (attachments only): how much that one part would lift the biker's total on its own, 10·log₁₀(1 + σ_part / σ_biker).

Click a row to flash that object in the 3D view.

## The heat map

The heat map (above the 3D view, in both modes) colours each face by its contribution in dBsm, over the top 50dB. It's the single most useful picture for a designer: bright faces are the ones worth reshaping, angling or covering.

Each face's own power is worked out by the solver, then every object's faces are scaled so they add up to that object's value in the table. Without that, a corner reflector looks dim. Its rays add in phase, so its total is far bigger than the sum of its faces' individual powers. Within an object, the spread between faces is still each face's own power.
