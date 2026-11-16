# Assumptions and limits

The solver is built to compare designs against each other, not to predict what one specific radar will measure. These are the things it leaves out or simplifies, roughly in order of how much I think they matter.

## Ground multipath

The road isn't in the solve; it's only drawn in the 3D view. A real radar 0.5 to 1m above asphalt sees the rider directly, and also by way of a bounce off the road. Those paths interfere, and that can add or remove several dB depending on distance and height. It's the biggest piece of missing physics.

Of note, it affects the biker alone and the biker with your part in much the same way. So the change vs. biker alone should hold up better than the absolute dBsm. I may add a ground plane with image theory later; for now, I'd treat absolute values as ±several dB.

## Near field

A target is in a radar's far field beyond about 2D²/λ, where D is its size. For a 2m bike at 3.89mm that's about 2km, so every distance in the app is in the near field. The solver handles this by giving every face its own direction and range from the radar (spherical waves), instead of one plane wave. The catch is that RCS is no longer a fixed property of the target; it changes with distance. That's expected for close-range automotive radar, and the app notes it under the result.

## Materials are estimates

Every preset's R, A and s are engineering estimates at 77GHz, worked out from typical permittivities. They're a reasonable starting point, not lab data (see [Materials](materials.md)). A material's values also don't change with angle in this model, which leads into the next point.

## No angle-dependent Fresnel coefficients

Real dielectrics reflect more at grazing angles than head-on. The solver uses each material's normal-incidence reflectance at every angle. For metals that's nearly exact. For plastics and skin, grazing returns are underestimated, but grazing faces contribute little in the monostatic case anyway.

## No polarization

The solver works with scalar fields. Real radars are linearly polarized, and some shapes (dihedrals especially) return different power for different polarizations. A trihedral corner reflector, the shape most attachments would use, returns the same for either one.

## No edge diffraction or creeping waves

Physical optics gets the specular return right, but it doesn't model the weak fields diffracted by sharp edges or the creeping waves that travel around curved surfaces. They matter for stealth aircraft. For a bike, they're tens of dB below the specular glints.

## Tessellation

Curved surfaces are built from flat triangles. If a facet is wider than the first Fresnel zone of its curvature (w ≈ √(r·λ/2)), a smooth tube turns into a row of flat mirrors and gives spiky, overestimated glints. The primitives in [radar_detector/geometry.py](../radar_detector/geometry.py) are tessellated to that rule. Imported meshes are used as they come, so a coarse STL export of a curved part can read high. The decimate-on-import option makes this worse if pushed too far.

## Shadowing is per face

The shadow test casts one ray per face, to its centroid. A face is either lit (with some transmission) or not, so a face that's half in shadow counts as fully lit or fully dark. On a fine mesh this averages out. On a few large faces it doesn't, which is why the sample corner reflector is subdivided.

## Multi-bounce is approximate

The multi-bounce rays use a 5mm grid by default. A ray either reflects or passes through at each face (it passes through when T ≥ 0.5), rather than splitting into both. Each tube's aperture factor treats it as a small square aperture. The 10cm trihedral lands within 1.5dB of theory with these settings (see [Validation](validation.md)).

## Retroreflectors are modelled by behaviour

A retroreflective material (see [Materials](materials.md)) is a phase-conjugating surface with a flat response out to its coverage angle. Real tags have structure inside that response. The metalens tag in Ghasemi et al. (2026) swings by about 10dB across its ±40°, peaks sharply at broadside and has grating lobes near ±50°. The fitted preset matches the tag's broadside peak and its median, not its ripple. Polarization also matters for these tags: the measured one was mounted for a vertically polarized radar, and the solver doesn't model polarization at all.

## The baseline rider is a stick figure

The baseline biker is built from tubes, spheres and tori: 41,828 faces of capsule limbs and a round head. It's close enough to give the right order of magnitude and a realistic mix of metal and body, but it isn't a scan of a real person. Its posture is fixed.

## The heat map is incoherent

Within an object, each face's colour in the heat map is its own contribution as power, scaled so the object's faces add up to its value in the table. It shows which objects and which parts of them matter, but not how faces within an object interfere with each other.
