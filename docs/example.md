# Worked example: a corner reflector behind the saddle

To demo the whole loop, I add the classic radar-visibility part, a 10cm aluminum corner reflector, and see what it does. Every number here is what the app gives with the default settings.

## 1. The biker alone

With nothing added, the radar sits 10m directly behind the rider. The Results tab shows the baseline biker at +0.6dBsm, averaged over 76.5 to 77.5GHz. That's equivalent to a 1.2m metal sphere, and it sits in the typical cyclist range on the bar.

The line under the headline says the value moves by 9.6dB across the band. At 77GHz alone it would read +1.3dBsm, which is a good reminder of why I average.

There's no change vs. biker alone yet, because there's nothing to compare.

## 2. Add the reflector

Under Geometry, press "Add sample", then "Corner reflector (10cm, aluminum)". It lands behind the saddle, above the rear wheel, with its mouth facing backwards.

The headline jumps to **+14.0dB**, in green: about 25× the return. The total is now +14.6dBsm, which is equivalent to a 6.1m metal sphere and well into the typical car range.

For a sanity check, the closed-form RCS of a 10cm trihedral is +14.4dBsm. Nearly all of the scene's return is coming from the corner, which the table confirms: "Corner reflector" sits on top at +14.5dBsm and 97% of the share, and on its own it lifts the biker's total by +14.1dB.

## 3. Look at the heat map

Tick "Show where the return comes from" above the 3D view. The inside faces of the corner light up brightest: all of the 10 strongest faces in the scene, and 186 of the strongest 200, are on the reflector. The next brightest are the bottom bracket shell and the rear hub, which are short metal cylinders facing the radar.

This was not a major success initially. The first version of the heat map coloured each face by its own power alone, and the bike frame outshone the corner even though the corner made 97% of the return. A corner works because its rays add in phase, and per-face power can't show that. Now each object's faces are scaled to add up to that object's value in the table, so the picture and the table agree.

## 4. Run the aspect sweep

Open the Compare tab, choose "Aspect sweep" and press "Run aspect sweep 0–360°". It sweeps both the current scene and the biker alone, 144 solves in about 3s.

The polar plot shows the catch. The reflector adds more than 3dB from about 35° left of straight behind to 35° right of it, peaking at +21dB about 5° off the rear. From the sides and the front it adds nothing at all (0.0dB), because the corner's mouth only faces backwards.

## 5. Run the distance comparison

Choose "Vs. distance" and press "Run comparison" (200 solves, about 5s). From the rear, the reflector adds +25.7dB at 2m, +14.0dB at 10m and about +16dB at 30 to 50m. At 2m the rider is only partly inside the 20° cone while the reflector sits near its centre, which inflates the change. From the left, right and front, every distance shows 0.0dB.

## 6. Swap in a retroreflective tag

To compare, delete the corner reflector and add "Metalens retro tag (53mm, measured)" from "Add sample". It lands in the same spot. This is the tag from Ghasemi et al. (2026), with the preset fitted to their measurements.

From straight behind, the change vs. biker alone is +5.8dB. Turn the bearing to 10°, 20° or 30° in Advanced mode and it drops to +0.2 to +0.8dB, inside the solver's noise. Off-axis the tag returns about −17dBsm, and the biker alone returns +0.6dBsm, so the tag barely moves the total. The per-object table still shows the tag's own return holding steady, which is what the paper measured. Their radar had 3.6cm range resolution, so it could separate the tag from the rider.

Swapping in "Ideal retroreflector (53mm)" shows what the same footprint could do. It adds more than 3dB at every bearing from 60° left of the rear to 60° right of it, which is wider than the 10cm corner's ±35°, from a part a quarter of the corner's size. It peaks at +15.5dB, 10° off the rear.

| Part behind the saddle | Rear (0°) | 20° | 40° | Beyond 3dB |
|---|---|---|---|---|
| 10cm corner reflector | +14.0dB | +16.9dB | −0.1dB | ±35° |
| 53mm metal plate | +9.2dB | −0.2dB | 0.0dB | 0° only |
| 53mm metalens tag (measured) | +5.8dB | +0.3dB | +0.4dB | 0° only |
| 53mm ideal retroreflector | +9.4dB | +14.5dB | +10.4dB | ±60° |

## What this says about the design

1. A corner reflector is a very effective rear-facing part. It took the rider from "cyclist" to "car" on the scale from behind.
2. It does nothing for a car coming from the side or head-on. Covering those needs reflectors facing those ways too.
3. The ±35° rear lobe is narrow enough that a turning car could miss it, so it's worth checking at the bearing a car actually approaches from.
4. A retroreflective tag trades peak return for coverage. The measured metalens tag is too weak to change a whole rider's total RCS off-axis. An ideal one of the same size would beat the corner everywhere except dead astern. The gap between the two is the room left for better tags.
