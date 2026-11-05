# Radar Detector

Most new cars watch the road with a 77GHz radar, and a cyclist is one of the harder things for that radar to see. A rider is mostly water and fabric, and the bike is a handful of thin tubes. The goal of this project was a tool to answer a simple question: if I bolt something onto a bike, how much more visible does it make the rider to a car's radar? Radar Detector imports 3D geometry, lets me place and orient it on a baseline rider, assigns materials to its surfaces, and computes the radar cross section (RCS) and the fraction of radar power that makes it back to the car.

## How it came together

In order to get a number worth trusting, the first thing I built was the solver in [radar_detector/rcs.py](radar_detector/rcs.py), well before any interface. At 77GHz the wavelength is 3.89mm, so almost every face on a bike-sized mesh is many wavelengths across. That rules out the lazy approach of treating each triangle as a point. Instead, each lit triangle gets the exact closed-form integral of the round-trip phase across its surface (physical optics). This formula has an ugly habit of dividing by zero whenever two corners of a triangle sit at the same phase. Writing it as a divided difference with a sinc term fixed that, and it now stays stable on any triangle.

Physical optics only covers single bounces, which is a problem for a bike accessory tool. The classic way to make something radar-visible is a corner reflector, and a corner reflector works entirely through three bounces. With this in mind, I added a shooting-and-bouncing-rays pass. It fires a grid of ray tubes from the radar (5mm spacing by default) and follows each one through up to three reflections. Every tube that leaves on its second or later bounce gets added coherently to the physical optics field. Single bounces stay with physical optics so nothing is counted twice.

Of note, a few effects that matter on a real road are folded in as well:

1. Spherical-wave illumination, so each face sees its own direction and range rather than one plane wave.
2. A Gaussian beam with an adjustable cone, so geometry near the edge of the beam counts for less.
3. Shadowing by ray casting (Embree via trimesh). Opaque surfaces block what's behind them, while thin plastics only dim it by their transmittance.
4. A diffuse term for rough materials like skin and fabric, which scatter rather than mirror.

Before trusting it on a bicycle, I checked it against closed-form answers in [tests/test_rcs.py](tests/test_rcs.py). A 10cm flat plate at normal incidence lands within 0.00dB of 4πA²/λ². A 10cm sphere comes in at +0.38dB of πr², and a 10cm triangular trihedral is within −0.28dB of 4πa⁴/3λ². The whole 37,000-face rider solves in about 0.2s.

The baseline biker in [radar_detector/geometry.py](radar_detector/geometry.py) is deliberately barebones: tubes, spheres and tori for the rider, frame, wheels and 32 steel spokes. Its first run put the rider at a suspiciously loud +8dBsm from behind. Sorting faces by contribution showed that nearly all of it came from the back of the saddle. I had modeled it as a box, and its flat rear face was acting as a 60cm² mirror aimed straight at the radar. This was not a major success. A real saddle is rounded, so the saddle is now an ellipsoid, and the rear view dropped to a more believable +1.3dBsm. The same lesson set the mesh resolution. Every curved primitive is tessellated so each flat facet is narrower than √(rλ/2), which is the width of the first Fresnel zone on that curve. Coarser facets turn a smooth tube into a row of tiny flat mirrors and give spiky, inflated glints.

With the physics settled, I built the interface in [radar_detector/gui.py](radar_detector/gui.py) on PySide6 and PyVista. The 3D view draws the radar as a red box with its beam cone, plus a line to the aim point labeled with the distance. Changing any setting re-runs the solver after a quarter-second pause, so moving the distance slider updates the results almost live.

## Materials

Each surface has a reflectance R, an absorptance A, a transmittance T = 1 − R − A, and a roughness that sets how much of the reflected power scatters diffusely. The library in [radar_detector/materials.py](radar_detector/materials.py) holds my engineering estimates at 77GHz. For example, skin at ε ≈ 6.5 − j8 works out to R ≈ 0.35. They're estimates rather than lab measurements, so the advanced mode lets me override any of them.

## Running it

It needs Python 3.10 or newer (I developed on 3.12):

```bash
git clone https://github.com/Lucalo44/Radar-Detector.git
cd Radar-Detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

To check the physics, run `python -m pytest tests`.

The tool has two modes, from simplest to most involved:

1. **Simple mode.** The radar sits directly behind the rider with a 20° cone and multi-bounce on.
   1. Click **Import geometry…** to load an STL, OBJ, PLY, VTK or glTF file and pick its units. Alternatively, use **Add sample** to drop in a corner reflector, flat plate or sphere.
   2. Imports start flush against the back of the rider's head, which is where a helmet accessory would sit. Move it with Position, Rotation and Scale (+X is the direction of travel, +Y is the rider's left, +Z is up). Then pick its material.
   3. Set the distance with the slider. The result panel shows **Returning radar power** (the percentage of transmitted power that gets back to the radar) and **Total radar cross section** (in m² and dBsm).
   4. Untick an imported part in the geometry list to leave it out without deleting it. This makes it easy to flip back and forth and see what the part adds. **Delete** removes it for good.
   5. Uncheck **Show baseline biker** to analyze the geometry by itself.
2. **Advanced mode** adds everything else:
   1. Radar settings: cone width, distance, angle of incidence (0° from behind, 90° from the rider's left, 180° head-on), elevation, and the multi-bounce bounce count and ray spacing.
   2. Face-level materials. Click a surface in the 3D view to select it. A surface ends wherever neighboring faces bend by more than the edge angle (20° by default). Shift+click adds or removes a surface, Option+click toggles a single triangle, and clicking empty space clears the selection. Choose a material, adjust its R, A and roughness if needed, and click **Assign to selected faces**. **Save as new material…** keeps a custom material for reuse.
   3. A breakdown into specular, diffuse and multi-bounce RCS, a beam energy budget, and an RCS table for each object.
   4. **Show per-face RCS heat map** paints each face by its contribution, which shows exactly which surface is doing the reflecting.
   5. **Aspect sweep 0–360°** plots RCS against the angle of incidence on a polar chart, with a table and CSV export.
   6. **Save comparison charts…** computes returned power from 2m to 50m (2m steps) from the rear, both sides and the front. It runs the biker alone and the biker with every ticked attachment, aimed at the same point. It saves a four-panel chart of both cases, a chart of the difference in dB, and a CSV of every value. Expect about 45s.

To demo this, add the sample corner reflector in Simple mode. At 10m the rider jumps from +1.3dBsm to +14.2dBsm, a 10cm piece of aluminum outshining the entire bike and rider by about 13dB.

## Limits

Specifically, the solver leaves out ground-plane multipath off the road, the change in material reflectance with angle (Fresnel), polarization, and edge diffraction. Returned power uses the ideal gain of a Gaussian beam with the chosen cone, with no losses. At the close ranges a car radar cares about, the target is in the radar's near field, so RCS changes with distance. The tool flags this whenever it applies.
