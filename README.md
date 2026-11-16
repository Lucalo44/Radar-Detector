# Radar Detector

Radar Detector is a tool I built to estimate how visible a cyclist is to a car's 77GHz radar, and how much a bike or helmet attachment changes that. This page covers how to get it running.

## What's new

1. **The headline is now the change vs. the biker alone**, in dB and in plain words ("about 2.2× the return"), coloured by whether the part makes the rider more or less visible. The total RCS sits under it on a scale of typical pedestrian, cyclist and car values, with the size of a metal sphere that would give the same return.
2. **Results are averaged over the radar's band** (76.5 to 77.5GHz, 9 frequencies), like a real FMCW radar. Nudging a part by a millimetre no longer swings the number by several dB.
3. **The right side is now tabs**: Scene, Results, Compare and Docs. The aspect sweep and the distance charts run inside the Compare tab, with a progress bar and a cancel button, instead of in pop-ups or straight to disk.
4. **The Docs tab** explains what every number means, how the solver works step by step, what it leaves out, the materials and how the solver is validated. The same pages are in [docs/](docs/).
5. **It's faster.** Solving runs in the background so the window never freezes. A single solve takes about 0.15s, the 72-angle sweep about 1.4s (it was 25s) and the 200-solve chart set about 5s (it was 70s).
6. **The heat map is available in Simple mode too**, and dense imports can be decimated on the way in.
7. **Retroreflective materials.** Faces can now send a share of their reflected power straight back to the radar over a set coverage angle, like a metalens tag or Van Atta array. Two presets come with it: an ideal retroreflector, and one fitted to the measured 53mm metalens tag in Ghasemi, Hester and Eid (*Scientific Reports*, 2026). Both are under "Add sample" as 53mm tags.

Of note, two numbers changed on purpose. Each object's value in the table now adds its multi-bounce return coherently, which a corner reflector needs (it used to show up as 11% of a return it made 97% of). The radar also now always aims at the biker's centre when the biker is shown, so adding a part never moves the aim.

## Setup

It needs Python 3.10 or newer. I developed it on 3.12 on a Mac. To check your version, run `python3 --version`.

In order to keep its dependencies away from everything else on your machine, the tool runs in its own virtual environment:

```bash
git clone https://github.com/Lucalo44/Radar-Detector.git
cd Radar-Detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The install pulls in about 11 packages, including PySide6 and VTK for the interface and trimesh with Embree for ray casting. Expect it to take a few minutes the first time.

After doing so, launch it with:

```bash
python run.py
```

## Checking the install

To make sure the physics is behaving, run the test suite:

```bash
python -m pytest tests
```

All 46 tests should pass in a few seconds. Seven of them compare the solver against closed-form answers for a 10cm flat plate, sphere and corner reflector. The rest are described in the Docs tab under Validation, and in [docs/validation.md](docs/validation.md).

To check which ray-casting engine is active and time one solve, run:

```bash
python run.py --check
```

The last lines should read `Ray engine  trimesh.ray.ray_pyembree.RayMeshIntersector` and a test solve of under a second.

## If solves are slow

The tool relies on Embree (installed as `embreex`) for ray casting. If it's missing, trimesh quietly falls back to a pure-Python ray tracer. Nothing breaks, but a single solve goes from under a second to 20s or more, and a sweep becomes impractical. The app shows an amber banner across the top when this happens.

Embree installs on Intel and Apple Silicon Macs, Windows and x86-64 Linux. Linux on ARM has no prebuilt wheel, so `requirements.txt` skips it there. To try installing it by hand:

```bash
pip install embreex
python run.py --check
```

## Running it again later

The virtual environment has to be active each time you open a new terminal:

```bash
cd Radar-Detector
source .venv/bin/activate
python run.py
```

## Updating

To get the latest version, quit the app first, then pull and reinstall in case the requirements changed:

```bash
cd Radar-Detector
git pull
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Of note, an open window keeps running the old code, so a full restart is required. I also lost some time to a second clone sitting in my home folder. I kept pulling into one copy and launching the other, which was not a major success. If an update doesn't seem to take, check which folder you're actually running from.

## How it calculates

The solver in [radar_detector/rcs.py](radar_detector/rcs.py) works on the triangle mesh of the whole scene, where every face carries a material with a reflectance R, an absorptance A and a roughness s. At 77GHz the wavelength is 3.89mm, so most faces are many wavelengths across and can't be treated as points.

The radar cross section σ is the sum of three mechanisms:

1. **Specular single bounce (physical optics).** Each lit, unshadowed face contributes the exact closed-form integral of the round-trip phase across its triangle, weighted by its reflection amplitude √(R(1−s)) and the beam strength at that face. These add coherently, so faces can reinforce or cancel each other. This gives σ_spec = (4π/λ²)·|Σ field|².
2. **Diffuse single bounce.** The rough share s of each face's reflected power scatters like a Lambertian surface (σ = 4·ρ·area·cos²θ, with ρ = R·s). These terms add as power, not phase.
3. **Multi-bounce (shooting and bouncing rays).** A grid of ray tubes (5mm spacing by default) is fired from the radar and traced through up to 3 reflections. Tubes that come back after two or more bounces are added coherently to the physical optics field. This is the only part that catches corner reflectors and dihedrals.

Specifically, each face sees its own direction and range from the radar rather than one plane wave, and the beam falls off as a Gaussian with the chosen −3dB cone. Ray casting handles shadowing, and thin plastics dim what's behind them by their transmittance instead of blocking it.

Returned power comes from the monostatic radar equation, with an ideal Gaussian-beam antenna gain of G = 16·ln2/θ² for a cone of θ radians (about 19.6dBi at 20°):

P_returned / P_transmitted = G²·λ²·σ / ((4π)³·R⁴)

That 1/R⁴ term is why returned power drops by roughly 50dB between 2m and 50m, even when σ barely changes. The tool also reports what share of the transmitted power lands on the target and how much of that is reflected, absorbed or passed through. It ignores losses, ground-plane multipath, polarization, edge diffraction and the change in reflectance with angle.

The Docs tab in the app goes through each step in more detail, with a diagram per step, along with what the numbers mean and what the solver leaves out.
