# Radar Detector

Radar Detector is a tool I built to estimate how visible a cyclist is to a car's 77GHz radar, and how much a bike or helmet attachment changes that. This page covers how to get it running.

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

All 7 tests should pass in a few seconds. They compare the solver against closed-form answers for a 10cm flat plate, sphere and corner reflector.

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
