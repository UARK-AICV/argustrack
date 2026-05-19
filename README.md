# ArgusTrack

ArgusTrack is a annotation tool for **multi-camera** setups. It shows all camera views on one screen, supports **3D bounding boxes** projected to a **bird's-eye view (BEV)**, and saves annotations in a shared format under your dataset root.

Use it when you need consistent object IDs across cameras and frames, with calibration-aware placement on the ground plane.

## Features

- Open a dataset root and browse frames from every camera at once
- Draw and edit boxes per camera; link objects with global IDs across views
- BEV panel for top-down placement and ground-point editing
- Annotations stored under `annotations_positions/` as JSON (one file per frame)
- Optional preprocessing (YOLO-based detection + cross-camera matching) to bootstrap labels

## Dataset layout

In the app, use **File → Open Dir** and select the **root folder** of your dataset. The expected layout is:

```
dataset_root/
├── Image_subsets/           # frame images, grouped by camera
│   ├── 1/                   # camera folder (name can be 1, 2, … or similar)
│   │   ├── 00000.jpg
│   │   └── ...
│   ├── 2/
│   └── ...
├── calibrations/            # one folder per camera
│   ├── Camera1/
│   ├── Camera2/
│   └── ...
└── annotations_positions/   # written when you save (may not exist yet)
    ├── 00000.json
    └── ...
```

- **Image_subsets**: each subfolder is one camera; image filenames should sort consistently by frame index.
- **calibrations**: camera intrinsics/extrinsics used for BEV projection (folder names like `Camera1`, `Camera2`, …).
- **annotations_positions**: global per-frame annotations; created and updated by the tool.

If both `Image_subsets/` and `calibrations/` are present, ArgusTrack switches to multi-camera mode automatically.

## Start

**Requirements:** Python 3.9+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url>
cd argustrack
uv sync
uv run argustrack
```

Then **File → Open Dir** → choose `dataset_root`.

User settings are saved to `~/.argustrackrc` on first run. Logs:

- Windows: `%LOCALAPPDATA%\argustrack\argustrack.log`
- Linux / macOS: `~/.cache/argustrack/argustrack.log`

### Optional: detection-assisted preprocessing

For automatic box proposals before manual refinement:

```bash
uv pip install ultralytics opencv-python
```

Use **Preprocess** from the menu after opening a dataset (requires the layout above).

## Acknowledgement

This tool is built from [wkentaro/labelme](https://github.com/wkentaro/labelme).
