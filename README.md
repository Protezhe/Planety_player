# Planety Player

NDI face tracking with head-pose estimation, WebSocket overlay updates, and OBS scene control.

## Structure

- `planety_player/` - core package (tracker, config, OBS, WebSocket, head pose)
- `overlay.html` - browser overlay for snapshots
- `ndi_face_tracker.py` - legacy entrypoint wrapper (calls the package)
- `requirements.txt` - dependencies

## Run

```bash
source venv/bin/activate
python -m planety_player
```

## Configuration (CLI)

Examples:

```bash
python -m planety_player --source-name NDI_OBS --obs-password "secret"
python -m planety_player --snapshot-interval inf
python -m planety_player --face-model-path path/to/model.pt
```

Run `python -m planety_player --help` for the full list of options.
