# FacialRec

FacialRec is a local Python and OpenCV prototype for exploring the basic stages of a real-time face-matching workflow: loading reference images, generating encodings, detecting faces from a camera stream, comparing distances, and displaying matched identities.

## What it demonstrates

- Loading and validating image data from a directory
- Generating face encodings with `face_recognition`
- Processing live frames with OpenCV
- Comparing encoded representations and applying a tolerance threshold
- Handling missing files, unreadable images, and camera failures
- Separating the application into testable functions and a command-line entry point

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python facialRec.py --images-dir Images
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

Press `q` to close the camera window. Use `--camera` to choose a different camera index and `--tolerance` to adjust matching strictness.

## Responsible use

This is a learning project, not a production identity system. Face-recognition systems can produce inaccurate and uneven results across populations. Do not use this prototype for surveillance, access control, or consequential decisions without rigorous evaluation, consent, privacy protection, and appropriate oversight.

