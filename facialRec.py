import argparse
from pathlib import Path

import cv2
import numpy as np
import face_recognition


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_reference_encodings(images_dir: Path):
    encodings = []
    names = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detected = face_recognition.face_encodings(rgb_image)
        if not detected:
            print(f"Skipping image with no detectable face: {image_path.name}")
            continue

        encodings.append(detected[0])
        names.append(image_path.stem)

    if not encodings:
        raise ValueError(f"No usable reference faces found in {images_dir}")

    return encodings, names


def annotate_matches(frame, known_encodings, known_names, tolerance):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    for candidate, location in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, candidate)
        match_index = int(np.argmin(distances))
        name = "Unknown"

        if distances[match_index] <= tolerance:
            name = known_names[match_index]

        top, right, bottom, left = (coordinate * 4 for coordinate in location)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            frame,
            name.upper(),
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def run_camera(images_dir: Path, camera_index: int, tolerance: float):
    known_encodings, known_names = load_reference_encodings(images_dir)
    print(f"Loaded {len(known_names)} reference identities.")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {camera_index}")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Camera stopped returning frames.")

            annotated = annotate_matches(frame, known_encodings, known_names, tolerance)
            cv2.imshow("FacialRec", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local face-matching prototype.")
    parser.add_argument("--images-dir", type=Path, default=Path("Images"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.6)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera(args.images_dir, args.camera, args.tolerance)


