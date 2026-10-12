"""Convert official COCO 2017 captions to the format used by this project."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(caption: str) -> list[str]:
    """Return lowercase word tokens without punctuation."""
    return TOKEN_PATTERN.findall(caption.lower())


def load_split(annotation_path: Path, image_folder: str, split: str) -> list[dict]:
    with annotation_path.open("r", encoding="utf-8") as source:
        coco = json.load(source)

    captions_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        raw = annotation["caption"].strip()
        captions_by_image[annotation["image_id"]].append(
            {"raw": raw, "tokens": tokenize(raw)}
        )

    images = []
    for image in coco["images"]:
        sentences = captions_by_image.get(image["id"], [])
        if not sentences:
            continue
        images.append(
            {
                "filepath": image_folder,
                "filename": image["file_name"],
                "imgid": image["id"],
                "split": split,
                "sentences": sentences,
            }
        )
    return images


def main() -> None:
    workspace = Path(__file__).resolve().parent.parent
    default_raw = workspace / "data" / "raw"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=default_raw)
    parser.add_argument("--output", type=Path, default=default_raw / "dataset_coco.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--validation-images", type=int, default=2500)
    args = parser.parse_args()

    annotation_dir = args.raw_dir / "annotations"
    train_images = load_split(
        annotation_dir / "captions_train2017.json", "train2017", "train"
    )
    validation_images = load_split(
        annotation_dir / "captions_val2017.json", "val2017", "val"
    )

    # COCO 2017 publishes one 5,000-image validation set. Split it
    # deterministically so this project's VAL and TEST files are both usable.
    random.Random(args.seed).shuffle(validation_images)
    for index, image in enumerate(validation_images):
        image["split"] = "val" if index < args.validation_images else "test"

    result = {
        "dataset": "coco2017",
        "images": train_images + validation_images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, ensure_ascii=True, separators=(",", ":"))

    counts = {name: 0 for name in ("train", "val", "test")}
    for image in result["images"]:
        counts[image["split"]] += 1
    print(f"Wrote {args.output}")
    print(f"Image counts: {counts}")


if __name__ == "__main__":
    main()
