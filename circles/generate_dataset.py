import json
import os
import random
import cv2
import numpy as np

VAL_DIR = "val_dataset"
NUM_IMAGES = 10
IMG_SIZE = (500, 500)  # (Width, Height)


def create_synthetic_dataset():
    """Generates synthetic images with circles and outputs ground_truth.json."""
    os.makedirs(VAL_DIR, exist_ok=True)
    ground_truth = {}

    for i in range(1, NUM_IMAGES + 1):
        filename = f"image_{i:03d}.png"
        file_path = os.path.join(VAL_DIR, filename)

        # Base image: Dark background with mild random Gaussian noise
        img = np.random.randint(
            0, 30, (IMG_SIZE[1], IMG_SIZE[0], 3), dtype=np.uint8
        )

        num_circles = random.randint(1, 3)
        gt_boxes = []

        for _ in range(num_circles):
            radius = random.randint(25, 60)
            # Ensure circle stays strictly within image boundaries
            cx = random.randint(radius + 10, IMG_SIZE[0] - radius - 10)
            cy = random.randint(radius + 10, IMG_SIZE[1] - radius - 10)

            # Random bright color for contrast
            color = (
                random.randint(120, 255),
                random.randint(120, 255),
                random.randint(120, 255),
            )

            # Render solid circle
            cv2.circle(img, (cx, cy), radius, color, -1)

            # Compute ground-truth bounding box: [x1, y1, x2, y2]
            x1 = float(cx - radius)
            y1 = float(cy - radius)
            x2 = float(cx + radius)
            y2 = float(cy + radius)

            gt_boxes.append([x1, y1, x2, y2])

        # Write image file
        cv2.imwrite(file_path, img)
        ground_truth[filename] = gt_boxes

    # Write ground truth JSON file
    gt_path = os.path.join(VAL_DIR, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Dataset ready: {NUM_IMAGES} images created in '{VAL_DIR}/'")
    print(f"Ground truth saved to '{gt_path}'")


if __name__ == "__main__":
    create_synthetic_dataset()