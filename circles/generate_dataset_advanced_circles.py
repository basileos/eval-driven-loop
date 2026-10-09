import json
import os
import random
import cv2
import numpy as np

VAL_DIR = "val_dataset"
NUM_IMAGES = 20
IMG_SIZE = (600, 600)  # (Width, Height)


def generate_noisy_background(width: int, height: int) -> np.ndarray:
    """Generates a textured, non-uniform low-contrast background with Gaussian noise."""
    # Base background luminance gradient
    x = np.linspace(0, random.uniform(50, 100), width)
    y = np.linspace(0, random.uniform(50, 100), height)
    xx, yy = np.meshgrid(x, y)
    base = (xx + yy) / 2.0

    # Add Gaussian noise
    noise = np.random.normal(0, random.uniform(8, 18), (height, width))
    bg = np.clip(base + noise, 0, 255).astype(np.uint8)

    # Convert to 3-channel BGR
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)


def create_complex_synthetic_dataset():
    """Generates challenging synthetic validation images to stress-test CV models."""
    os.makedirs(VAL_DIR, exist_ok=True)
    ground_truth = {}

    for i in range(1, NUM_IMAGES + 1):
        filename = f"image_{i:03d}.png"
        file_path = os.path.join(VAL_DIR, filename)

        # 1. Start with noisy, non-uniform background
        img = generate_noisy_background(IMG_SIZE[0], IMG_SIZE[1])

        # 2. Add distractor shapes (e.g., rects/polygons) to create false positives
        num_distractors = random.randint(1, 3)
        for _ in range(num_distractors):
            rx1 = random.randint(20, IMG_SIZE[0] - 100)
            ry1 = random.randint(20, IMG_SIZE[1] - 100)
            rw = random.randint(30, 80)
            rh = random.randint(30, 80)
            color = (
                random.randint(60, 110),
                random.randint(60, 110),
                random.randint(60, 110),
            )
            cv2.rectangle(img, (rx1, ry1), (rx1 + rw, ry1 + rh), color, -1)

        # 3. Add target circles (including overlapping/partially occluded circles)
        num_circles = random.randint(2, 5)
        gt_boxes = []
        placed_circles = []

        for c_idx in range(num_circles):
            radius = random.randint(20, 65)

            # Allow 30% of circles to intentionally overlap with previous circles
            if placed_circles and random.random() < 0.35:
                ref_cx, ref_cy, ref_r = random.choice(placed_circles)
                # Position center close enough to cause bounding box overlaps
                offset_x = random.randint(-int(ref_r * 0.8), int(ref_r * 0.8))
                offset_y = random.randint(-int(ref_r * 0.8), int(ref_r * 0.8))
                cx = np.clip(ref_cx + offset_x, radius + 15, IMG_SIZE[0] - radius - 15)
                cy = np.clip(ref_cy + offset_y, radius + 15, IMG_SIZE[1] - radius - 15)
            else:
                cx = random.randint(radius + 15, IMG_SIZE[0] - radius - 15)
                cy = random.randint(radius + 15, IMG_SIZE[1] - radius - 15)

            cx, cy = int(cx), int(cy)
            placed_circles.append((cx, cy, radius))

            # Low contrast color variation (close to background intensities)
            intensity = random.randint(110, 160)
            color = (
                intensity + random.randint(-15, 15),
                intensity + random.randint(-15, 15),
                intensity + random.randint(-15, 15),
            )

            # Render circle
            cv2.circle(img, (cx, cy), radius, color, -1)

            # Compute Ground Truth Box [x1, y1, x2, y2]
            x1 = float(max(0, cx - radius))
            y1 = float(max(0, cy - radius))
            x2 = float(min(IMG_SIZE[0], cx + radius))
            y2 = float(min(IMG_SIZE[1], cy + radius))

            gt_boxes.append([x1, y1, x2, y2])

        # 4. Apply final Blur / Speckle noise pass to degrade sharp edge gradients
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (5, 5), 1.5)

        # Write image file & record annotations
        cv2.imwrite(file_path, img)
        ground_truth[filename] = gt_boxes

    # Save Ground Truth annotations
    gt_path = os.path.join(VAL_DIR, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated {NUM_IMAGES} hard validation images in '{VAL_DIR}/'")
    print(f"Ground truth written to '{gt_path}'")


if __name__ == "__main__":
    create_complex_synthetic_dataset()