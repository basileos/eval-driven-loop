import importlib.util
import json
import os
import sys
import time

CANDIDATE_PATH = "candidate_script.py"
VAL_DIR = "val_dataset"
GROUND_TRUTH_PATH = os.path.join(VAL_DIR, "ground_truth.json")


def calculate_box_iou(box1: list[float], box2: list[float]) -> float:
    """Computes Intersection over Union (IoU) for two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0.0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - inter_area)
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def load_candidate_module(script_path: str):
    """Dynamically imports candidate_script.py directly from disk."""
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Candidate script '{script_path}' not found.")

    spec = importlib.util.spec_from_file_location("candidate_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from '{script_path}'.")

    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    return candidate


def evaluate():
    """Evaluates predictions from candidate_script.py against val_dataset ground truth."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: Ground truth file '{GROUND_TRUTH_PATH}' missing.", file=sys.stderr)
        sys.exit(1)

    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)

    try:
        candidate = load_candidate_module(CANDIDATE_PATH)
    except Exception as e:
        print(f"Error loading candidate script: {e}", file=sys.stderr)
        sys.exit(1)

    if not hasattr(candidate, "process_image"):
        print("Error: candidate_script.py must export a 'process_image' function.", file=sys.stderr)
        sys.exit(1)

    image_scores = []
    for img_name, gt_boxes in ground_truth.items():
        img_path = os.path.join(VAL_DIR, img_name)

        try:
            start_time = time.time()
            pred_boxes = candidate.process_image(img_path)
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                print(f"WARNING: {img_name} took {elapsed:.2f}s to process!", file=sys.stderr)
        except Exception as e:
            print(f"Error executing process_image on {img_name}: {e}", file=sys.stderr)
            sys.exit(1)

        # Handle edge cases (empty boxes / no detections)
        if not gt_boxes and not pred_boxes:
            image_scores.append(1.0)
            continue
        if not gt_boxes or not pred_boxes:
            image_scores.append(0.0)
            continue

        # Compute IoU matching for each ground truth target
        best_ious = []
        for gt in gt_boxes:
            iou = max([calculate_box_iou(pred, gt) for pred in pred_boxes], default=0.0)
            best_ious.append(iou)

        image_scores.append(sum(best_ious) / len(best_ious))

    mean_iou = sum(image_scores) / len(image_scores) if image_scores else 0.0

    # Contract output format parsed by orchestrator.pyo regex
    print(f"SCORE: {mean_iou:.4f}")


if __name__ == "__main__":
    evaluate()
