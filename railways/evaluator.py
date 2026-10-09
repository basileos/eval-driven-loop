import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import numbers
import os
import struct
import sys
import time
import traceback
import cv2
import numpy as np

if __package__:
    from .evaluation_config import POOL_TIMEOUT_SECONDS
else:
    from evaluation_config import POOL_TIMEOUT_SECONDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATH = os.path.join(BASE_DIR, "candidate_script.py")
VAL_DIR = os.path.join(BASE_DIR, "val_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
MAX_WORKERS = min(
    4, max(1, int(os.environ.get("RAILWAYS_EVAL_WORKERS", DEFAULT_WORKERS)))
)
MIN_ACCEPTABLE_RAIL_LENGTH_RATIO = 0.05

RailPolyline = list[list[float]]
Track = dict[str, RailPolyline]
EvaluationSample = tuple[str, str, list[Track]]

_WORKER_PROCESS_IMAGE = None
_WORKER_INIT_ERROR = None


def _squared_distance(point_a: list[float], point_b: list[float]) -> float:
    return (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2


def track_to_polygon(track: Track) -> np.ndarray:
    """Close the area between a track's two rail polylines."""
    left_rail = track["left_rail"]
    right_rail = track["right_rail"]

    same_direction_cost = _squared_distance(
        left_rail[0], right_rail[0]
    ) + _squared_distance(left_rail[-1], right_rail[-1])
    opposite_direction_cost = _squared_distance(
        left_rail[0], right_rail[-1]
    ) + _squared_distance(left_rail[-1], right_rail[0])

    if opposite_direction_cost < same_direction_cost:
        right_rail = list(reversed(right_rail))

    boundary = left_rail + list(reversed(right_rail))
    return np.rint(np.asarray(boundary, dtype=np.float64)).astype(np.int32)


def rasterize_track_polygons(
    tracks: list[Track], img_shape: tuple[int, int]
) -> np.ndarray:
    """Rasterize the union of all areas bounded by paired railway rails."""
    mask = np.zeros(img_shape, dtype=np.uint8)
    for track in tracks:
        polygon = track_to_polygon(track)
        if len(polygon) >= 4:
            cv2.fillPoly(mask, [polygon], color=255)
    return mask


def calculate_polygon_iou(
    gt_tracks: list[Track],
    pred_tracks: list[Track],
    img_shape: tuple[int, int],
) -> float:
    """Compute IoU between the filled ground-truth and predicted track areas."""
    intersection, union = calculate_polygon_overlap(
        gt_tracks, pred_tracks, img_shape
    )
    return float(intersection / union) if union else 0.0


def calculate_polygon_overlap(
    gt_tracks: list[Track],
    pred_tracks: list[Track],
    img_shape: tuple[int, int],
) -> tuple[int, int]:
    """Return intersection and union pixels for dataset-level polygon IoU."""
    metrics = calculate_polygon_metrics(gt_tracks, pred_tracks, img_shape)
    return int(metrics["intersection"]), int(metrics["union"])


def _mask_geometry(mask: np.ndarray) -> tuple[int, list[int] | None, list[float] | None]:
    """Return area, bounding box, and normalized centroid for a binary mask."""
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0, None, None

    x, y, width, height = cv2.boundingRect(mask)
    moments = cv2.moments(mask, binaryImage=True)
    image_height, image_width = mask.shape
    centroid = [
        float(moments["m10"] / moments["m00"] / max(1, image_width - 1)),
        float(moments["m01"] / moments["m00"] / max(1, image_height - 1)),
    ]
    return area, [x, y, x + width - 1, y + height - 1], centroid


def calculate_polygon_metrics(
    gt_tracks: list[Track],
    pred_tracks: list[Track],
    img_shape: tuple[int, int],
) -> dict[str, object]:
    """Compute overlap plus geometry diagnostics for one image."""
    gt_mask = rasterize_track_polygons(gt_tracks, img_shape)
    pred_mask = rasterize_track_polygons(pred_tracks, img_shape)

    intersection = int(np.count_nonzero(cv2.bitwise_and(gt_mask, pred_mask)))
    union = int(np.count_nonzero(cv2.bitwise_or(gt_mask, pred_mask)))
    gt_area, gt_bbox, gt_centroid = _mask_geometry(gt_mask)
    pred_area, pred_bbox, pred_centroid = _mask_geometry(pred_mask)

    centroid_dx = None
    centroid_dy = None
    if gt_centroid is not None and pred_centroid is not None:
        centroid_dx = float(pred_centroid[0] - gt_centroid[0])
        centroid_dy = float(pred_centroid[1] - gt_centroid[1])

    return {
        "intersection": intersection,
        "union": union,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "gt_bbox": gt_bbox,
        "pred_bbox": pred_bbox,
        "gt_centroid": gt_centroid,
        "pred_centroid": pred_centroid,
        "centroid_dx": centroid_dx,
        "centroid_dy": centroid_dy,
        "coverage": float(intersection / gt_area) if gt_area else 0.0,
        "precision": float(intersection / pred_area) if pred_area else 0.0,
    }


def calculate_prediction_size_metrics(
    pred_tracks: list[Track], img_shape: tuple[int, int]
) -> dict[str, object]:
    """Measure candidate rail length and flag visually insignificant tracks."""
    image_height, image_width = img_shape
    image_diagonal = math.hypot(image_width, image_height)
    minimum_acceptable = MIN_ACCEPTABLE_RAIL_LENGTH_RATIO * image_diagonal
    shorter_rail_lengths = []
    undersized_tracks = 0

    for track in pred_tracks:
        rail_lengths = []
        all_points = []
        for rail_name in ("left_rail", "right_rail"):
            points = np.asarray(track[rail_name], dtype=np.float64)
            rail_lengths.append(
                float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            )
            all_points.append(points)

        shorter_length = min(rail_lengths)
        shorter_rail_lengths.append(shorter_length)
        combined = np.concatenate(all_points, axis=0)
        corridor_span = float(np.linalg.norm(np.ptp(combined, axis=0)))
        if (
            shorter_length < minimum_acceptable
            or corridor_span < minimum_acceptable
        ):
            undersized_tracks += 1

    minimum_length = min(shorter_rail_lengths, default=None)
    return {
        "minimum_predicted_rail_length_px": minimum_length,
        "minimum_predicted_rail_length_ratio": (
            float(minimum_length / image_diagonal)
            if minimum_length is not None and image_diagonal
            else None
        ),
        "undersized_predicted_tracks": undersized_tracks,
    }


def read_image_shape(img_path: str) -> tuple[int, int]:
    """Read image height/width without decoding a full PNG when possible."""
    with open(img_path, "rb") as image_file:
        header = image_file.read(24)

    if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
        width, height = struct.unpack(">II", header[16:24])
        if width > 0 and height > 0:
            return height, width

    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"Could not read image '{img_path}'.")
    return image.shape[:2]


def _validate_polyline(
    value: object,
    *,
    track_index: int,
    rail_name: str,
    img_shape: tuple[int, int],
) -> RailPolyline:
    h, w = img_shape
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(
            f"Track {track_index} {rail_name} must contain at least two "
            "[x, y] points."
        )

    normalized = []
    for point_index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"Track {track_index} {rail_name} point {point_index} must be "
                "exactly [x, y]."
            )

        x, y = point
        if any(
            isinstance(coordinate, bool)
            or not isinstance(coordinate, numbers.Real)
            or not math.isfinite(float(coordinate))
            for coordinate in (x, y)
        ):
            raise ValueError(
                f"Track {track_index} {rail_name} point {point_index} must contain "
                "two finite numeric coordinates."
            )

        x = float(x)
        y = float(y)
        if not (0.0 <= x < w and 0.0 <= y < h):
            raise ValueError(
                f"Track {track_index} {rail_name} point {point_index} coordinate "
                f"[{x}, {y}] is outside the {w}x{h} image."
            )

        normalized.append([x, y])

    return normalized


def validate_predicted_tracks(
    value: object, img_shape: tuple[int, int]
) -> list[Track]:
    """Validate paired rail polylines returned for each predicted track."""
    if not isinstance(value, list):
        raise TypeError(
            "process_image must return a list of track dictionaries; "
            f"got {type(value).__name__}."
        )

    normalized = []
    for track_index, track in enumerate(value):
        if not isinstance(track, dict):
            raise ValueError(
                f"Track {track_index} must be a dictionary with left_rail and "
                "right_rail keys."
            )
        if "left_rail" not in track or "right_rail" not in track:
            raise ValueError(
                f"Track {track_index} must contain both left_rail and right_rail."
            )

        normalized.append(
            {
                "left_rail": _validate_polyline(
                    track["left_rail"],
                    track_index=track_index,
                    rail_name="left_rail",
                    img_shape=img_shape,
                ),
                "right_rail": _validate_polyline(
                    track["right_rail"],
                    track_index=track_index,
                    rail_name="right_rail",
                    img_shape=img_shape,
                ),
            }
        )

    return normalized


def load_railway_ground_truth(val_dir: str) -> dict[str, list[Track]]:
    """
    Scans val_dir for images and pairs them with <image_name>.json.
    Extracts paired left_rail and right_rail polylines for every track.
    By dataset convention, an image without JSON is a negative or an image whose
    railway is not acceptable for scoring, so it receives no tracks ([]).
    """
    ground_truth = {}
    if not os.path.exists(val_dir):
        return ground_truth

    for filename in sorted(os.listdir(val_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        base_name = os.path.splitext(filename)[0]
        json_path = os.path.join(val_dir, f"{base_name}.json")

        tracks = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                for track in data.get("tracks", []):
                    left_rail = track.get("left_rail", [])
                    right_rail = track.get("right_rail", [])

                    # The optional third point value is reserved for future use.
                    # It is intentionally ignored by the current evaluator.
                    if len(left_rail) >= 2 and len(right_rail) >= 2:
                        tracks.append({
                            "left_rail": [[p[0], p[1]] for p in left_rail],
                            "right_rail": [[p[0], p[1]] for p in right_rail],
                        })

            except Exception as e:
                print(
                    f"Warning: Failed to parse annotation {json_path}: {e}",
                    file=sys.stderr,
                )

        # Missing JSON and annotations without valid tracks both mean 0 tracks.
        ground_truth[filename] = tracks

    return ground_truth


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


def initialize_worker(candidate_path: str) -> None:
    """Load candidate code once in each process and avoid nested OpenCV pools."""
    global _WORKER_PROCESS_IMAGE, _WORKER_INIT_ERROR

    cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)

    try:
        candidate = load_candidate_module(candidate_path)
        process_image = getattr(candidate, "process_image", None)
        if not callable(process_image):
            raise AttributeError(
                "candidate_script.py must export a callable process_image function."
            )
        _WORKER_PROCESS_IMAGE = process_image
    except BaseException:
        _WORKER_INIT_ERROR = traceback.format_exc(limit=12)


def evaluate_sample(sample: EvaluationSample) -> dict[str, object]:
    """Run and score one image inside an evaluator worker process."""
    img_name, img_path, gt_tracks = sample

    try:
        if _WORKER_INIT_ERROR is not None:
            raise RuntimeError(
                "Candidate initialization failed:\n" + _WORKER_INIT_ERROR
            )
        if _WORKER_PROCESS_IMAGE is None:
            raise RuntimeError("Candidate worker was not initialized.")

        img_shape = read_image_shape(img_path)
        started_at = time.perf_counter()
        raw_prediction = _WORKER_PROCESS_IMAGE(img_path)
        elapsed = time.perf_counter() - started_at
        pred_tracks = validate_predicted_tracks(raw_prediction, img_shape)

        metrics = calculate_polygon_metrics(
            gt_tracks, pred_tracks, img_shape
        )
        size_metrics = calculate_prediction_size_metrics(pred_tracks, img_shape)
        intersection = int(metrics["intersection"])
        union = int(metrics["union"])
        score = float(intersection / union) if union else 1.0

        return {
            "image": img_name,
            "image_shape": img_shape,
            "score": score,
            **metrics,
            **size_metrics,
            "elapsed": elapsed,
            "ground_truth_tracks": len(gt_tracks),
            "predicted_tracks": len(pred_tracks),
            "ground_truth_data": gt_tracks,
            "prediction_data": pred_tracks,
            "error": None,
        }
    except BaseException:
        return {
            "image": img_name,
            "image_shape": None,
            "score": 0.0,
            "intersection": 0,
            "union": 0,
            "gt_area": 0,
            "pred_area": 0,
            "gt_bbox": None,
            "pred_bbox": None,
            "gt_centroid": None,
            "pred_centroid": None,
            "centroid_dx": None,
            "centroid_dy": None,
            "coverage": 0.0,
            "precision": 0.0,
            "minimum_predicted_rail_length_px": None,
            "minimum_predicted_rail_length_ratio": None,
            "undersized_predicted_tracks": 0,
            "elapsed": 0.0,
            "ground_truth_tracks": len(gt_tracks),
            "predicted_tracks": 0,
            "ground_truth_data": gt_tracks,
            "prediction_data": [],
            "error": traceback.format_exc(limit=12),
        }


def evaluate_samples_parallel(
    samples: list[EvaluationSample],
    candidate_path: str = CANDIDATE_PATH,
) -> tuple[list[dict[str, object]], int, float]:
    """Evaluate images concurrently with a hard deadline and clean shutdown."""
    worker_count = min(MAX_WORKERS, len(samples))
    context = mp.get_context("spawn")
    pool = context.Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(candidate_path,),
    )
    started_at = time.perf_counter()

    try:
        pending = {
            sample[0]: pool.apply_async(evaluate_sample, (sample,))
            for sample in samples
        }
        results = []
        deadline = started_at + POOL_TIMEOUT_SECONDS

        while pending:
            for img_name, async_result in list(pending.items()):
                if not async_result.ready():
                    continue

                result = async_result.get()
                del pending[img_name]
                if result["error"] is not None:
                    raise RuntimeError(
                        f"Evaluation failed for {img_name}:\n{result['error']}"
                    )
                results.append(result)

            if pending and time.perf_counter() >= deadline:
                unfinished = ", ".join(sorted(pending)[:8])
                raise TimeoutError(
                    f"EVALUATION_TIMEOUT: parallel evaluation exceeded {POOL_TIMEOUT_SECONDS:.1f}s. "
                    f"Completed {len(results)}/{len(samples)} images with {worker_count} workers; "
                    f"{len(pending)} queued or running (not necessarily individually stalled). "
                    f"First pending images: {unfinished}"
                )

            if pending:
                time.sleep(0.01)

        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise

    wall_time = time.perf_counter() - started_at
    results.sort(key=lambda result: str(result["image"]))
    return results, worker_count, wall_time


def _draw_track_overlay(
    panel: np.ndarray,
    tracks: list[Track],
    scale_x: float,
    scale_y: float,
    *,
    fill_color: tuple[int, int, int],
    left_color: tuple[int, int, int],
    right_color: tuple[int, int, int],
    alpha: float,
    line_thickness: int = 3,
) -> None:
    """Draw filled corridors and rail boundaries using explicit coordinate scales."""
    polygons = []
    rail_pairs = []
    panel_height, panel_width = panel.shape[:2]

    for track in tracks:
        rails = []
        for rail_name in ("left_rail", "right_rail"):
            points = []
            for point in track.get(rail_name, []):
                x = int(round(float(point[0]) * scale_x))
                y = int(round(float(point[1]) * scale_y))
                points.append([
                    int(np.clip(x, 0, panel_width - 1)),
                    int(np.clip(y, 0, panel_height - 1)),
                ])
            if len(points) >= 2:
                rails.append(np.asarray(points, dtype=np.int32))

        if len(rails) == 2:
            rail_pairs.append((rails[0], rails[1]))
            polygons.append(np.concatenate((rails[0], rails[1][::-1])))

    if polygons:
        fill = panel.copy()
        cv2.fillPoly(fill, polygons, fill_color)
        cv2.addWeighted(fill, alpha, panel, 1.0 - alpha, 0.0, dst=panel)

    for left_rail, right_rail in rail_pairs:
        cv2.polylines(
            panel,
            [left_rail],
            False,
            left_color,
            line_thickness,
            cv2.LINE_AA,
        )
        cv2.polylines(
            panel,
            [right_rail],
            False,
            right_color,
            line_thickness,
            cv2.LINE_AA,
        )


def write_diagnostic_montage(
    results: list[dict[str, object]],
    val_dir: str,
    output_path: str,
) -> bool:
    """Write a native-resolution image manifest (legacy montage function name)."""
    positives = [
        result for result in results if int(result["ground_truth_tracks"]) > 0
    ]
    negatives = [
        result for result in results if int(result["ground_truth_tracks"]) == 0
    ]

    selected = sorted(positives, key=lambda result: float(result["score"]))[:1]
    if positives:
        selected.append(max(positives, key=lambda result: float(result["score"])))
    false_positive_negatives = [
        result for result in negatives if int(result["predicted_tracks"]) > 0
    ]
    if false_positive_negatives:
        selected.extend(sorted(
            false_positive_negatives,
            key=lambda result: int(result["pred_area"]),
            reverse=True,
        )[:2])
        # Include a correct negative if only one false-positive example exists.
        selected.extend(result for result in negatives if int(result["predicted_tracks"]) == 0)
    elif negatives:
        selected.extend(negatives[:2])

    deduplicated = []
    seen_images = set()
    for result in selected + results:
        image_name = str(result["image"])
        if image_name in seen_images:
            continue
        seen_images.add(image_name)
        deduplicated.append(result)
        if len(deduplicated) == 4:
            break

    attachments = []
    output_path = os.path.abspath(output_path)
    output_dir = os.path.splitext(output_path)[0] + "_images"
    os.makedirs(output_dir, exist_ok=True)
    for result in deduplicated:
        image_name = str(result["image"])
        image = cv2.imread(os.path.join(val_dir, image_name), cv2.IMREAD_COLOR)
        if image is None:
            continue

        panel = image.copy()
        panel_height, panel_width = image.shape[:2]
        scale_x = scale_y = 1.0

        _draw_track_overlay(
            panel,
            result["ground_truth_data"],
            scale_x,
            scale_y,
            fill_color=(0, 215, 255),
            left_color=(0, 0, 255),
            right_color=(255, 80, 0),
            alpha=0.16,
        )
        _draw_track_overlay(
            panel,
            result["prediction_data"],
            scale_x,
            scale_y,
            fill_color=(70, 255, 70),
            left_color=(255, 255, 0),
            right_color=(255, 0, 255),
            alpha=0.20,
        )

        gt_area = int(result["gt_area"])
        pred_area = int(result["pred_area"])
        area_ratio = pred_area / gt_area if gt_area else float("inf")
        dx = result["centroid_dx"]
        dy = result["centroid_dy"]
        offset = (
            f"offset=({float(dx):+.2f},{float(dy):+.2f})"
            if dx is not None and dy is not None
            else "offset=n/a"
        )
        ratio_text = f"{area_ratio:.2f}" if math.isfinite(area_ratio) else "inf"
        cv2.rectangle(panel, (0, 0), (panel_width, 50), (0, 0, 0), -1)
        cv2.putText(
            panel,
            image_name[:66],
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"IoU={float(result['score']):.3f} "
                f"coverage={float(result['coverage']):.3f} "
                f"precision={float(result['precision']):.3f} "
                f"area_ratio={ratio_text} {offset}"
            ),
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "GT red/blue + yellow fill | PRED cyan/magenta + green fill",
            (8, panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panel_path = os.path.join(output_dir, f"{len(attachments):02d}.png")
        if not cv2.imwrite(panel_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Could not write {panel_path}")
        attachments.extend([
            {"path": os.path.abspath(os.path.join(val_dir, image_name)),
             "label": f"Original diagnostic: {image_name}"},
            {"path": panel_path,
             "label": f"GT/prediction overlay: {image_name}; IoU={float(result['score']):.3f}"},
        ])

    if not attachments:
        return False
    with open(output_path, "w") as output:
        json.dump({"images": attachments}, output, indent=2)
    return True


def _format_debug_metrics(result: dict[str, object]) -> dict[str, object]:
    """Select JSON-safe per-image geometry metrics from an evaluation result."""
    gt_area = int(result["gt_area"])
    pred_area = int(result["pred_area"])
    return {
        "image": str(result["image"]),
        "ground_truth_tracks": int(result["ground_truth_tracks"]),
        "predicted_tracks": int(result["predicted_tracks"]),
        "iou": float(result["score"]),
        "coverage": float(result["coverage"]),
        "precision": float(result["precision"]),
        "ground_truth_area_px": gt_area,
        "predicted_area_px": pred_area,
        "area_ratio": float(pred_area / gt_area) if gt_area else None,
        "centroid_dx": (
            float(result["centroid_dx"])
            if result["centroid_dx"] is not None
            else None
        ),
        "centroid_dy": (
            float(result["centroid_dy"])
            if result["centroid_dy"] is not None
            else None
        ),
        "ground_truth_bbox": result["gt_bbox"],
        "prediction_bbox": result["pred_bbox"],
        "elapsed_seconds": float(result["elapsed"]),
        "minimum_predicted_rail_length_px": result[
            "minimum_predicted_rail_length_px"
        ],
        "minimum_predicted_rail_length_ratio": result[
            "minimum_predicted_rail_length_ratio"
        ],
        "undersized_predicted_tracks": int(result["undersized_predicted_tracks"]),
    }


def write_debug_images(
    results: list[dict[str, object]],
    val_dir: str,
    output_dir: str,
) -> int:
    """Write a full-resolution annotated image for every validation sample."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    written = 0

    for result in results:
        image_name = str(result["image"])
        image = cv2.imread(os.path.join(val_dir, image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not render debug image for '{image_name}'.")

        height, width = image.shape[:2]
        line_thickness = max(3, int(round(width / 900)))
        _draw_track_overlay(
            image,
            result["ground_truth_data"],
            1.0,
            1.0,
            fill_color=(0, 215, 255),
            left_color=(0, 0, 255),
            right_color=(255, 80, 0),
            alpha=0.16,
            line_thickness=line_thickness,
        )
        _draw_track_overlay(
            image,
            result["prediction_data"],
            1.0,
            1.0,
            fill_color=(70, 255, 70),
            left_color=(255, 255, 0),
            right_color=(255, 0, 255),
            alpha=0.20,
            line_thickness=line_thickness,
        )

        metrics = _format_debug_metrics(result)
        ratio = metrics["area_ratio"]
        ratio_text = f"{ratio:.3f}" if ratio is not None else "n/a"
        dx = metrics["centroid_dx"]
        dy = metrics["centroid_dy"]
        minimum_rail_length = metrics["minimum_predicted_rail_length_px"]
        minimum_rail_ratio = metrics["minimum_predicted_rail_length_ratio"]
        rail_length_text = (
            f"{minimum_rail_length:.1f}px ({minimum_rail_ratio:.2%})"
            if minimum_rail_length is not None and minimum_rail_ratio is not None
            else "n/a"
        )
        offset_text = (
            f"({dx:+.3f}, {dy:+.3f})"
            if dx is not None and dy is not None
            else "n/a"
        )

        header_height = max(150, int(round(height * 0.075)))
        header = np.zeros((header_height, width, 3), dtype=np.uint8)
        font_scale = max(0.7, min(1.5, width / 2600))
        text_thickness = max(1, line_thickness // 2)
        lines = [
            (
                f"{image_name} | GT tracks={metrics['ground_truth_tracks']} | "
                f"pred tracks={metrics['predicted_tracks']}"
            ),
            (
                f"IoU={metrics['iou']:.4f} | coverage={metrics['coverage']:.4f} | "
                f"precision={metrics['precision']:.4f} | "
                f"area ratio={ratio_text} | centroid offset={offset_text}"
            ),
            (
                f"minimum predicted rail={rail_length_text} | "
                "GT: red/blue rails + yellow polygon | "
                "Candidate: cyan/magenta rails + green polygon"
            ),
        ]
        baseline_step = header_height // 4
        for index, line in enumerate(lines, start=1):
            cv2.putText(
                header,
                line,
                (18, baseline_step * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        canvas = np.vstack((header, image))
        stem = os.path.splitext(image_name)[0]
        output_name = f"{stem}__debug.jpg"
        output_path = os.path.join(output_dir, output_name)
        if not cv2.imwrite(
            output_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]
        ):
            raise OSError(f"Could not write debug image '{output_path}'.")

        metrics["debug_image"] = output_name
        manifest.append(metrics)
        written += 1

    manifest_path = os.path.join(output_dir, "metrics.json")
    with open(manifest_path, "w") as manifest_file:
        json.dump(
            {
                "legend": {
                    "ground_truth": "red/blue rails with yellow polygon",
                    "candidate": "cyan/magenta rails with green polygon",
                },
                "images": manifest,
            },
            manifest_file,
            indent=2,
        )
        manifest_file.write("\n")

    return written


def evaluate(
    candidate_path: str = CANDIDATE_PATH,
    val_dir: str = VAL_DIR,
    diagnostic_montage_path: str | None = None,
    diagnostic_output_dir: str | None = None,
):
    """Evaluate predictions with dataset-level filled track-polygon IoU."""
    candidate_path = os.path.abspath(candidate_path)
    val_dir = os.path.abspath(val_dir)
    ground_truth = load_railway_ground_truth(val_dir)

    if not ground_truth:
        print(f"Error: No valid images found in '{val_dir}'.", file=sys.stderr)
        sys.exit(1)

    samples = [
        (img_name, os.path.join(val_dir, img_name), gt_tracks)
        for img_name, gt_tracks in ground_truth.items()
    ]

    try:
        results, worker_count, wall_time = evaluate_samples_parallel(
            samples, candidate_path
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    total_intersection = sum(int(result["intersection"]) for result in results)
    total_union = sum(int(result["union"]) for result in results)
    total_gt_area = sum(int(result["gt_area"]) for result in results)
    total_pred_area = sum(int(result["pred_area"]) for result in results)
    dataset_iou = (
        float(total_intersection / total_union) if total_union else 0.0
    )
    positive_coverage = (
        float(total_intersection / total_gt_area) if total_gt_area else 0.0
    )
    polygon_precision = (
        float(total_intersection / total_pred_area) if total_pred_area else 0.0
    )
    slowest = max(results, key=lambda result: float(result["elapsed"]))
    positives = [
        result for result in results if int(result["ground_truth_tracks"]) > 0
    ]
    negatives = [
        result for result in results if int(result["ground_truth_tracks"]) == 0
    ]
    detected_positives = sum(
        int(result["predicted_tracks"]) > 0 for result in positives
    )
    correct_negatives = sum(
        int(result["predicted_tracks"]) == 0 for result in negatives
    )
    false_positive_rate = (
        float((len(negatives) - correct_negatives) / len(negatives))
        if negatives
        else 0.0
    )
    detected_positive_rate = (
        float(detected_positives / len(positives)) if positives else 0.0
    )
    mean_positive_iou = (
        sum(float(result["score"]) for result in positives) / len(positives)
        if positives
        else 0.0
    )
    total_predictions = sum(int(result["predicted_tracks"]) for result in results)
    undersized_predictions = sum(
        int(result["undersized_predicted_tracks"]) for result in results
    )

    if diagnostic_montage_path:
        try:
            if not write_diagnostic_montage(
                results, val_dir, diagnostic_montage_path
            ):
                print(
                    f"Warning: Could not write full-resolution diagnostic images to "
                    f"'{diagnostic_montage_path}'.",
                    file=sys.stderr,
                )
        except Exception as error:
            print(
                f"Warning: Failed to render full-resolution diagnostics: {error}",
                file=sys.stderr,
            )

    if diagnostic_output_dir:
        try:
            debug_image_count = write_debug_images(
                results, val_dir, diagnostic_output_dir
            )
            print(
                f"DEBUG: wrote {debug_image_count} annotated images and "
                f"metrics.json to '{os.path.abspath(diagnostic_output_dir)}'"
            )
        except Exception as error:
            print(
                f"Error: Failed to write per-image diagnostics: {error}",
                file=sys.stderr,
            )
            sys.exit(1)

    for result in results:
        elapsed = float(result["elapsed"])
        if elapsed > 1.0:
            print(
                f"WARNING: {result['image']} took {elapsed:.2f}s to process!",
                file=sys.stderr,
            )

    # Contract output format parsed by orchestrator.py regex
    print(
        f"EVAL: images={len(results)} workers={worker_count} "
        f"wall={wall_time:.2f}s slowest={slowest['image']}:{slowest['elapsed']:.2f}s"
    )
    print(
        f"SUMMARY: positives={len(positives)} detected={detected_positives} "
        f"mean_positive_iou={mean_positive_iou:.4f} negatives={len(negatives)} "
        f"correct_negatives={correct_negatives} total_predictions={total_predictions} "
        f"undersized_predictions={undersized_predictions} "
        f"intersection_px={total_intersection} union_px={total_union}"
    )
    print(
        f"METRICS: dataset_iou={dataset_iou:.6f} "
        f"mean_positive_iou={mean_positive_iou:.6f} "
        f"positive_coverage={positive_coverage:.6f} "
        f"polygon_precision={polygon_precision:.6f} "
        f"false_positive_rate={false_positive_rate:.6f} "
        f"detected_positive_rate={detected_positive_rate:.6f} "
        f"wall_seconds={wall_time:.3f}"
    )
    for result in results:
        gt_area = int(result["gt_area"])
        pred_area = int(result["pred_area"])
        area_ratio = pred_area / gt_area if gt_area else float("inf")
        area_ratio_text = (
            f"{area_ratio:.3f}" if math.isfinite(area_ratio) else "inf"
        )
        dx = result["centroid_dx"]
        dy = result["centroid_dy"]
        minimum_rail_length = result["minimum_predicted_rail_length_px"]
        minimum_rail_ratio = result["minimum_predicted_rail_length_ratio"]
        dx_text = f"{float(dx):+.3f}" if dx is not None else "na"
        dy_text = f"{float(dy):+.3f}" if dy is not None else "na"
        minimum_rail_text = (
            f"{float(minimum_rail_length):.1f}"
            if minimum_rail_length is not None
            else "na"
        )
        minimum_rail_ratio_text = (
            f"{float(minimum_rail_ratio):.4f}"
            if minimum_rail_ratio is not None
            else "na"
        )
        print(
            f"IMAGE: {result['image']} gt={result['ground_truth_tracks']} "
            f"pred={result['predicted_tracks']} iou={float(result['score']):.4f} "
            f"coverage={float(result['coverage']):.4f} "
            f"precision={float(result['precision']):.4f} "
            f"area_ratio={area_ratio_text} dx={dx_text} dy={dy_text} "
            f"min_rail_px={minimum_rail_text} "
            f"min_rail_ratio={minimum_rail_ratio_text}"
        )
    if undersized_predictions:
        print(
            f"DIAGNOSIS: UNDERSIZED DETECTIONS. {undersized_predictions} predicted "
            f"track(s) have a rail or corridor span below "
            f"{MIN_ACCEPTABLE_RAIL_LENGTH_RATIO:.0%} of the image diagonal and "
            "should be rejected as line-like clutter."
        )
    if total_predictions == 0 and positives:
        print(
            "DIAGNOSIS: EMPTY DETECTOR. Empty predictions receive zero dataset "
            "IoU; all positive railway images were missed. The next candidate "
            "must materially increase positive recall instead of returning []."
        )
    print(f"SCORE: {dataset_iou:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate paired railway polygons and render diagnostics."
    )
    parser.add_argument("candidate_path", nargs="?", default=CANDIDATE_PATH)
    parser.add_argument("val_dir", nargs="?", default=VAL_DIR)
    parser.add_argument(
        "diagnostic_montage_path",
        nargs="?",
        default=None,
        help="Optional JSON manifest path for separate native-resolution diagnostic images.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Write one full-resolution annotated image per validation sample.",
    )
    cli_args = parser.parse_args()
    evaluate(
        candidate_path=cli_args.candidate_path,
        val_dir=cli_args.val_dir,
        diagnostic_montage_path=cli_args.diagnostic_montage_path,
        diagnostic_output_dir=cli_args.debug_dir,
    )
