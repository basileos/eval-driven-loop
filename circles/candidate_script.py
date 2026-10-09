import cv2
import numpy as np


def process_image(img_path: str) -> list[list[float]]:
    try:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    except Exception:
        return []

    if image is None or image.size == 0 or image.ndim != 3:
        return []

    height, width = image.shape[:2]
    min_dim = min(height, width)
    if min_dim < 7:
        return []

    output = []

    def emit(cx, cy, radius, deltas=(0.0,)):
        if not np.all(np.isfinite((cx, cy, radius))) or radius < 2.0:
            return

        for delta in deltas:
            r = radius + delta
            x1 = float(np.clip(cx - r, 0.0, width - 1.0))
            y1 = float(np.clip(cy - r, 0.0, height - 1.0))
            x2 = float(np.clip(cx + r, 0.0, width - 1.0))
            y2 = float(np.clip(cy + r, 0.0, height - 1.0))

            if x2 > x1 and y2 > y1:
                output.append([
                    float(round(x1, 3)),
                    float(round(y1, 3)),
                    float(round(x2, 3)),
                    float(round(y2, 3)),
                ])

    def add_arc_models(contour):
        if contour is None or len(contour) < 18:
            return

        points = contour[:, 0, :].astype(np.float64)
        if len(points) > 180:
            indices = np.linspace(
                0, len(points) - 1, 180, dtype=np.int32
            )
            points = points[indices]

        count = len(points)
        if count < 18:
            return

        triplets = []
        for fraction in (28, 20, 14, 10, 7):
            gap = max(4, count // fraction)
            starts = np.linspace(
                0, count - 1, min(14, count), dtype=np.int32
            )
            for start in starts:
                triplets.append((
                    int(start),
                    int((start + gap) % count),
                    int((start + 2 * gap) % count),
                ))

        rng = np.random.default_rng(712367 + count)
        random_triplets = rng.integers(0, count, size=(52, 3))
        triplets.extend(map(tuple, random_triplets.tolist()))

        px = points[:, 0]
        py = points[:, 1]
        models = []

        for first, second, third in triplets:
            if len({first, second, third}) < 3:
                continue

            x1, y1 = points[first]
            x2, y2 = points[second]
            x3, y3 = points[third]

            denominator = 2.0 * (
                x1 * (y2 - y3)
                + x2 * (y3 - y1)
                + x3 * (y1 - y2)
            )
            if abs(denominator) < 5.0:
                continue

            q1 = x1 * x1 + y1 * y1
            q2 = x2 * x2 + y2 * y2
            q3 = x3 * x3 + y3 * y3

            cx = (
                q1 * (y2 - y3)
                + q2 * (y3 - y1)
                + q3 * (y1 - y2)
            ) / denominator
            cy = (
                q1 * (x3 - x2)
                + q2 * (x1 - x3)
                + q3 * (x2 - x1)
            ) / denominator
            radius = float(np.hypot(x1 - cx, y1 - cy))

            if (
                not np.all(np.isfinite((cx, cy, radius)))
                or radius < 8.0
                or radius > 0.46 * min_dim
                or cx < -radius
                or cy < -radius
                or cx > width - 1 + radius
                or cy > height - 1 + radius
            ):
                continue

            tolerance = float(
                np.clip(0.9 + 0.009 * radius, 1.05, 1.75)
            )
            residual = np.abs(np.hypot(px - cx, py - cy) - radius)
            keep = residual <= tolerance
            inlier_count = int(np.count_nonzero(keep))

            if inlier_count < max(9, int(0.055 * np.pi * radius)):
                continue

            ix = px[keep]
            iy = py[keep]
            matrix = np.column_stack((
                2.0 * ix,
                2.0 * iy,
                np.ones(inlier_count),
            ))

            try:
                solution, _, rank, _ = np.linalg.lstsq(
                    matrix, ix * ix + iy * iy, rcond=None
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            if rank < 3:
                continue

            new_cx, new_cy, constant = solution
            radius_squared = (
                constant + new_cx * new_cx + new_cy * new_cy
            )
            if radius_squared <= 0.0:
                continue

            new_radius = float(np.sqrt(radius_squared))
            if (
                new_radius < 8.0
                or new_radius > 0.46 * min_dim
                or np.hypot(new_cx - cx, new_cy - cy)
                > max(4.0, 0.13 * radius)
                or abs(new_radius - radius)
                > max(4.0, 0.13 * radius)
            ):
                continue

            residual = np.abs(
                np.hypot(px - new_cx, py - new_cy) - new_radius
            )
            keep = residual <= tolerance
            inlier_count = int(np.count_nonzero(keep))
            if inlier_count < 9:
                continue

            angles = np.mod(
                np.arctan2(
                    py[keep] - new_cy,
                    px[keep] - new_cx,
                ),
                2.0 * np.pi,
            )
            bins = np.minimum(
                71,
                (
                    angles * (72.0 / (2.0 * np.pi))
                ).astype(np.int32),
            )
            coverage = len(np.unique(bins)) / 72.0
            if coverage < 0.10:
                continue

            score = (
                inlier_count
                + 45.0 * coverage
                - 8.0 * float(np.median(residual[keep]))
            )
            models.append((
                score,
                float(new_cx),
                float(new_cy),
                new_radius,
            ))

        models.sort(reverse=True)
        selected = []

        for model in models:
            _, cx, cy, radius = model
            duplicate = any(
                np.hypot(cx - previous[1], cy - previous[2])
                <= max(2.0, 0.045 * max(radius, previous[3]))
                and abs(radius - previous[3])
                <= max(2.0, 0.055 * max(radius, previous[3]))
                for previous in selected
            )
            if duplicate:
                continue

            selected.append(model)
            emit(
                cx,
                cy,
                radius,
                (0.0, 0.55, 1.1, 1.8, 2.7, 3.7),
            )

            if len(selected) >= 9:
                break

    flat_image = image.reshape(-1, 3)
    packed_colors = (
        (flat_image[:, 0].astype(np.uint32) << 16)
        | (flat_image[:, 1].astype(np.uint32) << 8)
        | flat_image[:, 2].astype(np.uint32)
    )

    color_values, color_counts = np.unique(
        packed_colors, return_counts=True
    )
    frequent = np.flatnonzero(color_counts >= 100)
    chromatic_colors = []

    if frequent.size:
        ordered = frequent[
            np.argsort(color_counts[frequent])[::-1]
        ]

        for index in ordered:
            value = int(color_values[index])
            blue = (value >> 16) & 255
            green = (value >> 8) & 255
            red = value & 255

            if max(blue, green, red) - min(blue, green, red) >= 2:
                chromatic_colors.append(index)
                if len(chromatic_colors) >= 18:
                    break

    for index in chromatic_colors:
        mask = (
            packed_colors == int(color_values[index])
        ).reshape(height, width).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if len(contours) > 60:
            contours = sorted(
                contours,
                key=lambda contour: abs(cv2.contourArea(contour)),
                reverse=True,
            )[:60]

        for contour in contours:
            if (
                len(contour) >= 18
                and abs(float(cv2.contourArea(contour))) >= 35.0
            ):
                add_arc_models(contour)

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    except cv2.error:
        return output

    planes = (
        gray,
        lab[:, :, 1],
        lab[:, :, 2],
    )

    for plane in planes:
        if float(np.std(plane)) < 0.7:
            continue

        smooth = cv2.GaussianBlur(plane, (5, 5), 0.9)

        for threshold_mode in (
            cv2.THRESH_BINARY,
            cv2.THRESH_BINARY_INV,
        ):
            _, mask = cv2.threshold(
                smooth,
                0,
                255,
                threshold_mode | cv2.THRESH_OTSU,
            )
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )

            if len(contours) > 300:
                contours = sorted(
                    contours,
                    key=lambda contour: abs(cv2.contourArea(contour)),
                    reverse=True,
                )[:300]

            for contour in contours:
                area = abs(float(cv2.contourArea(contour)))
                perimeter = float(cv2.arcLength(contour, True))

                if area < 30.0 or perimeter < 20.0:
                    continue

                x, y, box_width, box_height = cv2.boundingRect(contour)
                span_x = float(box_width - 1)
                span_y = float(box_height - 1)

                if min(span_x, span_y) < 8.0:
                    continue
                if (
                    max(span_x, span_y)
                    / max(1.0, min(span_x, span_y))
                    > 1.16
                ):
                    continue

                circularity = (
                    4.0 * np.pi * area
                    / (perimeter * perimeter + 1e-9)
                )
                if circularity < 0.68:
                    continue

                output.append([
                    float(x),
                    float(y),
                    float(x + box_width - 1),
                    float(y + box_height - 1),
                ])

    hough_input = cv2.GaussianBlur(gray, (5, 5), 1.0)
    try:
        circles = cv2.HoughCircles(
            hough_input,
            cv2.HOUGH_GRADIENT,
            dp=1.3,
            minDist=max(6, min_dim // 65),
            param1=82,
            param2=10,
            minRadius=3,
            maxRadius=max(5, int(0.32 * min_dim)),
        )
    except cv2.error:
        circles = None

    if circles is not None:
        for cx, cy, radius in circles[0][:100]:
            emit(
                float(cx),
                float(cy),
                float(radius),
                (-1.2, 0.0, 1.2),
            )

    if not output:
        return []

    unique_output = []
    seen = set()

    for box in output:
        key = tuple(round(float(value), 2) for value in box)
        if key in seen:
            continue
        seen.add(key)
        unique_output.append([float(value) for value in box])

    unique_output.sort(
        key=lambda box: (box[1], box[0], box[3] - box[1])
    )
    return unique_output