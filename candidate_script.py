import math

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def process_image(img_path: str) -> list[dict[str, list[list[float]]]]:
    if cv2 is None or np is None:
        return []

    try:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.size == 0:
            return []

        source_h, source_w = image.shape[:2]
        if source_h < 16 or source_w < 16:
            return []

        try:
            cv2.setNumThreads(1)
            cv2.ocl.setUseOpenCL(False)
        except (AttributeError, cv2.error):
            pass

        resize_scale = min(1.0, 960.0 / max(source_h, source_w))
        if resize_scale < 1.0:
            work_w = max(16, int(round(source_w * resize_scale)))
            work_h = max(16, int(round(source_h * resize_scale)))
            image = cv2.resize(
                image, (work_w, work_h), interpolation=cv2.INTER_AREA
            )

        height, width = image.shape[:2]
        diagonal = math.hypot(width, height)
        scale_x = source_w / float(width)
        scale_y = source_h / float(height)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(
            clipLimit=1.7, tileGridSize=(8, 8)
        ).apply(gray)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0.7)

        median = float(np.median(smooth))
        low = int(np.clip(0.50 * median, 14, 105))
        high = int(np.clip(max(low + 28, 1.42 * median), 45, 225))
        edges = cv2.Canny(
            smooth, low, high, apertureSize=3, L2gradient=True
        )

        min_segment_length = max(12.0, 0.014 * diagonal)
        segments = []

        def add_lines(lines, limit):
            if lines is None:
                return
            values = np.asarray(lines, dtype=np.float64)
            if values.size < 4:
                return
            values = values.reshape(-1, 4)
            values = values[np.isfinite(values).all(axis=1)]
            if values.size == 0:
                return

            delta = values[:, 2:4] - values[:, 0:2]
            lengths = np.hypot(delta[:, 0], delta[:, 1])
            keep = lengths >= min_segment_length
            values = values[keep]
            lengths = lengths[keep]
            if values.size == 0:
                return

            if len(values) > limit:
                indices = np.argpartition(lengths, -limit)[-limit:]
                values = values[indices]
                lengths = lengths[indices]

            for row, length_value in zip(values, lengths):
                length = float(length_value)
                if not math.isfinite(length) or length <= 0.0:
                    continue

                p1 = row[:2].astype(np.float64)
                p2 = row[2:].astype(np.float64)
                direction = (p2 - p1) / length
                if direction[0] < 0.0 or (
                    abs(direction[0]) < 1e-10 and direction[1] < 0.0
                ):
                    direction = -direction

                segments.append(
                    {
                        "p1": p1,
                        "p2": p2,
                        "mid": 0.5 * (p1 + p2),
                        "direction": direction,
                        "angle": math.atan2(
                            float(direction[1]), float(direction[0])
                        ) % math.pi,
                        "length": length,
                    }
                )

        try:
            detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)
            detection = detector.detect(smooth)
            if detection and detection[0] is not None:
                add_lines(detection[0], 420)
        except (AttributeError, TypeError, ValueError, cv2.error):
            pass

        try:
            hough = cv2.HoughLinesP(
                edges,
                1.0,
                np.pi / 360.0,
                threshold=max(17, int(round(0.018 * diagonal))),
                minLineLength=max(14, int(round(0.023 * diagonal))),
                maxLineGap=max(5, int(round(0.014 * diagonal))),
            )
            add_lines(hough, 180)
        except (TypeError, ValueError, cv2.error):
            pass

        if len(segments) < 2:
            return []

        segments.sort(key=lambda item: item["length"], reverse=True)
        segments = segments[:400]

        def angle_distance(a, b):
            difference = abs(a - b) % math.pi
            return min(difference, math.pi - difference)

        clusters = []
        angle_limit = math.radians(6.5)
        lateral_limit = max(2.8, 0.0035 * diagonal)

        for segment in segments:
            best = None
            best_cost = float("inf")

            for cluster in clusters:
                angle_error = angle_distance(
                    segment["angle"], cluster["angle"]
                )
                if angle_error > angle_limit:
                    continue

                displacement = segment["mid"] - cluster["center"]
                lateral_error = abs(
                    float(np.dot(displacement, cluster["normal"]))
                )
                if lateral_error > lateral_limit:
                    continue

                cost = lateral_error + 8.0 * angle_error
                if cost < best_cost:
                    best = cluster
                    best_cost = cost

            if best is None:
                direction = segment["direction"].copy()
                clusters.append(
                    {
                        "segments": [segment],
                        "center": segment["mid"].copy(),
                        "direction": direction,
                        "normal": np.array(
                            [-direction[1], direction[0]], dtype=np.float64
                        ),
                        "angle": segment["angle"],
                        "weight": segment["length"],
                    }
                )
                continue

            old_weight = best["weight"]
            added_weight = segment["length"]
            total_weight = old_weight + added_weight
            best["segments"].append(segment)
            best["center"] = (
                best["center"] * old_weight
                + segment["mid"] * added_weight
            ) / total_weight

            direction = segment["direction"].copy()
            if float(np.dot(direction, best["direction"])) < 0.0:
                direction = -direction
            direction = (
                best["direction"] * old_weight + direction * added_weight
            )
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                direction /= norm
                best["direction"] = direction
                best["normal"] = np.array(
                    [-direction[1], direction[0]], dtype=np.float64
                )
                best["angle"] = math.atan2(
                    float(direction[1]), float(direction[0])
                ) % math.pi
            best["weight"] = total_weight

        fitted = []
        max_gap = max(7.0, 0.016 * diagonal)

        for cluster in clusters:
            points = np.asarray(
                [
                    point
                    for segment in cluster["segments"]
                    for point in (segment["p1"], segment["p2"])
                ],
                dtype=np.float64,
            )
            if len(points) < 2 or not np.isfinite(points).all():
                continue

            weights = np.repeat(
                np.asarray(
                    [segment["length"] for segment in cluster["segments"]],
                    dtype=np.float64,
                ),
                2,
            )
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue

            center = np.average(points, axis=0, weights=weights)
            centered = points - center
            covariance = (
                (centered * weights[:, None]).T @ centered
            ) / weight_sum

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(eigenvalues).all():
                continue

            direction = eigenvectors[:, -1].astype(np.float64)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            direction /= norm
            if direction[0] < 0.0 or (
                abs(direction[0]) < 1e-10 and direction[1] < 0.0
            ):
                direction = -direction

            intervals = []
            for segment in cluster["segments"]:
                a = float(np.dot(segment["p1"] - center, direction))
                b = float(np.dot(segment["p2"] - center, direction))
                intervals.append((min(a, b), max(a, b)))
            intervals.sort()

            merged = []
            for start, end in intervals:
                if not merged or start > merged[-1][1] + max_gap:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)

            coverage = float(sum(end - start for start, end in merged))
            projections = centered @ direction
            span = float(projections.max() - projections.min())

            if span < 0.050 * diagonal:
                continue
            if coverage < 0.026 * diagonal:
                continue

            fitted.append(
                {
                    "center": center,
                    "direction": direction,
                    "angle": math.atan2(
                        float(direction[1]), float(direction[0])
                    ) % math.pi,
                    "points": points,
                    "coverage": coverage,
                    "span": span,
                }
            )

        if len(fitted) < 2:
            return []

        fitted.sort(
            key=lambda line: line["coverage"] + 0.2 * line["span"],
            reverse=True,
        )
        fitted = fitted[:30]

        min_separation = max(4.0, 0.0045 * min(width, height))
        max_separation = min(
            220.0, max(32.0, 0.27 * min(width, height))
        )
        min_overlap = max(27.0, 0.043 * diagonal)

        def point_at(line, axis, projection):
            denominator = float(np.dot(line["direction"], axis))
            if abs(denominator) < 1e-8:
                return None
            parameter = (
                projection - float(np.dot(line["center"], axis))
            ) / denominator
            point = line["center"] + parameter * line["direction"]
            if not np.isfinite(point).all():
                return None
            return point

        pairs = []

        for i in range(len(fitted)):
            first = fitted[i]
            for j in range(i + 1, len(fitted)):
                second = fitted[j]
                angle_error = angle_distance(
                    first["angle"], second["angle"]
                )
                if angle_error > math.radians(11.0):
                    continue

                second_direction = second["direction"].copy()
                if float(
                    np.dot(first["direction"], second_direction)
                ) < 0.0:
                    second_direction = -second_direction

                axis = first["direction"] + second_direction
                norm = float(np.linalg.norm(axis))
                if norm < 1e-8:
                    continue
                axis /= norm
                normal = np.array([-axis[1], axis[0]], dtype=np.float64)

                first_projection = first["points"] @ axis
                second_projection = second["points"] @ axis
                start = max(
                    float(first_projection.min()),
                    float(second_projection.min()),
                )
                end = min(
                    float(first_projection.max()),
                    float(second_projection.max()),
                )
                overlap = end - start
                if overlap < min_overlap:
                    continue

                first_start = point_at(first, axis, start)
                first_end = point_at(first, axis, end)
                second_start = point_at(second, axis, start)
                second_end = point_at(second, axis, end)
                if any(
                    point is None
                    for point in (
                        first_start,
                        first_end,
                        second_start,
                        second_end,
                    )
                ):
                    continue

                signed_start = float(
                    np.dot(second_start - first_start, normal)
                )
                signed_end = float(
                    np.dot(second_end - first_end, normal)
                )
                if signed_start * signed_end <= 0.0:
                    continue

                separation_start = abs(signed_start)
                separation_end = abs(signed_end)
                separation = 0.5 * (
                    separation_start + separation_end
                )
                if not min_separation <= separation <= max_separation:
                    continue

                perspective_ratio = max(
                    separation_start, separation_end
                ) / max(min(separation_start, separation_end), 1e-6)
                if perspective_ratio > 3.2:
                    continue

                coverage_ratio = min(
                    1.0,
                    min(first["coverage"], second["coverage"])
                    / max(overlap, 1.0),
                )
                parallel_score = math.exp(
                    -0.5
                    * (angle_error / math.radians(5.8)) ** 2
                )
                geometry = (
                    overlap
                    / diagonal
                    * (0.52 + 0.48 * coverage_ratio)
                    * parallel_score
                )

                pairs.append(
                    {
                        "first": i,
                        "second": j,
                        "axis": axis,
                        "normal": normal,
                        "start": start,
                        "end": end,
                        "overlap": overlap,
                        "separation": separation,
                        "geometry": geometry,
                        "first_start": first_start,
                        "first_end": first_end,
                        "second_start": second_start,
                        "second_end": second_end,
                    }
                )

        if not pairs:
            return []

        pairs.sort(key=lambda pair: pair["geometry"], reverse=True)
        pairs = pairs[:56]

        segment_midpoints = np.asarray(
            [segment["mid"] for segment in segments], dtype=np.float64
        )
        segment_angles = np.asarray(
            [segment["angle"] for segment in segments], dtype=np.float64
        )
        segment_lengths = np.asarray(
            [segment["length"] for segment in segments], dtype=np.float64
        )
        gray_float = smooth.astype(np.float32) / 255.0
        scored = []

        for pair in pairs:
            first = fitted[pair["first"]]
            second = fitted[pair["second"]]
            axis = pair["axis"]
            overlap = pair["overlap"]
            separation = pair["separation"]

            first_denominator = float(
                np.dot(first["direction"], axis)
            )
            second_denominator = float(
                np.dot(second["direction"], axis)
            )
            if (
                abs(first_denominator) < 1e-8
                or abs(second_denominator) < 1e-8
            ):
                continue

            sample_count = int(np.clip(round(overlap * 0.5), 30, 88))
            longitudinal = np.linspace(
                pair["start"], pair["end"], sample_count
            )

            first_parameter = (
                longitudinal - float(np.dot(first["center"], axis))
            ) / first_denominator
            second_parameter = (
                longitudinal - float(np.dot(second["center"], axis))
            ) / second_denominator

            first_curve = (
                first["center"]
                + first_parameter[:, None] * first["direction"]
            )
            second_curve = (
                second["center"]
                + second_parameter[:, None] * second["direction"]
            )

            fractions = np.linspace(-0.25, 1.25, 25)
            samples = (
                first_curve[:, None, :]
                + fractions[None, :, None]
                * (second_curve - first_curve)[:, None, :]
            )

            try:
                strip = cv2.remap(
                    gray_float,
                    samples[:, :, 0].astype(np.float32),
                    samples[:, :, 1].astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT101,
                )
            except cv2.error:
                continue

            if strip.shape != (sample_count, len(fractions)):
                continue

            inside = strip[
                :, (fractions >= 0.05) & (fractions <= 0.95)
            ]
            if inside.shape[0] < 4 or inside.shape[1] < 4:
                continue

            longitudinal_difference = np.diff(inside, axis=0)
            total_change = float(
                np.mean(np.abs(longitudinal_difference))
            ) + 1e-6
            coherent_change = float(
                np.mean(
                    np.abs(np.mean(longitudinal_difference, axis=1))
                )
            )
            sleeper_score = float(
                np.clip(
                    (coherent_change / total_change - 0.08) / 0.52,
                    0.0,
                    1.0,
                )
            )

            transverse_gradient = np.abs(np.diff(strip, axis=1))
            fraction_midpoints = 0.5 * (
                fractions[:-1] + fractions[1:]
            )
            left_index = int(
                np.argmin(np.abs(fraction_midpoints))
            )
            right_index = int(
                np.argmin(np.abs(fraction_midpoints - 1.0))
            )
            boundary_strength = 0.5 * (
                float(np.mean(transverse_gradient[:, left_index]))
                + float(np.mean(transverse_gradient[:, right_index]))
            )
            general_gradient = float(
                np.mean(transverse_gradient)
            ) + 1e-6
            boundary_score = float(
                np.clip(
                    (
                        boundary_strength / general_gradient - 0.65
                    ) / 1.25,
                    0.0,
                    1.0,
                )
            )

            axis_angle = math.atan2(
                float(axis[1]), float(axis[0])
            ) % math.pi
            angle_errors = np.abs(segment_angles - axis_angle)
            angle_errors = np.minimum(
                angle_errors, math.pi - angle_errors
            )
            positions = segment_midpoints @ axis

            crossing_mask = (
                (positions >= pair["start"])
                & (positions <= pair["end"])
                & (angle_errors >= math.radians(49.0))
                & (segment_lengths >= 0.14 * separation)
                & (segment_lengths <= 4.5 * separation)
            )
            crossing_indices = np.flatnonzero(crossing_mask)
            crossing_length = 0.0

            if crossing_indices.size:
                crossing_positions = positions[crossing_indices]
                crossing_midpoints = segment_midpoints[crossing_indices]

                first_cross_parameter = (
                    crossing_positions
                    - float(np.dot(first["center"], axis))
                ) / first_denominator
                second_cross_parameter = (
                    crossing_positions
                    - float(np.dot(second["center"], axis))
                ) / second_denominator

                first_cross = (
                    first["center"]
                    + first_cross_parameter[:, None]
                    * first["direction"]
                )
                second_cross = (
                    second["center"]
                    + second_cross_parameter[:, None]
                    * second["direction"]
                )
                corridor = second_cross - first_cross
                corridor_squared = np.sum(corridor * corridor, axis=1)
                position_in_corridor = np.sum(
                    (crossing_midpoints - first_cross) * corridor,
                    axis=1,
                ) / np.maximum(corridor_squared, 1e-8)
                valid = (
                    (position_in_corridor >= -0.18)
                    & (position_in_corridor <= 1.18)
                )
                crossing_length = float(
                    np.minimum(
                        segment_lengths[crossing_indices][valid],
                        2.0 * separation,
                    ).sum()
                )

            crossing_score = float(
                np.clip(
                    crossing_length / max(0.76 * overlap, 1.0),
                    0.0,
                    1.0,
                )
            )

            inside_mean = float(np.mean(inside))
            outside = strip[
                :, (fractions <= -0.08) | (fractions >= 1.08)
            ]
            outside_mean = (
                float(np.mean(outside)) if outside.size else inside_mean
            )
            contrast_score = float(
                np.clip(
                    (
                        abs(inside_mean - outside_mean) - 0.006
                    ) / 0.11,
                    0.0,
                    1.0,
                )
            )

            evidence = (
                0.45 * crossing_score
                + 0.29 * sleeper_score
                + 0.20 * boundary_score
                + 0.06 * contrast_score
            )
            pair["evidence"] = evidence
            pair["score"] = pair["geometry"] * (
                0.30 + 0.70 * evidence
            )
            scored.append(pair)

        if not scored:
            return []

        scored.sort(key=lambda pair: pair["score"], reverse=True)
        best = scored[0]
        if best["geometry"] < 0.031:
            return []
        if best["evidence"] < 0.035 and best["geometry"] < 0.15:
            return []

        selected = []
        score_floor = 0.53 * best["score"]

        for pair in scored:
            if pair["score"] < score_floor:
                break
            if pair["evidence"] < 0.025 and pair["geometry"] < 0.16:
                continue

            midpoint = 0.25 * (
                pair["first_start"]
                + pair["first_end"]
                + pair["second_start"]
                + pair["second_end"]
            )
            conflict = False

            for existing in selected:
                if (
                    pair["first"] in (
                        existing["first"], existing["second"]
                    )
                    or pair["second"] in (
                        existing["first"], existing["second"]
                    )
                ):
                    conflict = True
                    break

                existing_midpoint = 0.25 * (
                    existing["first_start"]
                    + existing["first_end"]
                    + existing["second_start"]
                    + existing["second_end"]
                )
                if float(
                    np.linalg.norm(midpoint - existing_midpoint)
                ) < 0.42 * max(
                    pair["separation"], existing["separation"]
                ):
                    conflict = True
                    break

            if not conflict:
                selected.append(pair)
                if len(selected) >= 4:
                    break

        def convert(points):
            converted = []
            for point in points:
                if point is None or not np.isfinite(point).all():
                    return None
                x = float(
                    np.clip(point[0] * scale_x, 0.0, source_w - 1.0)
                )
                y = float(
                    np.clip(point[1] * scale_y, 0.0, source_h - 1.0)
                )
                if not math.isfinite(x) or not math.isfinite(y):
                    return None
                converted.append([x, y])

            if len(converted) < 2:
                return None
            if math.hypot(
                converted[-1][0] - converted[0][0],
                converted[-1][1] - converted[0][1],
            ) < 1.0:
                return None
            return converted

        result = []

        for pair in selected:
            first_points = [pair["first_start"], pair["first_end"]]
            second_points = [pair["second_start"], pair["second_end"]]

            first_offset = float(
                np.dot(
                    0.5 * (first_points[0] + first_points[1]),
                    pair["normal"],
                )
            )
            second_offset = float(
                np.dot(
                    0.5 * (second_points[0] + second_points[1]),
                    pair["normal"],
                )
            )

            if first_offset <= second_offset:
                left_points, right_points = first_points, second_points
            else:
                left_points, right_points = second_points, first_points

            left_rail = convert(left_points)
            right_rail = convert(right_points)
            if left_rail is not None and right_rail is not None:
                result.append(
                    {
                        "left_rail": left_rail,
                        "right_rail": right_rail,
                    }
                )

        return result

    except Exception:
        return []