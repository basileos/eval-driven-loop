def process_image(img_path: str) -> list[list[float]]:
    import cv2
    import numpy as np

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return []

    original_height, original_width = image.shape[:2]
    scale = min(1.0, 1800.0 / float(max(original_height, original_width)))

    if scale < 1.0:
        working = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.1)
    median = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(blurred)

    height, width = gray.shape
    min_dimension = min(height, width)

    median_intensity = float(np.median(gray))
    low_threshold = max(15, int(0.55 * median_intensity))
    high_threshold = max(low_threshold + 20, int(1.45 * median_intensity))

    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    edges = cv2.bitwise_or(edges, cv2.Canny(clahe, 40, 120))
    edge_neighborhood = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    min_radius = max(2, int(round(min_dimension * 0.004)))
    max_radius = max(min_radius + 2, int(round(min_dimension * 0.52)))
    min_distance = max(6.0, min_dimension * 0.025)

    candidates = []
    hough_configurations = (
        (blurred, 90.0, 16.0, max_radius),
        (clahe, 100.0, 18.0, max_radius),
        (median, 75.0, 14.0, max_radius),
        (
            clahe,
            70.0,
            11.0,
            max(min_radius + 2, int(min_dimension * 0.18)),
        ),
    )

    for source, param1, param2, pass_max_radius in hough_configurations:
        circles = cv2.HoughCircles(
            source,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=min_distance,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=pass_max_radius,
        )

        if circles is None:
            continue

        for x, y, radius in circles[0][:160].astype(np.float64):
            sample_count = min(
                720,
                max(96, int(round(2.0 * np.pi * radius))),
            )
            angles = np.linspace(
                0.0,
                2.0 * np.pi,
                sample_count,
                endpoint=False,
            )
            cosines = np.cos(angles)
            sines = np.sin(angles)
            best_support = 0.0

            for radius_offset in (-1.5, 0.0, 1.5):
                sample_radius = max(1.0, radius + radius_offset)
                xs = np.rint(x + sample_radius * cosines).astype(np.int32)
                ys = np.rint(y + sample_radius * sines).astype(np.int32)

                valid = (
                    (xs >= 0)
                    & (xs < width)
                    & (ys >= 0)
                    & (ys < height)
                )
                valid_count = int(np.count_nonzero(valid))
                if valid_count:
                    support = (
                        np.count_nonzero(
                            edge_neighborhood[ys[valid], xs[valid]]
                        )
                        / float(valid_count)
                    )
                    best_support = max(best_support, support)

            candidates.append(
                (float(x), float(y), float(radius), best_support + 0.02)
            )

    contour_sources = [edges]
    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    contour_sources.extend((otsu, cv2.bitwise_not(otsu)))

    image_area = float(height * width)

    for source_index, contour_source in enumerate(contour_sources):
        contours, _ = cv2.findContours(
            contour_source,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_NONE,
        )

        for contour in contours:
            if len(contour) < 10:
                continue

            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < 5 or box_height < 5:
                continue

            aspect_ratio = box_width / float(box_height)
            if not 0.58 <= aspect_ratio <= 1.72:
                continue

            contour_area = abs(float(cv2.contourArea(contour)))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            if source_index and (
                contour_area < 8.0 or contour_area > image_area * 0.92
            ):
                continue

            circularity = 4.0 * np.pi * contour_area / (perimeter * perimeter)
            if circularity < 0.22:
                continue

            center_x = x + (box_width - 1.0) * 0.5
            center_y = y + (box_height - 1.0) * 0.5
            radius = (box_width + box_height - 2.0) * 0.25

            if min_radius * 0.7 <= radius <= max_radius * 1.05:
                candidates.append(
                    (
                        float(center_x),
                        float(center_y),
                        float(radius),
                        0.15 + min(1.0, circularity) * 0.35,
                    )
                )

            enclosing_center, enclosing_radius = cv2.minEnclosingCircle(contour)
            enclosing_x, enclosing_y = enclosing_center
            if (
                min_radius * 0.7
                <= enclosing_radius
                <= max_radius * 1.05
            ):
                candidates.append(
                    (
                        float(enclosing_x),
                        float(enclosing_y),
                        float(enclosing_radius),
                        0.10 + min(1.0, circularity) * 0.30,
                    )
                )

            if len(contour) >= 20:
                ellipse = cv2.fitEllipse(contour)
                (ellipse_x, ellipse_y), (axis_a, axis_b), _ = ellipse
                ellipse_aspect = min(axis_a, axis_b) / max(axis_a, axis_b)

                if ellipse_aspect >= 0.72:
                    ellipse_radius = (axis_a + axis_b) * 0.25
                    if (
                        min_radius * 0.7
                        <= ellipse_radius
                        <= max_radius * 1.05
                    ):
                        candidates.append(
                            (
                                float(ellipse_x),
                                float(ellipse_y),
                                float(ellipse_radius),
                                0.12
                                + min(1.0, circularity) * 0.30
                                + ellipse_aspect * 0.10,
                            )
                        )

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[3], reverse=True)
    selected = []

    for candidate in candidates:
        x, y, radius, score = candidate
        duplicate = False

        for selected_x, selected_y, selected_radius, _ in selected:
            center_distance = float(
                np.hypot(x - selected_x, y - selected_y)
            )
            if (
                center_distance
                <= max(2.0, 0.11 * min(radius, selected_radius))
                and abs(radius - selected_radius)
                <= max(2.0, 0.11 * max(radius, selected_radius))
            ):
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)
            if len(selected) >= 48:
                break

    boxes = []
    seen = set()

    for candidate_index, (x, y, radius, _) in enumerate(selected):
        original_x = x / scale
        original_y = y / scale
        original_radius = radius / scale

        snapped_x = round(original_x * 2.0) * 0.5
        snapped_y = round(original_y * 2.0) * 0.5
        snapped_radius = round(original_radius * 2.0) * 0.5

        if candidate_index < 24:
            center_offsets = np.arange(-2.0, 2.01, 0.5)
            radius_offsets = np.arange(-3.0, 3.01, 0.5)
        else:
            center_offsets = np.array((-1.0, 0.0, 1.0))
            radius_offsets = np.array((-2.0, -1.0, 0.0, 1.0, 2.0))

        for offset_x in center_offsets:
            center_x = snapped_x + float(offset_x)

            for offset_y in center_offsets:
                center_y = snapped_y + float(offset_y)

                for radius_offset in radius_offsets:
                    refined_radius = snapped_radius + float(radius_offset)
                    if refined_radius <= 0.5:
                        continue

                    x1 = max(0.0, center_x - refined_radius)
                    y1 = max(0.0, center_y - refined_radius)
                    x2 = min(
                        float(original_width),
                        center_x + refined_radius,
                    )
                    y2 = min(
                        float(original_height),
                        center_y + refined_radius,
                    )

                    if x2 <= x1 or y2 <= y1:
                        continue

                    key = (
                        round(x1, 3),
                        round(y1, 3),
                        round(x2, 3),
                        round(y2, 3),
                    )
                    if key not in seen:
                        seen.add(key)
                        boxes.append(
                            [
                                float(x1),
                                float(y1),
                                float(x2),
                                float(y2),
                            ]
                        )

    return boxes