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
                output.append([round(x1, 3), round(y1, 3),
                               round(x2, 3), round(y2, 3)])

    def arc_models(contour):
        if contour is None or len(contour) < 18:
            return
        points = contour[:, 0, :].astype(np.float64)
        if len(points) > 180:
            points = points[np.linspace(0, len(points) - 1, 180,
                                        dtype=np.int32)]
        n = len(points)
        if n < 18:
            return
        triples = []
        for fraction in (28, 20, 14, 10, 7):
            gap = max(4, n // fraction)
            for start in np.linspace(0, n - 1, min(14, n), dtype=np.int32):
                triples.append((int(start), int((start + gap) % n),
                                int((start + 2 * gap) % n)))
        rng = np.random.default_rng(712367 + n)
        triples.extend(map(tuple, rng.integers(0, n, size=(52, 3)).tolist()))
        px, py = points[:, 0], points[:, 1]
        models = []
        for i, j, k in triples:
            if len({i, j, k}) < 3:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            x3, y3 = points[k]
            d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1)
                       + x3 * (y1 - y2))
            if abs(d) < 5.0:
                continue
            q1, q2, q3 = x1*x1+y1*y1, x2*x2+y2*y2, x3*x3+y3*y3
            cx = (q1*(y2-y3) + q2*(y3-y1) + q3*(y1-y2)) / d
            cy = (q1*(x3-x2) + q2*(x1-x3) + q3*(x2-x1)) / d
            radius = float(np.hypot(x1-cx, y1-cy))
            if (not np.all(np.isfinite((cx, cy, radius))) or radius < 8.0
                    or radius > .46 * min_dim or cx < -radius
                    or cy < -radius or cx > width - 1 + radius
                    or cy > height - 1 + radius):
                continue
            tolerance = float(np.clip(.9 + .009*radius, 1.05, 1.75))
            residual = np.abs(np.hypot(px-cx, py-cy) - radius)
            keep = residual <= tolerance
            nk = int(np.count_nonzero(keep))
            if nk < max(9, int(.055*np.pi*radius)):
                continue
            ix, iy = px[keep], py[keep]
            a = np.column_stack((2.0*ix, 2.0*iy, np.ones(nk)))
            try:
                solution, _, rank, _ = np.linalg.lstsq(
                    a, ix*ix+iy*iy, rcond=None)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if rank < 3:
                continue
            ncx, ncy, constant = solution
            rr = constant + ncx*ncx + ncy*ncy
            if rr <= 0:
                continue
            nr = float(np.sqrt(rr))
            if (nr < 8.0 or nr > .46*min_dim
                    or np.hypot(ncx-cx, ncy-cy) > max(4.0, .13*radius)
                    or abs(nr-radius) > max(4.0, .13*radius)):
                continue
            residual = np.abs(np.hypot(px-ncx, py-ncy)-nr)
            keep = residual <= tolerance
            nk = int(np.count_nonzero(keep))
            if nk < 9:
                continue
            angles = np.mod(np.arctan2(py[keep]-ncy, px[keep]-ncx),
                            2.0*np.pi)
            bins = np.minimum(71, (angles*(72.0/(2.0*np.pi))).astype(np.int32))
            coverage = len(np.unique(bins))/72.0
            if coverage < .10:
                continue
            score = nk + 45.0*coverage - 8.0*float(np.median(residual[keep]))
            models.append((score, float(ncx), float(ncy), nr))
        models.sort(reverse=True)
        selected = []
        for model in models:
            _, cx, cy, radius = model
            if any(np.hypot(cx-p[1], cy-p[2]) <= max(2.0, .045*max(radius,p[3]))
                   and abs(radius-p[3]) <= max(2.0, .055*max(radius,p[3]))
                   for p in selected):
                continue
            selected.append(model)
            emit(cx, cy, radius, (0.0, .55, 1.1, 1.8, 2.7, 3.7))
            if len(selected) >= 9:
                break

    flat = image.reshape(-1, 3)
    keys = ((flat[:, 0].astype(np.uint32) << 16)
            | (flat[:, 1].astype(np.uint32) << 8)
            | flat[:, 2].astype(np.uint32))
    values, counts = np.unique(keys, return_counts=True)
    frequent = np.flatnonzero(counts >= 100)
    colored = []
    for index in frequent[np.argsort(counts[frequent])[::-1]]:
        value = int(values[index])
        channels = ((value >> 16) & 255, (value >> 8) & 255, value & 255)
        if max(channels) - min(channels) >= 2:
            colored.append(index)
            if len(colored) >= 18:
                break
    for index in colored:
        mask = (keys == int(values[index])).reshape(height, width).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if len(contours) > 60:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:60]
        for contour in contours:
            if len(contour) >= 18 and abs(cv2.contourArea(contour)) >= 35.0:
                arc_models(contour)

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    except cv2.error:
        return output
    planes = (gray, lab[:, :, 1], lab[:, :, 2])
    for plane in planes:
        if float(np.std(plane)) < .7:
            continue
        smooth = cv2.GaussianBlur(plane, (5, 5), .9)
        for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(smooth, 0, 255, mode | cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            for contour in contours[:300]:
                area = abs(float(cv2.contourArea(contour)))
                perimeter = float(cv2.arcLength(contour, True))
                if area < 30.0 or perimeter < 20.0:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                sx, sy = w - 1.0, h - 1.0
                if min(sx, sy) < 8 or max(sx, sy)/max(1.0,min(sx,sy)) > 1.16:
                    continue
                circularity = 4*np.pi*area/(perimeter*perimeter+1e-9)
                if circularity < .68:
                    continue
                output.append([float(x), float(y), float(x+w-1), float(y+h-1)])

    if planes:
        hough_input = cv2.GaussianBlur(gray, (5, 5), 1.0)
        try:
            circles = cv2.HoughCircles(
                hough_input, cv2.HOUGH_GRADIENT, dp=1.3,
                minDist=max(6, min_dim//65), param1=82, param2=10,
                minRadius=3, maxRadius=max(5, int(.32*min_dim)))
        except cv2.error:
            circles = None
        if circles is not None:
            for cx, cy, radius in circles[0][:100]:
                emit(float(cx), float(cy), float(radius), (-1.2, 0.0, 1.2))

    if not output:
        return []
    unique = []
    seen = set()
    for box in output:
        key = tuple(round(float(v), 2) for v in box)
        if key not in seen:
            seen.add(key)
            unique.append([float(v) for v in box])
    unique.sort(key=lambda b: (b[1], b[0], b[3]-b[1]))
    return unique
