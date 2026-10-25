import time
import cv2
import numpy as np


def process_image(img_path: str) -> list[dict[str, object]]:
    started = time.perf_counter()
    timings = dict.fromkeys(
        ("decode", "preprocess", "lsd", "proposal_generation",
         "profile_fitting", "verification", "tracing", "total"), 0.0
    )
    process_image.last_timings = timings
    try:
        t = time.perf_counter()
        src = cv2.imread(img_path, cv2.IMREAD_COLOR)
        timings["decode"] = time.perf_counter() - t
        if src is None or src.size == 0:
            return []
        return _process_frame(src, timings)
    finally:
        timings["total"] = time.perf_counter() - started
        process_image.last_timings = dict(timings)


def _process_frame(src, timings):
    h, w = src.shape[:2]
    if min(h, w) < 32:
        return []
    diagonal = float(np.hypot(w, h))
    minimum_length = diagonal * 0.05
    max_gauge = min(180.0, diagonal * 0.045)
    t = time.perf_counter()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    b, g, r = [c.astype(np.int16) for c in cv2.split(src)]
    vegetation = ((g - r > 9) & (g - b > 5) & (g > r * 1.08)).astype(np.uint8)
    scale = min(1.0, 1280.0 / w, 720.0 / h)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    timings["preprocess"] += time.perf_counter() - t
    t = time.perf_counter()
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD, 1.0, 0.6, 2.0, 22.5, 0.0, 0.7, 1024
    )
    raw = detector.detect(small)[0]
    timings["lsd"] += time.perf_counter() - t
    if raw is None:
        return []
    t = time.perf_counter()
    lines = np.asarray(raw, np.float32).reshape(-1, 4).copy()
    lines *= np.array([w / sw, h / sh, w / sw, h / sh], np.float32)
    p, q = lines[:, :2], lines[:, 2:]
    lengths = np.linalg.norm(q - p, axis=1)
    mids = (p + q) * 0.5
    ids = np.flatnonzero(lengths >= max(12.0, diagonal * 0.0023))
    ids = ids[np.argsort(-lengths[ids], kind="stable")]
    selected, cells = [], {}
    cell_size = max(96.0, diagonal * 0.032)
    for idx in ids:
        key = tuple((mids[idx] // cell_size).astype(int))
        if cells.get(key, 0) >= 16:
            continue
        selected.append(int(idx))
        cells[key] = cells.get(key, 0) + 1
        if len(selected) >= 2200:
            break
    ids = np.asarray(selected, int)
    p, q, lengths, mids = p[ids], q[ids], lengths[ids], mids[ids]
    if len(p) == 0:
        timings["proposal_generation"] += time.perf_counter() - t
        return []
    directions = (q - p) / lengths[:, None]
    flip = (directions[:, 0] < 0) | (
        (np.abs(directions[:, 0]) < 1e-6) & (directions[:, 1] < 0)
    )
    directions[flip] *= -1

    def unit(v):
        return v / max(1e-6, float(np.linalg.norm(v)))

    def normal_of(u):
        return np.array([-u[1], u[0]], np.float32)

    def rotate(u, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]], np.float32)

    def inside(point):
        return 0 <= point[0] <= w - 1 and 0 <= point[1] <= h - 1

    # Index complete native-coordinate segments, including long overlaps.
    index_size = max(96.0, max_gauge * 1.5)
    spatial = {}
    for i in range(len(p)):
        lo = np.floor(np.minimum(p[i], q[i]) / index_size).astype(int)
        hi = np.floor(np.maximum(p[i], q[i]) / index_size).astype(int)
        for cy in range(lo[1], hi[1] + 1):
            for cx in range(lo[0], hi[0] + 1):
                spatial.setdefault((cx, cy), []).append(i)
    proposals = []
    for i in range(len(p)):
        lo = np.floor((np.minimum(p[i], q[i]) - max_gauge) / index_size).astype(int)
        hi = np.floor((np.maximum(p[i], q[i]) + max_gauge) / index_size).astype(int)
        neighbors = set()
        for cy in range(lo[1], hi[1] + 1):
            for cx in range(lo[0], hi[0] + 1):
                neighbors.update(spatial.get((cx, cy), ()))
        js = np.asarray(sorted(j for j in neighbors if j > i), int)
        if not len(js):
            continue
        u, n = directions[i], normal_of(directions[i])
        offsets = mids[js] - mids[i]
        widths = np.abs(offsets @ n)
        good = (
            (np.abs(directions[js] @ u) > 0.992)
            & (widths >= 3.5) & (widths <= max_gauge)
            & (np.abs(offsets @ u) < (lengths[i] + lengths[js]) * 0.5)
        )
        js = js[good]
        if not len(js):
            continue
        a0, a1 = sorted((float(p[i] @ u), float(q[i] @ u)))
        lo_t = np.maximum(a0, np.minimum(p[js] @ u, q[js] @ u))
        hi_t = np.minimum(a1, np.maximum(p[js] @ u, q[js] @ u))
        overlap = hi_t - lo_t
        gauge = np.abs((mids[js] - mids[i]) @ n)
        error = np.abs(directions[js] @ n)
        positions = np.flatnonzero(
            (overlap >= np.maximum(17.0, gauge * 1.25))
            & (error * overlap <= np.maximum(3.0, gauge * 0.30))
        )
        rank = overlap[positions] / (1 + 10 * error[positions])
        positions = positions[np.argsort(-rank, kind="stable")[:5]]
        for k in positions:
            j = js[k]
            v = directions[j].copy()
            if v @ u < 0:
                v = -v
            axis = unit(u + v)
            along = float((lo_t[k] + hi_t[k]) * 0.5)
            ca = mids[i] + u * (along - float(mids[i] @ u))
            cb = mids[j] + v * ((along - float(mids[j] @ u)) / float(v @ u))
            proposals.append((
                float(overlap[k]) / (1 + 10 * float(error[k])),
                (ca + cb) * 0.5, axis, abs(float((cb - ca) @ normal_of(axis)))
            ))
    proposals.sort(key=lambda item: -item[0])
    timings["proposal_generation"] += time.perf_counter() - t

    def correlation(a, b):
        if len(a) < 8:
            return 0.0
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def highpass(a, sigma):
        a = np.asarray(a, np.float32)
        return a - cv2.GaussianBlur(
            a[:, None], (1, 0), sigmaX=0, sigmaY=max(1.0, sigma)
        ).ravel()

    def valid_rows(mx, my, core):
        return (
            (mx[:, core] >= 0).all(axis=1) & (mx[:, core] <= w - 1).all(axis=1)
            & (my[:, core] >= 0).all(axis=1) & (my[:, core] <= h - 1).all(axis=1)
        )

    def remap_patch(mx, my, z, gauge, color):
        mx, my = mx.astype(np.float32), my.astype(np.float32)
        valid = valid_rows(mx, my, np.abs(z) <= gauge * 0.65)
        patch = cv2.remap(
            gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )[valid].astype(np.float32)
        green = None
        if color:
            green = cv2.remap(
                vegetation, mx, my, cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )[valid]
        return patch, z, green

    def sample(center, axis, gauge, extent, outer=1.4, color=False):
        normal = normal_of(axis)
        half = int(extent // 2)
        ts = np.arange(-half, half + 1, dtype=np.float32)
        radius = int(np.ceil(gauge * outer)) + 3
        z = np.arange(-radius, radius + 1, dtype=np.float32)
        mx = center[0] + ts[:, None] * axis[0] + z[None, :] * normal[0]
        my = center[1] + ts[:, None] * axis[1] + z[None, :] * normal[1]
        return remap_patch(mx, my, z, gauge, color)

    def rail_profile(patch, gauge):
        profile = patch.mean(axis=0)
        smooth = cv2.GaussianBlur(
            profile[None, :], (0, 1), sigmaX=max(1.1, gauge * 0.075)
        ).ravel()
        return np.abs(profile - smooth)

    def subpixel(ridge, k):
        if k <= 0 or k >= len(ridge) - 1:
            return float(k)
        a, b, c = float(ridge[k - 1]), float(ridge[k]), float(ridge[k + 1])
        denominator = a - 2 * b + c
        if denominator >= -1e-5:
            return float(k)
        return float(k) + float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))

    def fit_section(center, axis, gauge, extent, search_scale=1.0):
        started = time.perf_counter()
        try:
            patch, z, _ = sample(center, axis, gauge, extent, 0.95)
            if len(patch) < max(12, extent * 0.45):
                return None
            ridge = rail_profile(patch, gauge)
            radius = max(2.0, min(16.0, gauge * 0.22)) * search_scale
            ia = np.flatnonzero(np.abs(z + gauge * 0.5) <= radius)
            ib = np.flatnonzero(np.abs(z - gauge * 0.5) <= radius)
            if not len(ia) or not len(ib):
                return None
            widths = z[ib][None, :] - z[ia][:, None]
            shift = (z[ib][None, :] + z[ia][:, None]) * 0.5
            scores = (
                ridge[ia, None] + ridge[ib][None, :]
                - 1.5 * ((widths - gauge) / max(2.0, gauge * 0.15)) ** 2
                - 0.6 * (shift / max(2.0, gauge * 0.15)) ** 2
            )
            scores[(widths < gauge * 0.76) | (widths > gauge * 1.24)] = -1e6
            a, b = np.unravel_index(np.argmax(scores), scores.shape)
            a, b = int(ia[a]), int(ib[b])
            strength = min(float(ridge[a]), float(ridge[b]))
            if strength < 1.25:
                return None
            za = float(z[0]) + subpixel(ridge, a)
            zb = float(z[0]) + subpixel(ridge, b)
            if not max(3.5, gauge * 0.72) <= zb - za <= min(max_gauge, gauge * 1.28):
                return None
            point = center + normal_of(axis) * ((za + zb) * 0.5)
            return point, float(zb - za), strength
        finally:
            timings["profile_fitting"] += time.perf_counter() - started

    def rail_morphology(patch, z, gauge):
        if len(patch) < 24:
            return False
        longitudinal = cv2.GaussianBlur(
            patch, (1, 0), sigmaX=0, sigmaY=max(1.0, min(3.0, gauge * 0.10))
        )
        profile = longitudinal.mean(axis=0)
        flank = max(2, min(7, int(round(gauge * 0.065))))
        results = []
        for side in (-1, 1):
            indices = np.flatnonzero(
                np.abs(z - side * gauge * 0.5) <= max(2.0, gauge * 0.13)
            )
            indices = indices[(indices >= flank + 1) & (indices < len(z) - flank - 1)]
            best = 0.0
            for k in indices:
                for polarity in (-1.0, 1.0):
                    a = polarity * (longitudinal[:, k] - longitudinal[:, k - flank])
                    b = polarity * (longitudinal[:, k] - longitudinal[:, k + flank])
                    ma, mb = float(a.mean()), float(b.mean())
                    contrast = min(ma, mb)
                    if contrast < 0.85:
                        continue
                    balance = contrast / max(ma, mb, 1e-6)
                    persistent = float(np.mean((a > 0.3) & (b > 0.3)))
                    if balance < 0.19 or persistent < 0.32:
                        continue
                    blocks = np.array_split(np.minimum(a, b), 3)
                    if sum(float(x.mean()) > 0.2 for x in blocks) < 2:
                        continue
                    near = max(1, flank // 2)
                    inner = polarity * (
                        profile[k] - 0.5 * (profile[k - near] + profile[k + near])
                    )
                    if inner < max(0.35, contrast * 0.12):
                        continue
                    best = max(best, contrast * persistent * np.sqrt(balance))
            if best < 0.30:
                return False
            results.append(best)
        return len(results) == 2

    def sleeper_structure(patch, z, green, gauge, shifts, period):
        count = len(patch)
        slope = (shifts[1] - shifts[0]) / max(1.0, gauge * 0.49)
        margin = int(np.ceil(abs(slope) * gauge * 1.35)) + 3
        if count <= 2 * margin + 24:
            return False
        rows = np.arange(margin, count - margin, dtype=np.float32)
        columns = np.flatnonzero(np.abs(z) <= gauge * 0.38)
        if len(columns) < 3:
            return False
        mx = np.broadcast_to(
            np.arange(len(z), dtype=np.float32)[None, :], (len(rows), len(z))
        ).copy()
        my = (rows[:, None] + slope * z[None, :]).astype(np.float32)
        aligned = cv2.remap(
            patch, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        aligned_green = cv2.remap(
            green.astype(np.float32), mx, my, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1
        )
        residual = aligned - cv2.GaussianBlur(
            aligned, (1, 0), sigmaX=0, sigmaY=max(1.2, period * 0.45)
        )
        core = residual[:, columns]
        signal = core.mean(axis=1)
        amplitude = float(np.std(signal))
        if amplitude < 1.7:
            return False
        smooth = cv2.GaussianBlur(
            signal[:, None], (1, 0), sigmaX=0, sigmaY=0.65
        ).ravel()
        ol = (z >= -gauge * 1.35) & (z <= -gauge * 0.88)
        ore = (z >= gauge * 0.88) & (z <= gauge * 1.35)
        if not ol.any() or not ore.any():
            return False
        for mask in (ol, ore):
            outer = residual[:, mask].mean(axis=1)
            if correlation(signal, outer) > 0.72 and np.std(outer) > amplitude * 0.80:
                return False
        for polarity in (-1.0, 1.0):
            s = smooth * polarity
            peaks = np.flatnonzero(
                (s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])
                & (s[1:-1] > max(1.2, amplitude * 0.4))
            ) + 1
            chosen = []
            for k in peaks[np.argsort(-s[peaks], kind="stable")]:
                if all(abs(int(k) - old) >= max(2, period * 0.55) for old in chosen):
                    chosen.append(int(k))
            valid = []
            for k in sorted(chosen):
                response = polarity * core[k]
                level = max(0.75, float(s[k]) * 0.22)
                if np.mean(response > level) < 0.68:
                    continue
                if min(float(v.mean()) for v in np.array_split(response, 3)) < level:
                    continue
                left = float(np.mean(polarity * residual[k, ol]))
                right = float(np.mean(polarity * residual[k, ore]))
                if min(left, right) > float(s[k]) * 0.65:
                    continue
                if float(aligned_green[k, columns].mean()) > 0.55:
                    continue
                valid.append(k)
            if len(valid) < 4:
                continue
            spacing = np.diff(valid)
            regular = (spacing >= period * 0.55) & (spacing <= period * 1.65)
            if np.mean(regular) < 0.65:
                continue
            if valid[-1] - valid[0] < max(gauge * 1.25, period * 3):
                continue
            if rail_morphology(aligned, z, gauge):
                return True
        return False

    def stationary_evidence(patch, z, green, gauge, strict):
        if len(patch) < max(35, gauge * 1.6):
            return 0.0
        core = np.abs(z) <= gauge * 0.38
        green_rows = green[:, core].mean(axis=1)
        if green_rows.mean() > 0.52 and np.mean(green_rows > 0.45) > 0.65:
            return 0.0
        groups = [
            (z >= -0.36 * gauge) & (z <= -0.13 * gauge),
            np.abs(z) <= max(0.6, 0.10 * gauge),
            (z >= 0.13 * gauge) & (z <= 0.36 * gauge)
        ]
        if any(not group.any() for group in groups):
            return 0.0
        signals = [
            highpass(patch[:, group].mean(axis=1), max(2, gauge * 0.30))
            for group in groups
        ]
        limit = min(32, max(1, int(gauge * 0.32)), (len(patch) - 24) // 2)
        if limit < 1:
            return 0.0
        target = signals[1][limit:-limit]
        shifts_available = np.arange(-limit, limit + 1)
        correlations = []
        for signal in (signals[0], signals[2]):
            correlations.append(np.array([
                correlation(target, signal[limit + shift:len(signal) - limit + shift])
                for shift in shifts_available
            ]))
        joint = np.minimum(correlations[0], correlations[1][::-1])
        k = int(np.argmax(joint))
        shifts = [int(shifts_available[k]), -int(shifts_available[k])]
        coherence = float(joint[k])
        independent = [int(np.argmax(c)) for c in correlations]
        independent_shifts = [int(shifts_available[k]) for k in independent]
        if abs(sum(independent_shifts)) <= max(3, limit * 0.35):
            quality = min(float(c[k]) for c, k in zip(correlations, independent))
            if quality > coherence:
                coherence, shifts = quality, independent_shifts
        if coherence < 0.40:
            return 0.0
        aligned = [target]
        for signal, shift in zip((signals[0], signals[2]), shifts):
            aligned.append(signal[limit + shift:len(signal) - limit + shift])
        signal = np.mean(aligned, axis=0)
        amplitude = float(np.std(signal))
        if amplitude < 2.0 or len(signal) < 28:
            return 0.0
        low = max(2, int(gauge * 0.10))
        high = min(max(low + 3, int(gauge * 0.95)), len(signal) // 3)
        if high <= low + 1:
            return 0.0
        lags = np.arange(low, high + 1)
        ac = np.array([correlation(signal[:-k], signal[k:]) for k in lags])
        maxima = np.flatnonzero((ac[1:-1] > ac[:-2]) & (ac[1:-1] >= ac[2:])) + 1
        if not len(maxima):
            return 0.0
        best = int(maxima[np.argmax(ac[maxima])])
        period, periodicity = int(lags[best]), float(ac[best])
        half = max(1, period // 2)
        if periodicity < 0.20 or periodicity - correlation(
            signal[:-half], signal[half:]
        ) < 0.16:
            return 0.0
        ridge = rail_profile(patch, gauge)
        strengths = []
        for side in (-1, 1):
            region = np.abs(z - side * gauge * 0.5) <= max(2, gauge * 0.12)
            strengths.append(float(np.max(ridge[region])))
        if min(strengths) < 1.6:
            return 0.0
        outside_correlations, outside_amplitudes = [], []
        for side in (-1, 1):
            region = (side * z >= gauge * 0.86) & (side * z <= gauge * 1.30)
            if not region.any():
                return 0.0
            outer = highpass(patch[:, region].mean(axis=1), max(2, gauge * 0.30))[limit:-limit]
            outside_correlations.append(abs(correlation(signal, outer)))
            outside_amplitudes.append(float(np.std(outer)))
        if min(outside_correlations) > 0.53:
            return 0.0
        if max(outside_correlations) > 0.82 and min(outside_amplitudes) > amplitude * 0.75:
            return 0.0
        peaks = np.flatnonzero(
            (ridge[1:-1] > ridge[:-2]) & (ridge[1:-1] >= ridge[2:])
            & (ridge[1:-1] > min(strengths) * 0.85)
        ) + 1
        if np.sum(np.abs(z[peaks]) > gauge * 0.72) >= 5:
            return 0.0
        if strict and not sleeper_structure(patch, z, green, gauge, shifts, period):
            return 0.0
        return float(
            coherence * periodicity * min(1.5, amplitude / 5.0)
            * (1 - 0.35 * np.mean(outside_correlations))
        )

    def patch_evidence(patch, z, green, gauge, strict):
        full = stationary_evidence(patch, z, green, gauge, strict)
        if full >= (0.075 if strict else 0.055):
            return full
        count = len(patch)
        window = int(max(64, np.ceil(gauge * 2.25)))
        if count < window + max(20, int(window * 0.35)):
            return full
        starts = np.unique(np.linspace(0, count - window, 4).astype(int))
        valid = []
        for start in starts:
            quality = stationary_evidence(
                patch[start:start + window], z, green[start:start + window], gauge, True
            )
            if quality >= 0.075:
                valid.append((int(start), quality))
        for i, (start, quality) in enumerate(valid):
            for other, other_quality in valid[i + 1:]:
                if other - start >= max(24, window * 0.40):
                    full = max(full, min(quality, other_quality))
        return full

    def evidence(center, axis, gauge, extent, strict=False):
        started = time.perf_counter()
        try:
            patch, z, green = sample(center, axis, gauge, extent, color=True)
            return patch_evidence(patch, z, green, gauge, strict)
        finally:
            timings["verification"] += time.perf_counter() - started

    def extent_for(gauge):
        return int(np.clip(gauge * 4.5, 80, 440))

    def path_patch(path, widths):
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        keep = np.r_[True, np.diff(arc) > 0.05]
        path, widths, arc = path[keep], widths[keep], arc[keep]
        if len(path) < 3 or arc[-1] < 30:
            return None
        reference = float(np.max(widths))
        normalized = np.r_[0.0, np.cumsum(
            np.diff(arc) * reference / np.maximum((widths[1:] + widths[:-1]) * 0.5, 3.5)
        )]
        count = min(1000, int(np.ceil(normalized[-1])) + 1)
        positions = np.linspace(0, normalized[-1], count)
        path = np.column_stack([
            np.interp(positions, normalized, path[:, k]) for k in range(2)
        ]).astype(np.float32)
        widths = np.interp(positions, normalized, widths)
        tangent = np.gradient(path, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6)
        normals = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        radius = int(np.ceil(reference * 1.4)) + 3
        z = np.arange(-radius, radius + 1, dtype=np.float32)
        xy = path[:, None, :] + normals[:, None, :] * (
            z[None, :, None] * (widths / reference)[:, None, None]
        )
        patch, z, green = remap_patch(xy[:, :, 0], xy[:, :, 1], z, reference, True)
        return patch, z, green, reference

    rectification_left = [180]

    def rectified_evidence(center, axis, gauge, extent):
        if rectification_left[0] <= 0:
            return 0.0
        rectification_left[0] -= 1
        offsets = np.linspace(-extent * 0.5, extent * 0.5, 9)
        valid_offsets, centers, widths = [], [], []
        for offset in offsets:
            expected = center + axis * offset
            if not inside(expected):
                continue
            best = None
            for angle in (-0.07, 0.0, 0.07):
                v = rotate(axis, angle)
                fit = fit_section(
                    expected, v, gauge, int(np.clip(gauge * 1.2, 28, 72)), 1.25
                )
                if fit is None or not inside(fit[0]):
                    continue
                score = fit[2] - abs(angle) * 10
                if best is None or score > best[0]:
                    best = (score, fit[0], fit[1])
            if best is not None:
                valid_offsets.append(float(offset))
                centers.append(best[1])
                widths.append(best[2])
        if len(centers) < 5:
            return 0.0
        x = np.asarray(valid_offsets, np.float64)
        if x[-1] - x[0] < extent * 0.60:
            return 0.0
        centers = np.asarray(centers, np.float32)
        widths = np.asarray(widths, np.float32)
        normal = normal_of(axis)
        lateral = (centers - center) @ normal
        scaled = x / max(extent, 1)
        coefficients = np.polyfit(scaled, lateral, 2)
        fitted = np.polyval(coefficients, scaled)
        if np.percentile(np.abs(lateral - fitted), 80) > max(2.5, gauge * 0.09):
            return 0.0
        wc = np.polyfit(scaled, widths, 1)
        if np.percentile(np.abs(widths - np.polyval(wc, scaled)), 80) > max(2.5, gauge * 0.12):
            return 0.0
        ts = np.linspace(x[0], x[-1], max(9, int((x[-1] - x[0]) / 8) + 1))
        path = center + ts[:, None] * axis + np.polyval(
            coefficients, ts / extent
        )[:, None] * normal
        local_widths = np.polyval(wc, ts / extent)
        if np.any(local_widths < gauge * 0.65) or np.any(local_widths > gauge * 1.4):
            return 0.0
        started = time.perf_counter()
        try:
            result = path_patch(path.astype(np.float32), local_widths)
            if result is None:
                return 0.0
            return patch_evidence(*result, True)
        finally:
            timings["verification"] += time.perf_counter() - started

    # Recover both native rail ridges around coarse ballast or center lines.
    t = time.perf_counter()
    local_proposals, local_cells = [], {}
    local_count = 0
    for i in range(len(p)):
        if lengths[i] < max(25.0, diagonal * 0.005):
            continue
        key = tuple((mids[i] // cell_size).astype(int))
        if local_cells.get(key, 0) >= 3:
            continue
        local_cells[key] = local_cells.get(key, 0) + 1
        local_count += 1
        u, n = directions[i], normal_of(directions[i])
        extent = int(np.clip(lengths[i] * 0.55, 32, 80))
        patch, z, _ = sample(mids[i], u, max_gauge, extent, 1.02)
        if len(patch) < extent * 0.6:
            continue
        profile = patch.mean(axis=0)
        smooth = cv2.GaussianBlur(profile[None, :], (0, 1), sigmaX=2.4).ravel()
        ridge = np.abs(profile - smooth)
        peaks = np.flatnonzero(
            (ridge[1:-1] > ridge[:-2]) & (ridge[1:-1] >= ridge[2:]) & (ridge[1:-1] >= 1.7)
        ) + 1
        peaks = peaks[np.argsort(-ridge[peaks], kind="stable")[:18]]
        candidates = []
        for ai in range(len(peaks)):
            for bi in range(ai + 1, len(peaks)):
                a, b = sorted((int(peaks[ai]), int(peaks[bi])))
                width = float(z[b] - z[a])
                shift = float((z[a] + z[b]) * 0.5)
                if not 3.5 <= width <= min(max_gauge, lengths[i] * 1.8):
                    continue
                if abs(shift) > max(12.0, width * 0.65):
                    continue
                score = min(float(ridge[a]), float(ridge[b]))
                score /= 1 + abs(shift) / max(8.0, width)
                candidates.append((score, shift, width))
        candidates.sort(key=lambda item: -item[0])
        chosen = []
        for score, shift, width in candidates:
            if any(
                abs(shift - old[0]) < max(2, width * 0.15)
                and abs(width - old[1]) < max(2, width * 0.20) for old in chosen
            ):
                continue
            chosen.append((shift, width))
            center = mids[i] + n * shift
            if inside(center):
                local_proposals.append((min(float(lengths[i]), 120.0), center, u.copy(), width))
            if len(chosen) >= 4:
                break
        if local_count >= 190:
            break
    ordered, deferred, proposal_cells = [], [], {}
    for proposal in proposals:
        key = tuple((proposal[1] // cell_size).astype(int))
        count = proposal_cells.get(key, 0)
        if count < 4:
            ordered.append(proposal)
            proposal_cells[key] = count + 1
        else:
            deferred.append(proposal)
    primary = ordered + deferred
    ordered = []
    for i in range(max(len(primary), len(local_proposals))):
        if i < len(primary):
            ordered.append(primary[i])
        if i < len(local_proposals):
            ordered.append(local_proposals[i])
    timings["proposal_generation"] += time.perf_counter() - t
    seeds, attempted = [], []
    for support, center, axis, gauge in ordered:
        if not 3.5 <= gauge <= max_gauge:
            continue
        n = normal_of(axis)
        duplicate = False
        for old_center, old_axis, old_width in attempted:
            d = center - old_center
            if (
                abs(float(axis @ old_axis)) > 0.995
                and abs(float(d @ n)) < max(3.0, gauge * 0.25)
                and abs(float(d @ axis)) < max(28.0, gauge * 1.5)
                and abs(gauge - old_width) < max(2.0, gauge * 0.18)
            ):
                duplicate = True
                break
        if duplicate:
            continue
        attempted.append((center, axis, gauge))
        extent = extent_for(gauge)
        fit = fit_section(center, axis, gauge, min(extent, 72), 1.2)
        if fit is not None:
            point, width, strength = fit
            quality = evidence(point, axis, width, extent, True)
            if quality < 0.075 and strength >= 2 and support >= max(23, width * 0.75):
                quality = rectified_evidence(point, axis, width, extent)
                if quality >= 0.075:
                    confirmed = False
                    for sign in (-1, 1):
                        c = point + axis * sign * max(28, width)
                        if not inside(c):
                            continue
                        f = fit_section(c, axis, width, min(extent, 72))
                        if f is None:
                            continue
                        confirmation = evidence(f[0], axis, f[1], extent_for(f[1]), True)
                        if confirmation < 0.075:
                            confirmation = rectified_evidence(f[0], axis, f[1], extent_for(f[1]))
                        if confirmation >= 0.075:
                            confirmed = True
                            break
                    if not confirmed:
                        quality = 0.0
            if quality >= 0.075:
                seeds.append((quality, point, axis, width))
        if len(attempted) >= 760 or len(seeds) >= 32:
            break
    if not seeds:
        return []
    seeds.sort(key=lambda item: -item[0])
    rectification_left[0] = max(rectification_left[0], 70)
    recovery_left = [30]

    def bridge(a, ua, wa, b, ub, wb):
        gap = float(np.linalg.norm(b - a))
        count = max(3, int(np.ceil(gap / 18)) + 1)
        nodes = []
        for fraction in np.linspace(0, 1, count, dtype=np.float32)[1:-1]:
            f2, f3 = fraction * fraction, fraction ** 3
            point = (
                (2 * f3 - 3 * f2 + 1) * a + (f3 - 2 * f2 + fraction) * gap * ua
                + (-2 * f3 + 3 * f2) * b + (f3 - f2) * gap * ub
            )
            if not inside(point):
                return []
            nodes.append((point, float(wa * (1 - fraction) + wb * fraction)))
        return nodes

    def recover(current, axis, width):
        if recovery_left[0] <= 0:
            return None
        recovery_left[0] -= 1
        maximum = min(diagonal * 0.08, max(150, width * 6))
        distances = np.linspace(max(30, width * 1.1), maximum, 5)
        for distance in distances:
            candidates = []
            for angle in (-0.14, -0.07, 0.0, 0.07, 0.14):
                heading, v = rotate(axis, angle * 0.5), rotate(axis, angle)
                expected = current + heading * distance
                if not inside(expected):
                    continue
                for factor in (0.90, 1.0, 1.10):
                    fit = fit_section(
                        expected, v, width * factor, int(np.clip(width * 1.5, 28, 80)), 1.3
                    )
                    if fit is None or not inside(fit[0]) or fit[2] < 2:
                        continue
                    score = fit[2] - 10 * abs(angle) - 4 * abs(factor - 1)
                    candidates.append((score, fit[0], fit[1], v))
            candidates.sort(key=lambda item: -item[0])
            for _, point, next_width, v in candidates[:3]:
                quality = evidence(point, v, next_width, extent_for(next_width), True)
                if quality < 0.075:
                    quality = rectified_evidence(point, v, next_width, extent_for(next_width))
                if quality >= 0.075:
                    return quality, point, next_width, v
        return None

    def trace(center, axis, gauge, sign):
        u = axis * sign
        current, width = center.copy(), float(gauge)
        nodes, last_good, unverified = [], -1, 0.0
        trusted_point = current.copy()
        trusted_u, trusted_width = u.copy(), width
        recoveries = 0
        for iteration in range(int(diagonal / 12) + 2):
            step = float(np.clip(width * 0.75, 12, 40))
            expected = current + u * step
            if not inside(expected):
                limits = []
                for k, maximum in ((0, w - 1), (1, h - 1)):
                    if abs(float(u[k])) > 1e-6:
                        boundary = maximum if u[k] > 0 else 0
                        limits.append((boundary - float(current[k])) / float(u[k]))
                positive = [x for x in limits if x >= 0]
                if positive and unverified < max(28, width):
                    distance = min(positive)
                    if 0.5 < distance <= step:
                        nodes.append((current + u * distance, width))
                        last_good = len(nodes) - 1
                break
            best = None
            for angle in (-0.08, -0.04, 0.0, 0.04, 0.08):
                v = rotate(u, angle)
                fit = fit_section(expected, v, width, int(np.clip(width * 1.6, 28, 84)))
                if fit is None or not inside(fit[0]):
                    continue
                score = fit[2] - abs(angle) * 8
                if best is None or score > best[0]:
                    best = (score, fit[0], fit[1], v)
            unverified += step
            if best is None:
                current = expected
                nodes.append((current.copy(), width))
            else:
                _, point, next_width, v = best
                move = unit(point - current)
                u = unit(0.6 * v + 0.4 * move)
                current = point
                width = float(np.clip(0.6 * width + 0.4 * next_width, 3.5, max_gauge))
                nodes.append((point.copy(), width))
                if iteration % 2 == 1:
                    extent = extent_for(width)
                    quality = evidence(point, u, width, extent)
                    if quality < 0.055 and unverified > max(48, width * 2):
                        quality = rectified_evidence(point, u, width, extent)
                    if quality >= 0.055:
                        last_good = len(nodes) - 1
                        unverified = 0.0
                        trusted_point = point.copy()
                        trusted_u, trusted_width = u.copy(), width
            if unverified > max(72, trusted_width * 2.5):
                if recoveries >= 4:
                    break
                recovered = recover(trusted_point, trusted_u, trusted_width)
                if recovered is None:
                    break
                _, point, next_width, v = recovered
                nodes = nodes[:last_good + 1]
                nodes.extend(bridge(
                    trusted_point, trusted_u, trusted_width, point, v, next_width
                ))
                nodes.append((point.copy(), next_width))
                current, u, width = point.copy(), v, float(next_width)
                trusted_point, trusted_u, trusted_width = current.copy(), u.copy(), width
                last_good, unverified = len(nodes) - 1, 0.0
                recoveries += 1
        if last_good >= 0:
            return nodes[:min(len(nodes), last_good + 2)]
        return nodes[:1]

    def distance_to_path(point, path):
        a, d = path[:-1], np.diff(path, axis=0)
        fractions = np.sum((point - a) * d, axis=1)
        fractions /= np.maximum(np.sum(d * d, axis=1), 1e-6)
        foot = a + np.clip(fractions, 0, 1)[:, None] * d
        return float(np.min(np.linalg.norm(foot - point, axis=1)))

    def path_length(path):
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())

    def endpoint_direction(path, at_end):
        if at_end:
            return unit(path[-1] - path[max(0, len(path) - 4)])
        return unit(path[min(3, len(path) - 1)] - path[0])

    tracing_started = time.perf_counter()
    tracks = []
    for quality, center, axis, gauge in seeds:
        if any(distance_to_path(center, path) < max(5, gauge * 0.4) for path, _ in tracks):
            continue
        nodes = (
            trace(center, axis, gauge, -1)[::-1]
            + [(center, gauge)] + trace(center, axis, gauge, 1)
        )
        if len(nodes) < 3:
            continue
        path = np.asarray([node[0] for node in nodes], np.float32)
        widths = np.asarray([node[1] for node in nodes], np.float32)
        keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 2]
        path, widths = path[keep], widths[keep]
        if len(path) < 3 or path_length(path) < max(35, minimum_length * 0.2):
            continue
        tracks.append((path, widths))
        if len(tracks) >= 20:
            break
    for _ in range(20):
        best = None
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                for reverse_i in (False, True):
                    pa, wa = tracks[i]
                    if reverse_i:
                        pa, wa = pa[::-1], wa[::-1]
                    ua = endpoint_direction(pa, True)
                    for reverse_j in (False, True):
                        pb, wb = tracks[j]
                        if reverse_j:
                            pb, wb = pb[::-1], wb[::-1]
                        delta = pb[0] - pa[-1]
                        gap = float(np.linalg.norm(delta))
                        mean_width = float((wa[-1] + wb[0]) * 0.5)
                        if not 3 < gap < min(
                            diagonal * 0.08, max(minimum_length * 0.8, mean_width * 7)
                        ):
                            continue
                        ub = endpoint_direction(pb, False)
                        direction = delta / gap
                        if (
                            float(ua @ direction) < 0.975
                            or float(ub @ direction) < 0.975
                            or float(ua @ ub) < 0.965
                            or abs(float(wa[-1] - wb[0])) > max(3, mean_width * 0.32)
                        ):
                            continue
                        lateral = max(
                            abs(float(delta @ normal_of(ua))),
                            abs(float(delta @ normal_of(ub)))
                        )
                        if lateral > max(5, mean_width * 0.6):
                            continue
                        score = gap / max(mean_width, 4)
                        if best is None or score < best[0]:
                            best = (score, i, j, pa, wa, pb, wb, ua, ub)
        if best is None:
            break
        _, i, j, pa, wa, pb, wb, ua, ub = best
        middle = bridge(pa[-1], ua, wa[-1], pb[0], ub, wb[0])
        if middle:
            mp = np.asarray([node[0] for node in middle], np.float32)
            mw = np.asarray([node[1] for node in middle], np.float32)
            tracks[i] = np.vstack((pa, mp, pb)), np.concatenate((wa, mw, wb))
        else:
            tracks[i] = np.vstack((pa, pb)), np.concatenate((wa, wb))
        tracks.pop(j)
    tracks.sort(key=lambda item: -path_length(item[0]))
    output, accepted_paths = [], []
    rectification_left[0] = max(rectification_left[0], 48)
    for path, widths in tracks:
        if len(path) > 4:
            path = path.copy()
            path[1:-1] = path[:-2] * 0.2 + path[1:-1] * 0.6 + path[2:] * 0.2
            widths = widths.copy()
            widths[1:-1] = widths[:-2] * 0.2 + widths[1:-1] * 0.6 + widths[2:] * 0.2
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        length = float(arc[-1])
        if length < minimum_length:
            continue
        if any(
            np.mean([
                distance_to_path(point, old) < max(5, float(width) * 0.45)
                for point, width in zip(path, widths)
            ]) > 0.70 for old in accepted_paths
        ):
            continue
        tangents = np.empty_like(path)
        tangents[0], tangents[-1] = path[1] - path[0], path[-1] - path[-2]
        tangents[1:-1] = path[2:] - path[:-2]
        tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
        positions = np.linspace(
            length * 0.08, length * 0.92, min(11, max(3, int(length / 90) + 1))
        )
        verified, checked = [], set()
        for position in positions:
            k = int(np.argmin(np.abs(arc - position)))
            if k in checked:
                continue
            checked.add(k)
            width = float(widths[k])
            extent = extent_for(width)
            quality = evidence(path[k], tangents[k], width, extent, True)
            if quality < 0.075:
                # Verify the traced curve and changing gauge on native pixels.
                start = max(0.0, float(arc[k]) - extent * 0.5)
                stop = min(length, float(arc[k]) + extent * 0.5)
                ts = np.linspace(start, stop, max(9, int((stop - start) / 8) + 1))
                local_path = np.column_stack([
                    np.interp(ts, arc, path[:, d]) for d in range(2)
                ]).astype(np.float32)
                local_widths = np.interp(ts, arc, widths)
                started = time.perf_counter()
                result = path_patch(local_path, local_widths)
                if result is not None:
                    quality = patch_evidence(*result, True)
                timings["verification"] += time.perf_counter() - started
            if quality < 0.075:
                quality = rectified_evidence(path[k], tangents[k], width, extent)
            if quality >= 0.075:
                verified.append(float(arc[k]))
        if len(verified) < 2:
            continue
        verified_span = max(verified) - min(verified)
        if verified_span < min(
            minimum_length * 0.35, max(30, float(np.median(widths)) * 1.5)
        ):
            continue
        if length < 7 * float(np.median(widths)):
            if len(verified) < 3 or verified_span < length * 0.45:
                continue
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        left = path - normals * widths[:, None] * 0.5
        right = path + normals * widths[:, None] * 0.5
        for rail in (left, right):
            rail[:, 0] = np.clip(rail[:, 0], 0, w - 1)
            rail[:, 1] = np.clip(rail[:, 1], 0, h - 1)
        shorter = min(path_length(left), path_length(right))
        span = float(np.linalg.norm(np.ptp(np.vstack((left, right)), axis=0)))
        if shorter < minimum_length or span < minimum_length:
            continue
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            continue
        if left[:, 0].mean() > right[:, 0].mean():
            left, right = right, left
        accepted_paths.append(path)
        output.append({
            "left_rail": left.astype(float).tolist(),
            "right_rail": right.astype(float).tolist(),
            "confidence": float(len(verified) / len(checked)),
            "confidence_method": "verified_patch_fraction_v1"
        })
        if len(output) >= 6:
            break
    timings["tracing"] += time.perf_counter() - tracing_started
    return output