from collections import Counter
import time

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# Exclusive stages: nested work (e.g. fitting inside tracing) is charged to the
# innermost stage only, so the stages sum to "total".
STAGES = ("decode", "preprocess", "lsd", "proposal_generation", "seed_loop",
          "profile_fitting", "verification", "trace", "total")

# Detection runs on the frame resized by this factor; outputs are mapped back
# to input pixels. Absolute pixel limits (min gauge, strip extents) do not scale.
WORK_SCALE = 1.0

# Search budgets: speed/recall trade-offs, tuned with evaluator.py and the OOD
# set (see SPEC_candidate_fast_speed.md). 960x540 LSD lost val IoU.
LSD_SIZE = (1280, 720)       # LSD runs on the frame downscaled to fit this box
# Verified seeds per frame. On REC_0023 drone footage tracks verify late:
# frames with tracks (of 125) at budget 60/100/140/180/220/280 were
# 63/76/86/89/90/90 (unbounded original: 91). Val/OOD alone suggested 60.
SEED_BUDGET = 220
# A full-window verification failing at one of these gates is not retried on
# shorter sub-windows: across 81 frames that never rescued an accepted track.
NO_RETRY_GATES = frozenset({"length", "green", "ridge", "coherence"})

# g > 1.08 * r and g - r > 9  <=>  g > _GREEN_OVER_RED[r]  (exact for uint8).
_GREEN_OVER_RED = (np.minimum(256, np.maximum(
    np.arange(256) + 10, np.floor(1.08 * np.arange(256)) + 1
)) - 1).astype(np.uint8)


class _StageClock:
    def __init__(self, timings):
        self.timings = timings
        self.stack = []
        self.mark = time.perf_counter()

    def push(self, name):
        self._charge()
        self.stack.append(name)

    def pop(self):
        self._charge()
        self.stack.pop()

    def _charge(self):
        now = time.perf_counter()
        if self.stack:
            self.timings[self.stack[-1]] += now - self.mark
        self.mark = now


def process_image(img_path: str) -> list[dict[str, object]]:
    return _run(process_image, lambda: cv2.imread(img_path, cv2.IMREAD_COLOR))


def process_frame(frame, reference_shape=None) -> list[dict[str, object]]:
    """Detect on decoded BGR pixels; avoids the video path's file round trip.

    reference_shape is the full frame's shape when `frame` is a crop of it
    (streaming ROI checks): size-relative limits and the LSD scale then follow
    the full frame, so a crop is searched exactly like that region of a full
    search. Coordinates stay relative to the crop.
    """
    return _run(process_frame, lambda: frame, reference_shape)


def _run(owner, load, reference_shape=None):
    started = time.perf_counter()
    timings = dict.fromkeys(STAGES, 0.0)
    counts = Counter()
    owner.last_timings, owner.last_counts = timings, counts
    clock = _StageClock(timings)
    try:
        clock.push("decode")
        image = load()
        clock.pop()
        if image is None or image.size == 0:
            return []
        reference = tuple(reference_shape[:2]) if reference_shape else image.shape[:2]
        if WORK_SCALE == 1.0:
            return _detect(image, reference, clock, counts)
        clock.push("preprocess")
        native = image.shape[:2]
        image = cv2.resize(image, None, fx=WORK_SCALE, fy=WORK_SCALE,
                           interpolation=cv2.INTER_AREA)
        clock.pop()
        if min(image.shape[:2]) < 1:
            return []
        tracks = _detect(image, [v * WORK_SCALE for v in reference], clock, counts)
        # Pixel-centre mapping from the resized grid back to input pixels.
        ratio = np.array([native[1] / image.shape[1], native[0] / image.shape[0]])
        for track in tracks:
            for key in ("left_rail", "right_rail"):
                track[key] = ((np.asarray(track[key]) + 0.5) * ratio - 0.5).tolist()
        return tracks
    finally:
        timings["total"] = time.perf_counter() - started
        owner.last_timings = dict(timings)
        owner.last_counts = dict(counts)


def _detect(image, reference, clock, counts):
    height, width = image.shape[:2]
    if min(height, width) < 32:
        return []
    diagonal = float(np.hypot(reference[1], reference[0]))
    minimum_length = 0.05 * diagonal
    maximum_gauge = min(180.0, 0.045 * diagonal)

    def unit(v):
        return v / max(float(np.linalg.norm(v)), 1e-6)

    def normal(v):
        return np.array([-v[1], v[0]], np.float32)

    def rotate(v, angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([c * v[0] - s * v[1],
                         s * v[0] + c * v[1]], np.float32)

    def inside(p):
        return 0 <= p[0] <= width - 1 and 0 <= p[1] <= height - 1

    def arc_length(path):
        return np.r_[0.0, np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1)
        )]

    def corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

    def sliding_corr(target, signal):
        """corr(target, window) for every len(target) window of signal."""
        windows = sliding_window_view(signal, len(target))
        windows = windows - windows.mean(axis=1, keepdims=True)
        target = target - target.mean()
        return windows @ target / (
            np.linalg.norm(windows, axis=1) * np.linalg.norm(target) + 1e-6
        )

    def autocorr(signal, lags):
        """corr(signal[:-k], signal[k:]) for every lag k, from running sums."""
        s = signal.astype(np.float64)
        n = len(s)
        c1 = np.r_[0.0, np.cumsum(s)]
        c2 = np.r_[0.0, np.cumsum(s * s)]
        products = np.correlate(s, s, "full")[n - 1:][lags]
        m = n - lags
        sx, sy = c1[m], c1[n] - c1[lags]
        vx = np.maximum(c2[m] - sx * sx / m, 0)
        vy = np.maximum(c2[n] - c2[lags] - sy * sy / m, 0)
        return (products - sx * sy / m) / (np.sqrt(vx * vy) + 1e-6)

    def highpass(a, sigma):
        a = np.asarray(a, np.float32)
        return a - cv2.GaussianBlur(
            a[:, None], (1, 0), sigmaX=0, sigmaY=max(1.0, sigma)
        ).ravel()

    clock.push("preprocess")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    vegetation = cv2.bitwise_and(
        cv2.compare(g, cv2.LUT(r, _GREEN_OVER_RED), cv2.CMP_GT),
        cv2.compare(cv2.subtract(g, b), 5, cv2.CMP_GT),
    ) & 1
    scale = min(1.0, LSD_SIZE[0] / reference[1], LSD_SIZE[1] / reference[0])
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    clock.pop()

    clock.push("lsd")
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD, 1.0, 0.6, 2.0, 22.5, 0.0, 0.7, 1024
    )
    raw = detector.detect(small)[0]
    clock.pop()
    if raw is None:
        return []

    def remap(xs, ys, across, gauge, color=False):
        """Sample rows whose core lies inside the image; xs/ys are (rows, across)."""
        core = np.flatnonzero(np.abs(across) <= gauge * 0.65)
        if len(core):
            # The core is a contiguous run of columns; test it before the
            # float32 cast so boundary rows match the full-precision check.
            cx = xs[:, core[0]:core[-1] + 1]
            cy = ys[:, core[0]:core[-1] + 1]
            valid = (
                (cx.min(axis=1) >= 0) & (cx.max(axis=1) <= width - 1)
                & (cy.min(axis=1) >= 0) & (cy.max(axis=1) <= height - 1)
            )
            if not valid.any():
                empty = np.zeros((0, len(across)), np.float32)
                return empty, across, empty.astype(np.uint8) if color else None
            if not valid.all():
                xs, ys = xs[valid], ys[valid]
        mx = np.ascontiguousarray(xs, np.float32)
        my = np.ascontiguousarray(ys, np.float32)
        patch = cv2.remap(
            gray, mx, my, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.float32)
        green = None
        if color:
            green = cv2.remap(
                vegetation, mx, my, cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=1
            )
        return patch, across, green

    def strip(center, direction, gauge, extent, outer=1.4, color=False):
        along = np.arange(-int(extent / 2), int(extent / 2) + 1,
                          dtype=np.float32)[:, None]
        radius = int(np.ceil(gauge * outer)) + 3
        across = np.arange(-radius, radius + 1, dtype=np.float32)
        n = normal(direction)
        # Same per-element operations as center + along*u + across*n.
        xs = (center[0] + along * direction[0]) + across * n[0]
        ys = (center[1] + along * direction[1]) + across * n[1]
        return remap(xs, ys, across, gauge, color)

    def ridge_profile(patch, gauge):
        profile = patch.mean(axis=0)
        baseline = cv2.GaussianBlur(
            profile[None], (0, 1), sigmaX=max(1.1, gauge * 0.075)
        ).ravel()
        return np.abs(profile - baseline)

    def subpixel(values, k):
        if not 0 < k < len(values) - 1:
            return float(k)
        a, b, c = map(float, values[k - 1:k + 2])
        denominator = a - 2 * b + c
        if denominator >= -1e-5:
            return float(k)
        return k + float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))

    def morphology_ok(patch, across, gauge, threshold=0.30):
        """Both sides show a narrow, persistent rail ridge scoring >= threshold.

        Polarity -1 is the exact negation of polarity +1, so each side's
        differences are computed once and both signs are tested from them.
        """
        if len(patch) < 16:
            return False
        smooth = cv2.GaussianBlur(
            patch, (1, 0), sigmaX=0,
            sigmaY=max(1.0, min(3.0, gauge * 0.10))
        )
        flank = max(2, min(7, round(gauge * 0.065)))
        near = max(1, flank // 2)
        rows = len(smooth)
        # Row blocks exactly as np.array_split(..., 3).
        size, extra = divmod(rows, 3)
        edges = np.cumsum([0] + [size + (k < extra) for k in range(3)])
        for side in (-1, 1):
            ids = np.flatnonzero(
                np.abs(across - side * gauge / 2) <= max(2, gauge * 0.12)
            )
            ids = ids[(ids >= flank) & (ids < len(across) - flank)]
            if not len(ids):
                return False
            center = smooth[:, ids]
            a = center - smooth[:, ids - flank]
            b = center - smooth[:, ids + flank]
            ma, mb = a.mean(axis=0), b.mean(axis=0)
            inner = (
                center.mean(axis=0)
                - (smooth[:, ids - near].mean(axis=0)
                   + smooth[:, ids + near].mean(axis=0)) / 2
            )
            low, high = np.minimum(a, b), np.maximum(a, b)
            passed = False
            for polarity in (-1, 1):
                pa, pb = polarity * ma, polarity * mb
                contrast = np.minimum(pa, pb)
                balance = contrast / np.maximum(np.maximum(pa, pb), 1e-6)
                if polarity > 0:
                    persistence = np.count_nonzero(
                        (a > 0.3) & (b > 0.3), axis=0
                    ) / rows
                    ridge = low
                else:
                    persistence = np.count_nonzero(
                        (a < -0.3) & (b < -0.3), axis=0
                    ) / rows
                    ridge = high
                blocks = np.stack([
                    polarity * ridge[edges[k]:edges[k + 1]].mean(axis=0)
                    for k in range(3)
                ])
                valid = (
                    (contrast >= 0.85) & (balance >= 0.19)
                    & (persistence >= 0.32)
                    & ((blocks > 0.2).sum(axis=0) >= 2)
                    & (polarity * inner >= np.maximum(0.35, contrast * 0.12))
                )
                scores = contrast * persistence * np.sqrt(np.maximum(balance, 0))
                if (scores[valid] >= threshold).any():
                    passed = True
                    break
            if not passed:
                return False
        return True

    def fit_section(center, direction, gauge, extent, search=1.0,
                    allow_single=False):
        counts["fit_section"] += 1
        clock.push("profile_fitting")
        try:
            patch, across, green = strip(
                center, direction, gauge, extent, 1.05, True
            )
            if len(patch) < max(14, extent * 0.48):
                return None
            core = np.abs(across) <= gauge * 0.32
            if core.any() and green[:, core].mean() > 0.76:
                return None
            ridge = ridge_profile(patch, gauge)
            radius = max(2.0, min(17.0, gauge * 0.22)) * search
            ia = np.flatnonzero(np.abs(across + gauge / 2) <= radius)
            ib = np.flatnonzero(np.abs(across - gauge / 2) <= radius)
            if not len(ia) or not len(ib):
                return None
            gauges = across[ib][None] - across[ia][:, None]
            shifts = (across[ib][None] + across[ia][:, None]) / 2
            score = (
                ridge[ia, None] + ridge[ib][None]
                - 1.2 * ((gauges - gauge) / max(2, gauge * 0.15)) ** 2
                - 0.5 * (shifts / max(2, gauge * 0.18)) ** 2
            )
            score[(gauges < gauge * 0.78) | (gauges > gauge * 1.22)] = -1e6
            ai, bi = np.unravel_index(np.argmax(score), score.shape)
            ai, bi = int(ia[ai]), int(ib[bi])
            strengths = (float(ridge[ai]), float(ridge[bi]))
            paired = min(strengths) >= 1.3
            if not paired and (not allow_single or max(strengths) < 3.0):
                return None
            za = float(across[0]) + subpixel(ridge, ai)
            zb = float(across[0]) + subpixel(ridge, bi)
            if not paired:
                if strengths[0] > strengths[1]:
                    zb = za + gauge
                else:
                    za = zb - gauge
            fitted_gauge = zb - za
            if not 3.5 <= fitted_gauge <= maximum_gauge:
                return None
            point = center + normal(direction) * ((za + zb) / 2)
            if not inside(point):
                return None
            # Persistence is evaluated on original pixels, independently of
            # the section's averaged cross-track profile.
            flank = max(2, min(7, round(gauge * 0.065)))
            persistence = []
            for k in (ai, bi):
                if k < flank or k + flank >= len(across):
                    persistence.append(0.0)
                    continue
                contrast = patch[:, k] - (
                    patch[:, k - flank] + patch[:, k + flank]
                ) / 2
                polarity = 1 if contrast.mean() >= 0 else -1
                persistence.append(float(np.mean(polarity * contrast > 0.4)))
            if paired and min(persistence) < 0.30:
                return None
            strength = min(strengths) if paired else max(strengths) * 0.25
            return point, float(fitted_gauge), strength, paired
        finally:
            clock.pop()

    last_gate = None  # gate that rejected the latest evidence() call

    def evidence(patch, across, green, gauge):
        """Rail morphology plus localized, transverse sleeper repetition.

        Every gate only rejects and none feeds the returned score except
        through its own inputs, so gates run cheapest/most selective first.
        """
        nonlocal last_gate
        last_gate = None
        counts["evidence"] += 1

        def reject(gate):
            nonlocal last_gate
            last_gate = gate
            counts["evidence_reject_" + gate] += 1
            return 0.0

        if len(patch) < max(32, gauge * 1.7):
            return reject("length")
        core = np.abs(across) <= gauge * 0.37
        if not core.any() or green[:, core].mean() > 0.55:
            return reject("green")

        ridge = ridge_profile(patch, gauge)
        sides = [
            np.abs(across - side * gauge / 2) <= max(2, gauge * 0.12)
            for side in (-1, 1)
        ]
        # An empty side window also fails the morphology gate.
        if not all(mask.any() for mask in sides):
            return reject("ridge")
        strengths = [float(ridge[mask].max()) for mask in sides]
        if min(strengths) < 1.6:
            return reject("ridge")
        ridges = np.flatnonzero(
            (ridge[1:-1] > ridge[:-2])
            & (ridge[1:-1] >= ridge[2:])
            & (ridge[1:-1] > min(strengths) * 0.85)
        ) + 1
        if np.sum(np.abs(across[ridges]) > gauge * 0.72) >= 5:
            return reject("ridge_clutter")

        groups = [
            (across >= -gauge * 0.36) & (across <= -gauge * 0.13),
            np.abs(across) <= max(0.6, gauge * 0.10),
            (across >= gauge * 0.13) & (across <= gauge * 0.36),
        ]
        if any(not mask.any() for mask in groups):
            return reject("groups")
        signals = [
            highpass(patch[:, mask].mean(axis=1), max(2, gauge * 0.30))
            for mask in groups
        ]
        limit = min(32, max(1, int(gauge * 0.32)), (len(patch) - 24) // 2)
        if limit < 1:
            return reject("groups")
        target = signals[1][limit:-limit]
        shifts = np.arange(-limit, limit + 1)
        scores = [sliding_corr(target, signal) for signal in (signals[0], signals[2])]
        joint = np.minimum(scores[0], scores[1][::-1])
        best = int(np.argmax(joint))
        coherence = float(joint[best])
        if coherence < 0.40:
            return reject("coherence")
        shift = int(shifts[best])
        signal = (
            target
            + signals[0][limit + shift:len(patch) - limit + shift]
            + signals[2][limit - shift:len(patch) - limit - shift]
        ) / 3
        amplitude = float(np.std(signal))
        if amplitude < 2.0:
            return reject("amplitude")
        low = max(2, int(gauge * 0.10))
        high = min(max(low + 3, int(gauge * 0.95)), len(signal) // 3)
        if high <= low + 2:
            return reject("period")
        lags = np.arange(low, high + 1)
        ac = autocorr(signal, lags)
        peaks = np.flatnonzero(
            (ac[1:-1] > ac[:-2]) & (ac[1:-1] >= ac[2:])
        ) + 1
        if not len(peaks):
            return reject("period")
        k = int(peaks[np.argmax(ac[peaks])])
        period, periodicity = int(lags[k]), float(ac[k])
        half = max(1, period // 2)
        if periodicity < 0.20 or (
            periodicity - corr(signal[:-half], signal[half:]) < 0.16
        ):
            return reject("period")

        outer_masks = [
            (side * across >= gauge * 0.88)
            & (side * across <= gauge * 1.35)
            for side in (-1, 1)
        ]
        if any(not mask.any() for mask in outer_masks):
            return reject("outer")
        outer_corr = []
        for mask in outer_masks:
            outside = highpass(
                patch[:, mask].mean(axis=1), max(2, gauge * 0.30)
            )[limit:-limit]
            outer_corr.append(abs(corr(signal, outside)))
        if min(outer_corr) > 0.53:
            return reject("outer")

        if not morphology_ok(patch, across, gauge):
            return reject("morphology")

        slope = -2.0 * shift / max(1.0, gauge * 0.49)
        margin = int(np.ceil(abs(slope) * gauge * 1.4)) + 3
        if len(patch) - 2 * margin < 24:
            return reject("cross")
        rows = np.arange(margin, len(patch) - margin, dtype=np.float32)
        mx = np.broadcast_to(
            np.arange(len(across), dtype=np.float32),
            (len(rows), len(across))
        ).copy()
        my = (rows[:, None] + slope * across[None]).astype(np.float32)
        aligned = cv2.remap(
            patch, mx, my, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        aligned_green = cv2.remap(
            green, mx, my, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1
        )
        residual = aligned - cv2.GaussianBlur(
            aligned, (1, 0), sigmaX=0, sigmaY=max(1.2, period * 0.45)
        )
        cross = residual[:, core]
        cross_signal = cross.mean(axis=1)
        cross_amplitude = float(np.std(cross_signal))
        if cross_amplitude < 1.7:
            return reject("cross")
        for mask in outer_masks:
            outside = residual[:, mask].mean(axis=1)
            if (corr(cross_signal, outside) > 0.72
                    and np.std(outside) > cross_amplitude * 0.80):
                return reject("cross")

        smooth_signal = cv2.GaussianBlur(
            cross_signal[:, None], (1, 0), sigmaX=0, sigmaY=0.65
        ).ravel()
        for polarity in (-1, 1):
            response = polarity * smooth_signal
            peaks = np.flatnonzero(
                (response[1:-1] > response[:-2])
                & (response[1:-1] >= response[2:])
                & (response[1:-1] > max(1.2, cross_amplitude * 0.4))
            ) + 1
            selected = []
            for k in peaks[np.argsort(-response[peaks], kind="stable")]:
                if all(abs(int(k) - j) >= max(2, period * 0.55)
                       for j in selected):
                    selected.append(int(k))
            valid = []
            for k in sorted(selected):
                row = polarity * cross[k]
                level = max(0.75, float(response[k]) * 0.22)
                if np.mean(row > level) < 0.68:
                    continue
                if min(float(v.mean()) for v in np.array_split(row, 3)) < level:
                    continue
                exterior = [
                    float((polarity * residual[k, mask]).mean())
                    for mask in outer_masks
                ]
                if min(exterior) > response[k] * 0.65:
                    continue
                if aligned_green[k, core].mean() > 0.55:
                    continue
                valid.append(k)
            if len(valid) < 4:
                continue
            spacing = np.diff(valid)
            if np.mean((spacing >= period * 0.55)
                       & (spacing <= period * 1.65)) < 0.65:
                continue
            if valid[-1] - valid[0] < max(gauge * 1.25, period * 3):
                continue
            counts["evidence_pass"] += 1
            return float(
                coherence * periodicity * min(1.5, amplitude / 5)
                * (1 - 0.35 * np.mean(outer_corr))
            )
        return reject("sleepers")

    clock.push("proposal_generation")
    lines = np.asarray(raw, np.float32).reshape(-1, 4).copy()
    lines *= np.array([width / sw, height / sh] * 2, np.float32)
    p, q = lines[:, :2], lines[:, 2:]
    lengths = np.linalg.norm(q - p, axis=1)
    centers = (p + q) / 2
    cell = max(64.0, diagonal * 0.020)
    ids = np.flatnonzero(lengths >= max(12, diagonal * 0.0025))
    ids = ids[np.argsort(-lengths[ids], kind="stable")]
    selected, occupancy = [], {}
    for i in ids:
        key = tuple((centers[i] // cell).astype(int))
        if occupancy.get(key, 0) >= 12:
            continue
        occupancy[key] = occupancy.get(key, 0) + 1
        selected.append(int(i))
        if len(selected) >= 1600:
            break
    if not selected:
        clock.pop()
        return []
    p, q, lengths, centers = (
        p[selected], q[selected], lengths[selected], centers[selected]
    )
    directions = (q - p) / lengths[:, None]
    directions[directions[:, 0] < 0] *= -1
    proposals = []
    for i, u in enumerate(directions):
        n = normal(u)
        delta = centers - centers[i]
        gauge = np.abs(delta @ n)
        along = delta @ u
        candidates = np.flatnonzero(
            (np.arange(len(p)) > i)
            & (np.abs(directions @ u) > 0.989)
            & (gauge >= 3.5) & (gauge <= maximum_gauge)
            & (np.abs(along) < (lengths + lengths[i]) / 2)
        )
        if not len(candidates):
            continue
        a0, a1 = sorted((float(p[i] @ u), float(q[i] @ u)))
        lo = np.maximum(a0, np.minimum(p[candidates] @ u, q[candidates] @ u))
        hi = np.minimum(a1, np.maximum(p[candidates] @ u, q[candidates] @ u))
        overlap = hi - lo
        error = np.abs(directions[candidates] @ n)
        valid = np.flatnonzero(
            (overlap >= np.maximum(12, gauge[candidates] * 0.8))
            & (error * overlap < np.maximum(3, gauge[candidates] * 0.30))
        )
        quality = overlap / (1 + error * 12)
        for k in valid[np.argsort(-quality[valid], kind="stable")[:5]]:
            j = int(candidates[k])
            v = directions[j].copy()
            if v @ u < 0:
                v = -v
            axis = unit(u + v)
            count = min(4, max(1, int(overlap[k] / max(50, gauge[j] * 2))))
            for position in np.linspace(
                lo[k] + overlap[k] * 0.2, hi[k] - overlap[k] * 0.2, count
            ):
                ca = centers[i] + u * (position - centers[i] @ u)
                cb = centers[j] + v * (
                    (position - centers[j] @ u) / max(float(v @ u), 1e-6)
                )
                proposals.append((
                    float(quality[k]), (ca + cb) / 2, axis,
                    abs(float((cb - ca) @ normal(axis)))
                ))

    # A coarse segment can supply direction even if its companion rail
    # disappeared during downsampling. Search only bounded native strips.
    local_cells = set()
    for i in range(len(p)):
        if lengths[i] < max(24, diagonal * 0.005):
            continue
        key = tuple((centers[i] // cell).astype(int))
        if key in local_cells:
            continue
        local_cells.add(key)
        extent = int(np.clip(lengths[i] * 0.5, 28, 80))
        patch, across, _ = strip(
            centers[i], directions[i], maximum_gauge, extent, 1.02
        )
        if len(patch) < extent * 0.6:
            continue
        profile = patch.mean(axis=0)
        ridge = np.abs(profile - cv2.GaussianBlur(
            profile[None], (0, 1), sigmaX=2.4
        ).ravel())
        peaks = np.flatnonzero(
            (ridge[1:-1] > ridge[:-2])
            & (ridge[1:-1] >= ridge[2:])
            & (ridge[1:-1] > 1.8)
        ) + 1
        peaks = peaks[np.argsort(-ridge[peaks], kind="stable")[:14]]
        pairs = []
        for a in range(len(peaks)):
            for b in range(a + 1, len(peaks)):
                ia, ib = sorted((int(peaks[a]), int(peaks[b])))
                gauge = float(across[ib] - across[ia])
                shift = float((across[ib] + across[ia]) / 2)
                if (3.5 <= gauge <= min(maximum_gauge, lengths[i] * 1.8)
                        and abs(shift) <= max(12, gauge * 0.65)):
                    score = min(float(ridge[ia]), float(ridge[ib]))
                    pairs.append((score / (1 + abs(shift) / max(8, gauge)),
                                  shift, gauge))
        pairs.sort(key=lambda item: -item[0])
        for _, shift, gauge in pairs[:3]:
            proposals.append((
                min(float(lengths[i]), 120),
                centers[i] + normal(directions[i]) * shift,
                directions[i].copy(), gauge
            ))
        if len(local_cells) >= 130:
            break

    proposals.sort(key=lambda item: -item[0])
    buckets = {}
    for proposal in proposals:
        key = tuple((proposal[1] // cell).astype(int))
        bucket = buckets.setdefault(key, [])
        if len(bucket) < 14:
            bucket.append(proposal)
    ordered = [
        bucket[rank]
        for rank in range(14)
        for bucket in buckets.values()
        if rank < len(bucket)
    ]
    clock.pop()

    def path_distance(point, path):
        d = np.diff(path, axis=0)
        a = path[:-1]
        fraction = np.sum((point - a) * d, axis=1)
        fraction /= np.maximum(np.sum(d * d, axis=1), 1e-6)
        foot = a + np.clip(fraction, 0, 1)[:, None] * d
        return float(np.linalg.norm(foot - point, axis=1).min())

    def verify_seed(center, axis, gauge):
        counts["verify_seed"] += 1
        clock.push("verification")
        try:
            extent = float(np.clip(gauge * 5.5, 80, 460))
            patch, across, green = strip(
                center, axis, gauge, extent, 1.4, True
            )
            if len(patch) < 32:
                return 0.0
            quality = evidence(patch, across, green, gauge)
            if quality >= 0.075:
                return quality
            if last_gate in NO_RETRY_GATES:
                counts["subwindow_retry_skipped"] += 1
                return quality
            window = int(max(48, gauge * 2.8))
            if len(patch) >= window + 12:
                for start in np.unique(
                    np.linspace(0, len(patch) - window, 3).astype(int)
                ):
                    quality = max(quality, evidence(
                        patch[start:start + window], across,
                        green[start:start + window], gauge
                    ))
            return quality
        finally:
            clock.pop()

    def trace(seed, seed_axis, seed_gauge, sign):
        """Track locally fitted sections; bridge gaps only after reacquisition."""
        point = seed.copy()
        direction = seed_axis * sign
        gauge = float(seed_gauge)
        path, gauges = [point.copy()], [gauge]
        pending = []
        missed_distance = 0.0
        traveled = 0.0
        curvature = 0.0
        single_distance = 0.0
        for _ in range(230):
            step = float(np.clip(gauge * 0.65, 10, 30))
            predicted_axis = rotate(direction, np.clip(curvature * step, -0.045, 0.045))
            expected = point + predicted_axis * step
            if not inside(expected):
                border_steps = []
                for coordinate, component, bound in zip(
                    point, predicted_axis, (width - 1, height - 1)
                ):
                    if component > 1e-6:
                        border_steps.append((bound - coordinate) / component)
                    elif component < -1e-6:
                        border_steps.append(-coordinate / component)
                step = min(border_steps, default=0.0) - 0.5
                if step < 2:
                    break
                expected = point + predicted_axis * step
            extent = int(np.clip(gauge * 1.45, 24, 76))
            best = None
            angles = (-0.07, 0.0, 0.07) if not pending else (-0.10, 0.0, 0.10)
            for angle in angles:
                candidate_axis = rotate(predicted_axis, angle)
                fit = fit_section(
                    expected, candidate_axis, gauge, extent,
                    search=1.0 if not pending else 1.35,
                    allow_single=not pending and single_distance < gauge * 1.5
                )
                if fit is None:
                    continue
                fitted_point, fitted_gauge, strength, paired = fit
                shift = float((fitted_point - expected) @ normal(predicted_axis))
                if abs(shift) > max(3, gauge * 0.25):
                    continue
                if abs(np.log(fitted_gauge / gauge)) > 0.18:
                    continue
                score = (
                    min(strength, 12) - 8 * abs(angle)
                    - 0.8 * abs(shift) / max(2, gauge * 0.15)
                    - 2.0 * abs(np.log(fitted_gauge / gauge))
                    + (0.8 if paired else -1.5)
                )
                if best is None or score > best[0]:
                    best = (score, fit, candidate_axis)

            traveled += step
            if traveled > diagonal * 1.5:
                break
            if best is None:
                pending.append((expected.copy(), gauge))
                point = expected
                direction = predicted_axis
                missed_distance += step
                if missed_distance > min(diagonal * 0.055, max(65, gauge * 5.0)):
                    break
                continue

            _, (new_point, new_gauge, strength, paired), fitted_axis = best
            if not paired:
                single_distance += step
                pending.append((new_point.copy(), gauge))
                point = new_point
                direction = unit(direction * 0.75 + fitted_axis * 0.25)
                missed_distance += step
                if single_distance > gauge * 1.5:
                    break
                continue

            if pending:
                # A weak reacquisition cannot authorize a vegetation bridge.
                if strength < 1.8:
                    point = expected
                    pending.append((expected.copy(), gauge))
                    missed_distance += step
                    if missed_distance > min(diagonal * 0.055, max(65, gauge * 5)):
                        break
                    continue
                start = path[-1]
                chord = new_point - start
                gap = float(np.linalg.norm(chord))
                if gap <= 0 or float(unit(chord) @ direction) < 0.94:
                    break
                old_axis = (
                    unit(path[-1] - path[-2]) if len(path) > 1 else seed_axis * sign
                )
                end_axis = fitted_axis
                for fraction in np.linspace(0, 1, len(pending) + 2)[1:-1]:
                    f2, f3 = fraction ** 2, fraction ** 3
                    bridge = (
                        (2 * f3 - 3 * f2 + 1) * start
                        + (f3 - 2 * f2 + fraction) * gap * old_axis
                        + (-2 * f3 + 3 * f2) * new_point
                        + (f3 - f2) * gap * end_axis
                    )
                    if not inside(bridge):
                        bridge = start + fraction * chord
                    path.append(bridge.astype(np.float32))
                    gauges.append((1 - fraction) * gauges[-1] + fraction * new_gauge)
                pending.clear()
                curvature = 0.0

            displacement = new_point - point
            measured = unit(displacement) if np.linalg.norm(displacement) > 1 else fitted_axis
            next_axis = unit(predicted_axis * 0.45 + fitted_axis * 0.25 + measured * 0.30)
            bend = float(np.arctan2(
                direction[0] * next_axis[1] - direction[1] * next_axis[0],
                direction @ next_axis
            ))
            curvature = float(np.clip(
                0.7 * curvature + 0.3 * bend / max(step, 1), -0.003, 0.003
            ))
            point, direction = new_point, next_axis
            gauge = float(0.55 * gauge + 0.45 * new_gauge)
            path.append(point.copy())
            gauges.append(gauge)
            missed_distance = single_distance = 0.0

        return np.asarray(path, np.float32), np.asarray(gauges, np.float32)

    def tangents(path):
        vectors = np.gradient(path, axis=0)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1)[:, None], 1e-6)

    def verify_path(path, gauges):
        counts["verify_path"] += 1
        clock.push("verification")
        try:
            arc = arc_length(path)
            total = float(arc[-1])
            count = min(11, max(3, int(total / max(90, np.median(gauges) * 3)) + 1))
            positions = np.linspace(0, total, count)
            qualities = []
            for position in positions:
                gauge = float(np.interp(position, arc, gauges))
                extent = float(np.clip(gauge * 4.5, 70, 400))
                lo, hi = max(0, position - extent / 2), min(total, position + extent / 2)
                samples = np.arange(lo, hi + 0.25, dtype=np.float32)
                if len(samples) < 32:
                    qualities.append(0.0)
                    continue
                points = np.column_stack([
                    np.interp(samples, arc, path[:, k]) for k in range(2)
                ]).astype(np.float32)
                local_gauges = np.interp(samples, arc, gauges)
                reference = float(np.median(local_gauges))
                axes = tangents(points)
                normals = np.column_stack((-axes[:, 1], axes[:, 0]))
                radius = int(np.ceil(reference * 1.4)) + 3
                across = np.arange(-radius, radius + 1, dtype=np.float32)
                offsets = across[None] * (local_gauges / reference)[:, None]
                patch, across, green = remap(
                    points[:, :1] + normals[:, :1] * offsets,
                    points[:, 1:] + normals[:, 1:] * offsets,
                    across, reference, True
                )
                quality = evidence(patch, across, green, reference)
                window = int(max(48, reference * 2.8))
                if quality < 0.075 and len(patch) >= window + 12:
                    for offset in (0, (len(patch) - window) // 2, len(patch) - window):
                        quality = max(quality, evidence(
                            patch[offset:offset + window], across,
                            green[offset:offset + window], reference
                        ))
                qualities.append(quality)
            return positions, np.asarray(qualities)
        finally:
            clock.pop()

    output, accepted, attempted = [], [], set()
    seed_checks, traced = 0, 0
    clock.push("seed_loop")
    try:
        for _, center, axis, gauge in ordered:
            if seed_checks >= SEED_BUDGET or traced >= 16:
                break
            if not 3.5 <= gauge <= maximum_gauge or not inside(center):
                continue
            angle = float(np.arctan2(axis[1], axis[0]))
            key = (
                round(float(center[0]) / max(10, gauge)),
                round(float(center[1]) / max(10, gauge)),
                round(angle / 0.10),
                round(np.log(gauge) / 0.18)
            )
            if key in attempted:
                continue
            attempted.add(key)
            if any(path_distance(center, old) < max(5, gauge * 0.45)
                   for old in accepted):
                continue
            extent = int(np.clip(gauge * 1.6, 28, 80))
            fit = fit_section(center, axis, gauge, extent, 1.2)
            if fit is None or not fit[3]:
                continue
            center, gauge = fit[:2]
            seed_checks += 1
            quality = verify_seed(center, axis, gauge)
            if quality < 0.075:
                continue

            traced += 1
            counts["traced"] += 1
            clock.push("trace")
            try:
                backward, bg = trace(center, axis, gauge, -1)
                forward, fg = trace(center, axis, gauge, 1)
            finally:
                clock.pop()
            path = np.vstack((backward[:0:-1], forward))
            gauges = np.r_[bg[:0:-1], fg]
            if len(path) < 3:
                continue
            keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1]
            path, gauges = path[keep], gauges[keep]
            if len(path) < 3 or arc_length(path)[-1] < minimum_length:
                continue

            # Final boundary fitting follows the traced curve at native scale.
            axes = tangents(path)
            left = path - np.column_stack((-axes[:, 1], axes[:, 0])) * gauges[:, None] / 2
            right = path + np.column_stack((-axes[:, 1], axes[:, 0])) * gauges[:, None] / 2
            for rail in (left, right):
                if len(rail) > 3:
                    rail[1:-1] = (
                        rail[:-2] * 0.15 + rail[1:-1] * 0.70 + rail[2:] * 0.15
                    )
            path = (left + right) / 2
            gauges = np.linalg.norm(right - left, axis=1)
            positions, qualities = verify_path(path, gauges)
            passed = qualities >= 0.075
            total = float(arc_length(path)[-1])
            if passed.sum() < 2:
                continue
            confirmed = positions[passed]
            if confirmed[-1] - confirmed[0] < min(
                minimum_length * 0.4, max(30, float(np.median(gauges)) * 1.5)
            ):
                continue
            # Reject a long drift supported by only one small seed neighborhood.
            if confirmed[-1] - confirmed[0] < total * 0.20:
                continue
            if any(
                np.mean([
                    path_distance(point, old) < max(5, float(local_gauge) * 0.45)
                    for point, local_gauge in zip(path[::3], gauges[::3])
                ]) > 0.70 for old in accepted
            ):
                continue

            for rail in (left, right):
                rail[:, 0] = np.clip(rail[:, 0], 0, width - 1)
                rail[:, 1] = np.clip(rail[:, 1], 0, height - 1)
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                continue
            span = float(np.linalg.norm(np.ptp(np.vstack((left, right)), axis=0)))
            if min(arc_length(left)[-1], arc_length(right)[-1], span) < minimum_length:
                continue
            if left[:, 0].mean() > right[:, 0].mean():
                left, right = right, left
            accepted.append(path)
            output.append({
                "left_rail": left.astype(float).tolist(),
                "right_rail": right.astype(float).tolist(),
                "confidence": float(passed.sum() / len(passed)),
                "confidence_method": "verified_patch_fraction_v1",
            })
            if len(output) >= 6:
                break
    finally:
        clock.pop()
    return output
