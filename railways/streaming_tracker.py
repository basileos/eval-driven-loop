"""Stateful video adapter: full acquisition, feature-based motion, and ROI revalidation.

Tracking propagates geometry; it is NOT a new railway classification. No model
weights, detector thresholds, or image sampling resolution are changed here.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

if __package__:
    from .evaluator import calculate_prediction_size_metrics, validate_predicted_tracks
    from .run_candidate import attach_confidence
else:
    from evaluator import calculate_prediction_size_metrics, validate_predicted_tracks
    from run_candidate import attach_confidence


def points_of(track):
    return np.asarray(track['left_rail'] + track['right_rail'], np.float32)


def roi_of(track, shape, margin):
    points = points_of(track)
    lo = np.floor(points.min(axis=0) - margin).astype(int)
    hi = np.ceil(points.max(axis=0) + margin + 1).astype(int)
    h, w = shape[:2]
    return max(0, lo[0]), max(0, lo[1]), min(w, hi[0]), min(h, hi[1])


def clip_polyline(points, width, height):
    """Longest run of the polyline inside the image, cut exactly at the border."""
    points = np.asarray(points, np.float64)
    upper = np.array([width - 1, height - 1], np.float64)
    inside = ((points >= 0) & (points <= upper)).all(axis=1)
    best, start = None, None
    for index, flag in enumerate(np.append(inside, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None:
        return None

    def border(a, b):
        # Point where segment a (inside) -> b (outside) leaves the image.
        t = 1.0
        for axis in (0, 1):
            delta = b[axis] - a[axis]
            if b[axis] < 0:
                t = min(t, -a[axis] / delta)
            elif b[axis] > upper[axis]:
                t = min(t, (upper[axis] - a[axis]) / delta)
        return np.clip(a + t * (b - a), 0, upper)

    first, last = best
    clipped = list(points[first:last])
    if first > 0:
        clipped.insert(0, border(points[first], points[first - 1]))
    if last < len(points):
        clipped.append(border(points[last - 1], points[last]))
    clipped = np.asarray(clipped)
    return clipped if len(clipped) >= 2 else None


def dense_polyline(points, count=32):
    """`count` points evenly spaced by arc length."""
    p = np.asarray(points, np.float64)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, arc[-1], count)
    return np.column_stack([np.interp(t, arc, p[:, i]) for i in (0, 1)])


def gauge_of(track):
    """Median distance from left-rail samples to the nearest right-rail sample.

    Independent of point indexing, so rails clipped at different places still
    measure correctly.
    """
    left, right = dense_polyline(track['left_rail']), dense_polyline(track['right_rail'])
    return float(np.median(np.linalg.norm(left[:, None] - right[None], axis=2).min(axis=1)))


def support_distance(predicted, candidate):
    """How far the predicted rails lie from the candidate's rails (median px).

    Directed: a candidate that extends the same line beyond the predicted
    track still matches, which match_distance (symmetric) penalises.
    """
    def directed(a, b):
        return float(np.median(np.linalg.norm(a[:, None] - b[None], axis=2).min(axis=1)))

    a, b = [dense_polyline(predicted[key]) for key in ('left_rail', 'right_rail')]
    c, d = [dense_polyline(candidate[key]) for key in ('left_rail', 'right_rail')]
    return min(max(directed(a, c), directed(b, d)), max(directed(a, d), directed(b, c)))


def offset_track(track, dx, dy):
    result = dict(track)
    for key in ('left_rail', 'right_rail'):
        result[key] = (np.asarray(track[key]) + [dx, dy]).tolist()
    return result


def match_distance(first, second):
    """Symmetric nearest-point distance after sampling both rail polylines."""
    def dense(track, key):
        p = np.asarray(track[key], np.float32)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.linspace(0, arc[-1], 32)
        return np.column_stack([np.interp(t, arc, p[:, i]) for i in (0, 1)])

    def distance(a, b):
        d = np.linalg.norm(a[:, None] - b[None, :], axis=2)
        return max(float(np.median(d.min(axis=0))), float(np.median(d.min(axis=1))))

    a, b = [dense(first, key) for key in ('left_rail', 'right_rail')]
    c, d = [dense(second, key) for key in ('left_rail', 'right_rail')]
    return min(max(distance(a, c), distance(b, d)), max(distance(a, d), distance(b, c)))


class StreamingTracker:
    def __init__(self, detect, *, detect_roi=None, verify_every=10,
                 full_search_every=150, roi_margin=96, max_gap_seconds=10.0,
                 motion_max_side=960, association_gauges=1.5, verify_region='frame'):
        self.detect = detect
        self.detect_roi = detect_roi
        self.verify_every = verify_every
        self.full_search_every = full_search_every
        self.roi_margin = roi_margin
        self.max_gap_seconds = max_gap_seconds
        # An ROI re-detection confirms the predicted track when it lies within
        # this many rail gauges (min 12 px). Consecutive detections of the same
        # track already differ by ~1 gauge on drone footage; the centre of an
        # adjacent parallel track is ~3 gauges away.
        self.association_gauges = association_gauges
        # 'frame': periodic verification re-detects on the whole frame, so the
        # track extent follows newly visible rail. 'roi': cheaper crop around
        # the prediction, but geometry can then never grow past that crop.
        if verify_region not in ('frame', 'roi'):
            raise ValueError("verify_region must be 'frame' or 'roi'")
        self.verify_region = verify_region
        # Frame-to-frame motion is a RANSAC homography over ORB matches on frames
        # downscaled so the longer side is at most this many pixels (None:
        # native). Local optical flow could not follow drone footage across
        # live-sized gaps (a search alone takes ~0.5 s, i.e. 60-160 px motion);
        # matched features can, at the detector's own frame-to-frame jitter.
        # Motion is always measured from the keyframe (the last frame whose
        # tracks came from the detector), so errors do not accumulate.
        self.motion_max_side = motion_max_side
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.keyframe = None
        self._current_features = None
        self.sequence = -1
        self.last_search = -full_search_every
        self.last_verified = -verify_every
        self.previous_gray = None
        self.previous_timestamp = None
        self.tracks = []
        self.diagnostics = {}

    def reset(self):
        self.tracks = []
        self.previous_gray = None
        self.keyframe = None
        self.previous_timestamp = None

    def _detect(self, image, reference_shape=None):
        started = time.perf_counter()
        self.diagnostics['detector_calls'] += 1
        self.diagnostics['detector_pixels'] += int(image.shape[0] * image.shape[1])
        try:
            raw = (self.detect_roi(image, reference_shape)
                   if reference_shape is not None and self.detect_roi is not None else self.detect(image))
            tracks = validate_predicted_tracks(raw, image.shape[:2])
            attach_confidence(raw, tracks)
            return tracks
        finally:
            self.diagnostics['detector_seconds'] += time.perf_counter() - started

    def _valid_size(self, track, shape):
        return not calculate_prediction_size_metrics([track], shape)['undersized_predicted_tracks']

    def _search(self, frame, reason):
        self.diagnostics.update(mode='search', reason=reason)
        self.last_search = self.sequence
        self.last_verified = self.sequence
        return [track for track in self._detect(frame) if self._valid_size(track, frame.shape[:2])]

    def _features(self, gray):
        """ORB keypoints (native-pixel coordinates), descriptors and the scale used."""
        if self._current_features is not None:
            return self._current_features
        scale = 1.0
        if self.motion_max_side:
            scale = min(1.0, self.motion_max_side / max(gray.shape))
        small = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        keypoints, descriptors = self.orb.detectAndCompute(small, None)
        points = np.float32([k.pt for k in keypoints]).reshape(-1, 2)
        if len(points):
            # Coarse pyramid levels quantize ORB positions; refine on full detail.
            points = cv2.cornerSubPix(
                small, points.reshape(-1, 1, 2).copy(), (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            ).reshape(-1, 2)
        factor = np.array([small.shape[1] / gray.shape[1], small.shape[0] / gray.shape[0]])
        points = (points + 0.5) / factor - 0.5
        self._current_features = (points, descriptors, scale)
        return self._current_features

    def _propagate(self, gray):
        def lost(reason):
            self.diagnostics['propagation_failure'] = reason
            return None

        if self.keyframe is None:
            return lost('no_reference')
        (previous_points, previous_descriptors, _), keyframe_tracks = self.keyframe
        points, descriptors, scale = self._features(gray)
        if (previous_descriptors is None or descriptors is None
                or len(previous_descriptors) < 30 or len(descriptors) < 30):
            return lost('few_features')
        pairs = self.matcher.knnMatch(previous_descriptors, descriptors, k=2)
        good = [m for m, n in (pair for pair in pairs if len(pair) == 2)
                if m.distance < 0.8 * n.distance]
        if len(good) < 30:
            return lost('few_matches')
        a = previous_points[[m.queryIdx for m in good]]
        b = points[[m.trainIdx for m in good]]
        # 3 px at matching resolution; the model is fitted in native pixels.
        homography, inliers = cv2.findHomography(
            a, b, cv2.RANSAC, 3.0 / scale, maxIters=2000, confidence=0.995)
        if homography is None or inliers is None or not np.isfinite(homography).all():
            return lost('no_motion_model')
        inliers = inliers.ravel().astype(bool)
        if inliers.sum() < 25 or inliers.mean() < 0.25:
            return lost('few_inliers')

        def project(points):
            points = np.asarray(points, np.float64).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(points, homography).reshape(-1, 2)

        height, width = gray.shape
        propagated = []
        for track in keyframe_tracks:
            # Local similarity of the homography at the track: reject zoom,
            # flips and spins no aerial camera makes between processed frames.
            center = np.asarray(track['left_rail'] + track['right_rail'], np.float64).mean(axis=0)
            origin, step_x, step_y = project([center, center + [1, 0], center + [0, 1]])
            jacobian = np.column_stack((step_x - origin, step_y - origin))
            determinant = float(np.linalg.det(jacobian))
            angle = abs(float(np.arctan2(jacobian[1, 0] - jacobian[0, 1],
                                         jacobian[0, 0] + jacobian[1, 1])))
            self.diagnostics.update(motion_scale=float(np.sqrt(max(determinant, 0))),
                                    motion_rotation=angle)
            # Altitude changes zoom the scene ~1.3x within a second of footage.
            if determinant <= 0 or not 0.67 <= np.sqrt(determinant) <= 1.5 or angle > 0.35:
                return lost('implausible_motion')
            updated = {}
            for key in ('left_rail', 'right_rail'):
                # The camera moves along the line: keep the visible part.
                projected = clip_polyline(project(track[key]), width, height)
                if projected is None:
                    return lost('left_frame')
                updated[key] = projected.tolist()
            if not self._valid_size(updated, gray.shape):
                return lost('undersized')
            # The rail pair must stay a rail pair: its spacing follows the
            # local zoom. Homographies that splay or merge the rails are wrong.
            expected = gauge_of(track) * np.sqrt(determinant)
            if not 0.67 <= gauge_of(updated) / max(expected, 1e-6) <= 1.5:
                return lost('gauge_changed')
            updated.update(confidence=None, confidence_method=None)
            propagated.append(updated)
        self.diagnostics['motion_support'] = float(inliers.mean())
        return propagated

    def _verify(self, frame, predicted):
        output = []
        rois = []
        self.diagnostics['rois'] = rois
        shape = frame.shape[:2]
        if self.verify_region == 'frame':
            candidates = [item for item in self._detect(frame) if self._valid_size(item, shape)]
        for track in predicted:
            if self.verify_region == 'roi':
                x0, y0, x1, y1 = roi_of(track, frame.shape, self.roi_margin)
                rois.append([int(v) for v in (x0, y0, x1, y1)])
                found = [offset_track(item, x0, y0) for item in self._detect(frame[y0:y1, x0:x1], shape)]
                found = [item for item in found if self._valid_size(item, shape)]
            else:
                found = candidates
            ranked = sorted([(support_distance(track, item), i) for i, item in enumerate(found)])
            # Keep association tight; do not silently switch to an adjacent track.
            limit = max(12.0, self.association_gauges * gauge_of(track))
            if not ranked or ranked[0][0] > limit:
                self.diagnostics['verification_failure'] = 'no_roi_detection' if not ranked else 'geometry_mismatch'
                self.diagnostics['association_distance_px'] = ranked[0][0] if ranked else None
                return None
            if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 4:
                self.diagnostics['verification_failure'] = 'ambiguous_association'
                return None
            chosen = found[ranked[0][1]]
            if any(match_distance(chosen, prior) < 6 for prior in output):
                self.diagnostics['verification_failure'] = 'duplicate_track'
                return None
            output.append(chosen)
        return output

    def process(self, frame, timestamp_seconds=None):
        self.sequence += 1
        self._current_features = None
        self.diagnostics = dict(mode='search', reason=None, detector_calls=0,
                                detector_pixels=0, detector_seconds=0.0,
                                motion_seconds=0.0, motion_support=None)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            reason = None
            if self.previous_gray is not None and gray.shape != self.previous_gray.shape:
                self.reset()
                reason = 'resolution_changed'
            if self.previous_timestamp is not None and timestamp_seconds is not None:
                gap = timestamp_seconds - self.previous_timestamp
                if gap <= 0 or gap > self.max_gap_seconds:
                    self.reset()
                    reason = 'timestamp_gap'
            due_full = self.sequence - self.last_search >= self.full_search_every
            if not self.tracks or due_full:
                tracks = self._search(frame, reason or ('periodic_refresh' if self.tracks else 'acquire'))
            else:
                started = time.perf_counter()
                tracks = self._propagate(gray)
                self.diagnostics['motion_seconds'] = time.perf_counter() - started
                self.diagnostics.update(mode='track', reason=None)
                if tracks is not None and self.sequence - self.last_verified >= self.verify_every:
                    self.diagnostics['mode'] = 'roi_verify'
                    tracks = self._verify(frame, tracks)
                    if tracks:
                        self.last_verified = self.sequence
                if not tracks:
                    self.reset()
                    # Reacquire on this frame immediately; the live runner already
                    # bounds work to one in-flight job and skips frames when busy.
                    tracks = self._search(frame, 'tracking_lost')
            self.tracks = tracks
            self.previous_gray = gray if tracks else None
            if not tracks:
                self.keyframe = None
            elif self.diagnostics['mode'] != 'track':  # geometry came from the detector
                self.keyframe = (self._features(gray), tracks)
            self.previous_timestamp = timestamp_seconds if tracks else None
            self.diagnostics['verification_age_frames'] = self.sequence - self.last_verified if tracks else None
            return tracks
        except Exception:
            self.reset()
            self.diagnostics.update(mode='error', reason='processing_failed')
            raise
