"""Stateful video adapter: full acquisition, optical flow, and ROI revalidation.

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
    def __init__(self, detect, *, detect_roi=None, verify_every=10, search_every=30,
                 full_search_every=150, roi_margin=96, max_gap_seconds=10.0):
        self.detect = detect
        self.detect_roi = detect_roi
        self.verify_every = verify_every
        self.search_every = search_every
        self.full_search_every = full_search_every
        self.roi_margin = roi_margin
        self.max_gap_seconds = max_gap_seconds
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

    def _propagate(self, gray):
        propagated = []
        qualities = []
        for track in self.tracks:
            x0, y0, x1, y1 = roi_of(track, gray.shape, self.roi_margin)
            old = np.ascontiguousarray(self.previous_gray[y0:y1, x0:x1])
            new = np.ascontiguousarray(gray[y0:y1, x0:x1])
            if min(old.shape) < 24:
                return None
            mask = np.zeros(old.shape, np.uint8)
            for key in ('left_rail', 'right_rail'):
                p = np.rint(np.asarray(track[key]) - [x0, y0]).astype(np.int32)
                cv2.polylines(mask, [p], False, 255, 24)
            points = cv2.goodFeaturesToTrack(old, 160, 0.02, 8, mask=mask, blockSize=7)
            if points is None or len(points) < 12:
                return None
            kwargs = dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.02))
            moved, status, error = cv2.calcOpticalFlowPyrLK(old, new, points, None, **kwargs)
            if moved is None or status is None or not np.isfinite(moved).all():
                return None
            back, reverse_status, _ = cv2.calcOpticalFlowPyrLK(new, old, moved, None, **kwargs)
            if back is None or reverse_status is None:
                return None
            fb = np.linalg.norm((points - back).reshape(-1, 2), axis=1)
            good = ((status.ravel() != 0) & (reverse_status.ravel() != 0)
                    & np.isfinite(fb) & (fb < 1.5) & (error.ravel() < 25))
            if good.sum() < 12 or good.mean() < 0.5:
                return None
            a, b = points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
            transform, inliers = cv2.estimateAffinePartial2D(
                a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=500,
                confidence=0.99, refineIters=10,
            )
            if transform is None or inliers is None or not np.isfinite(transform).all():
                return None
            inliers = inliers.ravel().astype(bool)
            if inliers.sum() < 10 or inliers.mean() < 0.65:
                return None
            # Require support distributed along the track, not one tiny texture patch.
            rail = np.asarray(track['left_rail'])
            direction = rail[-1] - rail[0]
            span = float(np.linalg.norm(direction))
            if span < 1:
                return None
            if np.ptp(a[inliers] @ (direction / span)) < span * 0.35:
                return None
            scale = float(np.hypot(transform[0, 0], transform[1, 0]))
            angle = abs(float(np.arctan2(transform[1, 0], transform[0, 0])))
            if not 0.90 <= scale <= 1.10 or angle > 0.18:
                return None
            if float(np.median(np.linalg.norm(b[inliers] - a[inliers], axis=1))) > self.roi_margin * 0.65:
                return None
            updated = {}
            for key in ('left_rail', 'right_rail'):
                local = np.asarray(track[key]) - [x0, y0]
                projected = local @ transform[:, :2].T + transform[:, 2] + [x0, y0]
                h, w = gray.shape
                if (projected < 0).any() or (projected[:, 0] >= w).any() or (projected[:, 1] >= h).any():
                    return None
                updated[key] = projected.tolist()
            if not self._valid_size(updated, gray.shape):
                return None
            updated.update(confidence=None, confidence_method=None)
            propagated.append(updated)
            qualities.append(float(good.mean() * inliers.mean()))
        self.diagnostics['motion_support'] = min(qualities) if qualities else None
        return propagated

    def _verify(self, frame, predicted):
        output = []
        rois = []
        self.diagnostics['rois'] = rois
        for track in predicted:
            x0, y0, x1, y1 = roi_of(track, frame.shape, self.roi_margin)
            rois.append([int(v) for v in (x0, y0, x1, y1)])
            found = [offset_track(item, x0, y0) for item in self._detect(frame[y0:y1, x0:x1], frame.shape[:2])]
            found = [item for item in found if self._valid_size(item, frame.shape[:2])]
            ranked = sorted([(match_distance(track, item), i) for i, item in enumerate(found)])
            # Keep association tight; do not silently switch to an adjacent track.
            if not ranked or ranked[0][0] > 12:
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
        self.diagnostics = dict(mode='search_wait', reason=None, detector_calls=0,
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
                if reason or due_full or self.sequence - self.last_search >= self.search_every:
                    tracks = self._search(frame, reason or ('periodic_refresh' if self.tracks else 'acquire'))
                else:
                    tracks = []
                    self.diagnostics['reason'] = 'search_cooldown'
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
                    if self.sequence - self.last_search >= self.search_every:
                        tracks = self._search(frame, 'tracking_lost')
                    else:
                        tracks = []
                        self.diagnostics.update(mode='lost', reason='tracking_lost_search_cooldown')
            self.tracks = tracks
            self.previous_gray = gray if tracks else None
            self.previous_timestamp = timestamp_seconds if tracks else None
            self.diagnostics['verification_age_frames'] = self.sequence - self.last_verified if tracks else None
            return tracks
        except Exception:
            self.reset()
            self.diagnostics.update(mode='error', reason='processing_failed')
            raise
