"""Conservative live-input checks, separate from sealed model scores.

A failed check means insufficient evidence, never a claim that the scene is safe.
These engineering thresholds are not calibrated event probabilities.
"""
from __future__ import annotations

import cv2
import numpy as np

GUARD_VERSION = 'live-input-v3'

# Engineering evidence requirements, not calibrated probabilities. Count real
# instants, not persistent track IDs: tracks often fragment during occlusion.
FIGHT_PEOPLE_POLICY = dict(min_pair_frames=3, min_pair_span_seconds=.2,
                           max_pair_gap_seconds=.75, max_gap_in_person_diagonals=.5)


def assess_view(frames):
    """Detect severe loss of detail and spatially distributed camera movement."""
    unique = sorted({float(f[0]): f for f in frames}.values(), key=lambda f: f[0])
    gray = []
    for _, image, *_ in unique:
        h, w = image.shape[:2]
        small = cv2.resize(image, (320, max(1, round(h * 320 / w))))
        gray.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    if len(gray) < 2:
        return dict(ok=False, reason='连续画面不足，无法判断', version=GUARD_VERSION)
    detail = [float(cv2.Laplacian(im, cv2.CV_32F).var()) for im in gray]
    poor = [im.mean() < 8 or im.std() < 5 or sharpness < 8 for im, sharpness in zip(gray, detail)]
    diagnostics = dict(version=GUARD_VERSION, poor_frames=int(sum(poor)), frames=len(gray))
    if sum(poor) >= max(2, len(gray) // 4):
        return dict(ok=False, reason='画面过暗、模糊或被遮挡，请固定镜头并检查采集画质', **diagnostics)

    motions = []
    unresolved_changes = []
    for previous, current in zip(gray, gray[1:]):
        # Large discontinuities can defeat optical flow altogether. Remove a
        # uniform lighting shift before checking how much of the scene changed.
        delta = cv2.GaussianBlur(current, (5, 5), 0).astype(np.float32) - cv2.GaussianBlur(previous, (5, 5), 0)
        changed = float(np.mean(np.abs(delta - np.median(delta)) > 25))
        unresolved_changes.append(changed)
        points = cv2.goodFeaturesToTrack(previous, maxCorners=180, qualityLevel=.02, minDistance=8)
        if points is None or len(points) < 24:
            continue
        target, status, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        if target is None:
            continue
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, target, None)
        if back is None:
            continue
        valid = ((status[:, 0] == 1) & (back_status[:, 0] == 1) & (error[:, 0] < 30)
                 & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.5))
        a, b = points[valid, 0], target[valid, 0]
        if len(a) < 20:
            continue
        matrix, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
        if matrix is None or mask.mean() < .7:
            continue
        inliers = a[mask[:, 0] > 0]
        h, w = previous.shape
        # A moving arm/person in one small region must not count as camera motion.
        quadrants = (inliers[:, 0] >= w / 2).astype(int) + 2 * (inliers[:, 1] >= h / 2)
        if len(np.unique(quadrants)) < 3 or cv2.contourArea(cv2.convexHull(inliers)) < .2 * w * h:
            continue
        center = np.array([w / 2, h / 2])
        shift = np.linalg.norm(matrix[:, :2] @ center + matrix[:, 2] - center) / np.hypot(w, h)
        angle = abs(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        zoom = abs(np.hypot(matrix[0, 0], matrix[1, 0]) - 1)
        motions.append((float(shift), float(angle), float(zoom)))
        unresolved_changes[-1] = 0
    diagnostics['motion_pairs'] = len(motions)
    diagnostics['max_rotation_degrees'] = round(max((m[1] for m in motions), default=0), 2)
    diagnostics['max_shift_fraction'] = round(max((m[0] for m in motions), default=0), 4)
    diagnostics['unresolved_scene_changes'] = sum(value > .45 for value in unresolved_changes)
    large = any(shift > .025 or angle > 2.5 or zoom > .035 for shift, angle, zoom in motions)
    sustained = sum(shift > .006 or angle > .6 or zoom > .01 for shift, angle, zoom in motions) >= 4
    if large or sustained:
        return dict(ok=False, reason='镜头正在移动或旋转，暂停动作判断；固定后等待4秒连续画面', **diagnostics)
    if diagnostics['unresolved_scene_changes'] >= 2 or max(unresolved_changes, default=0) > .75:
        return dict(ok=False, reason='画面发生大幅变化，无法确认镜头稳定；请固定后等待4秒', **diagnostics)
    return dict(ok=True, reason=None, **diagnostics)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def single_person_only(clip):
    """Veto only a consistently visible singleton, not fragmented/occluded pairs.

    Requiring two persistent tracks caused substantial recall loss in validation.
    Count people per real instant instead; any credible second person keeps the
    RGB classifier eligible. Lack of pose observations is not proof of solitude.
    """
    points = _array(clip['keypoints'])
    indices = np.asarray(clip['frame_indices'])
    times = np.asarray(clip['timestamps'])
    valid = (points[:, :, 5:, 2] >= .3).sum(-1) >= 4
    _, positions = np.unique(indices, return_index=True)
    counts = valid[:, positions].sum(0)
    return bool(len(positions) >= 8 and np.ptp(times[positions]) >= 2
                and np.max(counts, initial=0) == 1 and np.mean(counts == 1) >= .8)


def fight_people_evidence(clip):
    """Require repeated, nearby distinct bodies before an RGB fight candidate.

    Missing poses never imply multiple people. A remote false person (e.g. a
    cushion) cannot unlock the classifier. Proximity is only a prerequisite,
    not proof of contact, violence, or even that every detector box is a person.
    All failures mean unknown; occluded/off-screen fights may be missed.
    """
    points = _array(clip['keypoints'])
    boxes = _array(clip['boxes'])
    indices = np.asarray(clip['frame_indices'])
    times = np.asarray(clip['timestamps'], dtype=float)
    _, positions = np.unique(indices, return_index=True)
    positions = positions[np.argsort(times[positions])]
    # Also deduplicate timestamps: a caller must not inflate evidence with IDs.
    _, unique = np.unique(times[positions], return_index=True)
    positions = positions[np.sort(unique)]
    counts, pair_flags = [], []
    rejected, duplicates = 0, 0
    for position in positions:
        people = []
        for p, box in zip(points[:, position], boxes[:, position]):
            if not np.isfinite(box).all():
                rejected += 1
                continue
            size = box[2:] - box[:2]
            if np.any(size <= 0):
                # Empty track slots are not detections.
                continue
            inside = ((p[:, :2] >= box[:2] - .05 * size)
                      & (p[:, :2] <= box[2:] + .05 * size)).all(-1)
            visible = np.isfinite(p).all(-1) & (p[:, 2] >= .3) & inside
            if visible[5:].sum() < 4 or visible[[5, 6, 11, 12]].sum() < 2:
                rejected += 1
                continue
            people.append((p, box, visible, float(np.linalg.norm(size))))
        # Suppress duplicate poses only when both boxes AND common joints agree.
        # Merely overlapping boxes must not erase grappling/occluded people.
        distinct = []
        for person in sorted(people, key=lambda x: float(x[0][x[2], 2].mean()), reverse=True):
            p, box, visible, diagonal = person
            duplicate = False
            for q, other, other_visible, other_diagonal in distinct:
                common = visible & other_visible
                common[:5] = False
                overlap = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
                intersection = float(np.prod(overlap))
                union = float(np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - intersection)
                if (common.sum() >= 4 and intersection / max(union, 1) > .65
                        and np.median(np.linalg.norm(p[common, :2] - q[common, :2], axis=1))
                        < .035 * min(diagonal, other_diagonal)):
                    duplicate = True
                    break
            if duplicate:
                duplicates += 1
            else:
                distinct.append(person)
        counts.append(len(distinct))
        nearby = False
        for i, (_, box, _, diagonal) in enumerate(distinct):
            for _, other, _, other_diagonal in distinct[i + 1:]:
                gap = np.maximum(0, np.maximum(box[:2] - other[2:], other[:2] - box[2:]))
                if np.linalg.norm(gap) <= FIGHT_PEOPLE_POLICY['max_gap_in_person_diagonals'] * min(diagonal, other_diagonal):
                    nearby = True
                    break
            if nearby:
                break
        pair_flags.append(nearby)
    real_times = times[positions]
    pair_times = real_times[np.asarray(pair_flags, dtype=bool)]
    runs = np.split(pair_times, np.flatnonzero(np.diff(pair_times) > FIGHT_PEOPLE_POLICY['max_pair_gap_seconds']) + 1)
    supported = [r for r in runs if len(r) >= FIGHT_PEOPLE_POLICY['min_pair_frames']
                 and r[-1] - r[0] + 1e-6 >= FIGHT_PEOPLE_POLICY['min_pair_span_seconds']]
    finite_times = bool(np.isfinite(real_times).all())
    eligible = bool(finite_times and len(positions) >= 8 and supported)
    maximum = max(counts, default=0)
    if eligible:
        code, reason = 'nearby_people', None
    elif not finite_times or len(positions) < 8:
        code, reason = 'insufficient_observations', '独立采样画面不足，无法核实多人互动'
    elif maximum < 2:
        code, reason = 'no_pair', '未获得可靠的两人同框证据，暂不判为打架；遮挡或画面外事件无法排除'
    elif not len(pair_times):
        code, reason = 'no_nearby_pair', '检测到的人体位置相隔较远，缺少近距离互动依据，暂不判为打架'
    else:
        code, reason = 'unstable_pair', '多人接近证据过短或不连续，可能存在误检，暂不判为打架'
    return dict(eligible=eligible, code=code, reason=reason, version=GUARD_VERSION,
                policy=dict(FIGHT_PEOPLE_POLICY), unique_frames=len(positions),
                timestamps=[round(float(t), 4) if np.isfinite(t) else None for t in real_times],
                people_per_frame=counts, nearby_pair_per_frame=pair_flags,
                max_people=maximum, pair_frames=int(len(pair_times)),
                longest_pair_run=max((len(r) for r in runs), default=0),
                rejected_pose_detections=rejected, duplicate_pose_detections=duplicates)


def rising_only_tracks(clip):
    """Identify clear ground-to-standing motion without a preceding descent.

    This veto is deliberately narrow. Missing joints or ambiguous movement are
    not inferred as rising. A real descent followed by recovery stays eligible.
    """
    points = _array(clip['keypoints'])
    boxes = _array(clip['boxes'])
    indices = np.asarray(clip['frame_indices'])
    times = np.asarray(clip['timestamps'])
    rising = []
    for track, bounds in zip(points, boxes):
        usable = (track[:, [5, 6, 11, 12], 2] >= .3).all(-1)
        unique = []
        seen = set()
        for i in np.flatnonzero(usable):
            if indices[i] not in seen:
                unique.append(i)
                seen.add(indices[i])
        if len(unique) < 8 or np.ptp(times[unique]) < 1.5:
            rising.append(False)
            continue
        k = track[unique]
        heights = bounds[unique, 3] - bounds[unique, 1]
        scale = max(1, float(np.quantile(heights, .8)))
        hips = k[:, [11, 12], :2].mean(1)
        shoulders = k[:, [5, 6], :2].mean(1)
        # Median over three real observations prevents a bad pose from inventing
        # a descent or rise. Downward image motion has positive y.
        y = np.array([np.median(hips[max(0, i - 1):i + 2, 1]) for i in range(len(hips))])
        downward = max((y[i + 1:].max() - y[i] for i in range(len(y) - 1)), default=0)
        rise = np.median(y[:3]) - np.median(y[-3:])
        torso = hips[-3:] - shoulders[-3:]
        upright = np.count_nonzero((torso[:, 1] > .12 * scale)
                                   & (np.abs(torso[:, 0]) < torso[:, 1] * .8)) >= 2
        rising.append(bool(rise > .18 * scale and downward < .12 * scale and upright))
    return rising


def exclude_tracks(clip, excluded):
    """Keep classifier scores attached to the same tracks as the input checks."""
    keep = [i for i, value in enumerate(excluded) if not value]
    return {**clip, 'keypoints': clip['keypoints'][keep], 'boxes': clip['boxes'][keep],
            'track_ids': [clip['track_ids'][i] for i in keep]}
