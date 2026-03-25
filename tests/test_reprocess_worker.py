"""Tests for reprocess bbox prior generation and track binding lookup."""

import json

import numpy as np

from models.session import PersonTrack
from workers.reprocess_worker import (
    _resolve_person_index,
    build_dense_rerun_bboxes,
    extract_verified_identity_keyframes,
)


def test_build_dense_rerun_bboxes_single_keyframe_applies_global_delta():
    original = np.array(
        [
            [10.0, 20.0, 110.0, 220.0],
            [12.0, 22.0, 112.0, 222.0],
            [14.0, 24.0, 114.0, 224.0],
        ],
        dtype=np.float32,
    )
    corrections = np.zeros_like(original)
    corrections[1] = [22.0, 32.0, 122.0, 232.0]

    updated = build_dense_rerun_bboxes(original, corrections)

    expected_delta = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    np.testing.assert_allclose(updated[0], original[0] + expected_delta, atol=1e-6)
    np.testing.assert_allclose(updated[1], corrections[1], atol=1e-6)
    np.testing.assert_allclose(updated[2], original[2] + expected_delta, atol=1e-6)


def test_resolve_person_index_prefers_person_meta_binding(tmp_path):
    person_dir = tmp_path / "person_0"
    person_dir.mkdir()
    (person_dir / "person_meta.json").write_text(
        json.dumps({"source_index": 1, "track_id": 20})
    )

    track = PersonTrack(person_id=0, person_dir=person_dir)
    all_tracks = [
        {"track_id": 10, "bbx_xyxy": np.zeros((2, 4), dtype=np.float32)},
        {"track_id": 20, "bbx_xyxy": np.ones((2, 4), dtype=np.float32)},
    ]

    assert _resolve_person_index(0, track, all_tracks) == 1


def test_extract_verified_identity_keyframes_prefers_corrected_bbox():
    track = PersonTrack(
        person_id=0,
        bboxes=np.array(
            [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]],
            dtype=np.float32,
        ),
        bbox_corrections=np.array(
            [[0.0, 0.0, 0.0, 0.0], [55.0, 65.0, 75.0, 85.0]],
            dtype=np.float32,
        ),
        keyframes=[
            {"frame": 0, "verified": False},
            {"frame": 1, "verified": True},
        ],
    )

    result = extract_verified_identity_keyframes(track)

    assert result == [
        {"frame": 1, "bbox": [55.0, 65.0, 75.0, 85.0], "verified": True}
    ]
