"""Tests for reprocess bbox prior generation and track binding lookup."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from models.session import PersonTrack, Session
from workers.reprocess_worker import (
    _resolve_person_index,
    build_dense_rerun_bboxes,
    extract_manual_bbox_keyframes,
    extract_verified_identity_keyframes,
    ReprocessWorker,
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


def test_extract_manual_bbox_keyframes_collects_sparse_edits():
    track = PersonTrack(
        person_id=0,
        bbox_corrections=np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0, 20.0, 30.0, 40.0],
                [0.0, 0.0, 0.0, 0.0],
                [50.0, 60.0, 70.0, 80.0],
            ],
            dtype=np.float32,
        ),
    )

    result = extract_manual_bbox_keyframes(track)

    assert sorted(result) == [1, 3]
    np.testing.assert_allclose(result[1], [10.0, 20.0, 30.0, 40.0], atol=1e-6)
    np.testing.assert_allclose(result[3], [50.0, 60.0, 70.0, 80.0], atol=1e-6)


def test_reprocess_worker_persists_regenerated_tracks(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    output_dir = tmp_path / "out"
    detection_dir = output_dir / "detection"
    person_dir = output_dir / "person_0"
    detection_dir.mkdir(parents=True)
    person_dir.mkdir()
    (person_dir / "person_meta.json").write_text(
        json.dumps({"source_index": 0, "track_id": 20})
    )

    original_boxes = np.array(
        [[10.0, 20.0, 30.0, 40.0], [12.0, 22.0, 32.0, 42.0]],
        dtype=np.float32,
    )
    regenerated_boxes = np.array(
        [[50.0, 60.0, 70.0, 80.0], [52.0, 62.0, 72.0, 82.0]],
        dtype=np.float32,
    )
    torch.save(
        {
            "tracks": [
                {
                    "track_id": 20,
                    "bbx_xyxy": torch.from_numpy(original_boxes),
                    "detection_mask": torch.tensor([True, True]),
                    "detection_conf": torch.tensor([0.8, 0.9]),
                }
            ]
        },
        detection_dir / "all_tracks.pt",
    )

    captured = {}

    def fake_reprocess_person(**kwargs):
        captured["manual_bbox_keyframes"] = kwargs["manual_bbox_keyframes"]
        captured["verified_identity_keyframes"] = kwargs["verified_identity_keyframes"]
        kwargs["all_tracks"][kwargs["person_index"]]["bbx_xyxy"] = torch.from_numpy(
            regenerated_boxes
        )
        return {"pt_path": str(person_dir / "demo" / "hmr4d_results.pt")}

    monkeypatch.setitem(
        sys.modules,
        "multi_person_split",
        SimpleNamespace(
            reprocess_person=fake_reprocess_person,
            _merge_duplicate_tracks=lambda tracks: (tracks, []),
            _stitch_fragmented_tracks=lambda tracks: (tracks, []),
        ),
    )

    session = Session(
        video_path=tmp_path / "video.mp4",
        output_dir=output_dir,
    )
    session.person_tracks[0] = PersonTrack(
        person_id=0,
        person_dir=person_dir,
        body_model_type="smplx",
        bboxes=original_boxes.copy(),
        original_bboxes=original_boxes.copy(),
        bbox_corrections=np.array(
            [[0.0, 0.0, 0.0, 0.0], [14.0, 24.0, 34.0, 44.0]],
            dtype=np.float32,
        ),
        keyframes=[{"frame": 1, "verified": True}],
    )

    worker = ReprocessWorker(session=session, person_ids=[0])
    worker.run()

    saved = torch.load(detection_dir / "all_tracks.pt", map_location="cpu", weights_only=False)
    np.testing.assert_allclose(
        saved["tracks"][0]["bbx_xyxy"].numpy(), regenerated_boxes, atol=1e-6
    )
    assert sorted(captured["manual_bbox_keyframes"]) == [1]
    assert captured["verified_identity_keyframes"] == [
        {"frame": 1, "bbox": [14.0, 24.0, 34.0, 44.0], "verified": True}
    ]


def test_reprocess_worker_routes_soma_tracks_to_gemx_backend(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    output_dir = tmp_path / "out"
    detection_dir = output_dir / "detection"
    person_dir = output_dir / "person_0"
    detection_dir.mkdir(parents=True)
    person_dir.mkdir()
    (person_dir / "person_meta.json").write_text(
        json.dumps({"source_index": 0, "track_id": 30})
    )

    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    torch.save(
        {
            "tracks": [
                {
                    "track_id": 30,
                    "bbx_xyxy": torch.from_numpy(boxes),
                    "detection_mask": torch.tensor([True]),
                    "detection_conf": torch.tensor([0.9]),
                }
            ]
        },
        detection_dir / "all_tracks.pt",
    )

    captured = {}

    def fake_reprocess_person(**kwargs):
        captured["estimation_backend"] = kwargs["estimation_backend"]
        return {"pt_path": str(person_dir / "gemx_demo" / "hpe_results.pt")}

    monkeypatch.setitem(
        sys.modules,
        "multi_person_split",
        SimpleNamespace(
            reprocess_person=fake_reprocess_person,
            _merge_duplicate_tracks=lambda tracks: (tracks, []),
            _stitch_fragmented_tracks=lambda tracks: (tracks, []),
        ),
    )

    session = Session(video_path=tmp_path / "video.mp4", output_dir=output_dir)
    session.person_tracks[0] = PersonTrack(
        person_id=0,
        person_dir=person_dir,
        body_model_type="soma",
        bboxes=boxes.copy(),
    )

    worker = ReprocessWorker(session=session, person_ids=[0])
    worker.run()

    assert captured["estimation_backend"] == "gemx"


def test_reprocess_worker_merges_duplicate_cached_tracks_before_resolve(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    output_dir = tmp_path / "out"
    detection_dir = output_dir / "detection"
    person_dir = output_dir / "person_0"
    detection_dir.mkdir(parents=True)
    person_dir.mkdir()
    (person_dir / "person_meta.json").write_text(
        json.dumps({"track_id": 30})
    )

    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    torch.save(
        {
            "tracks": [
                {"track_id": 10, "bbx_xyxy": torch.from_numpy(boxes)},
                {"track_id": 30, "bbx_xyxy": torch.from_numpy(boxes)},
            ]
        },
        detection_dir / "all_tracks.pt",
    )

    captured = {}

    def fake_reprocess_person(**kwargs):
        captured["person_index"] = kwargs["person_index"]
        captured["num_tracks"] = len(kwargs["all_tracks"])
        return {"pt_path": str(person_dir / "demo" / "hmr4d_results.pt")}

    monkeypatch.setitem(
        sys.modules,
        "multi_person_split",
        SimpleNamespace(
            reprocess_person=fake_reprocess_person,
            _merge_duplicate_tracks=lambda tracks: (tracks[1:], [(30, 10, 42)]),
            _stitch_fragmented_tracks=lambda tracks: (tracks, []),
        ),
    )

    session = Session(video_path=tmp_path / "video.mp4", output_dir=output_dir)
    session.person_tracks[30] = PersonTrack(
        person_id=30,
        person_dir=person_dir,
        body_model_type="smplx",
        bboxes=boxes.copy(),
    )

    worker = ReprocessWorker(session=session, person_ids=[30])
    worker.run()

    assert captured["num_tracks"] == 1
    assert captured["person_index"] == 0


def test_reprocess_worker_stitches_sequential_cached_tracks_before_resolve(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    output_dir = tmp_path / "out"
    detection_dir = output_dir / "detection"
    person_dir = output_dir / "person_0"
    detection_dir.mkdir(parents=True)
    person_dir.mkdir()
    (person_dir / "person_meta.json").write_text(json.dumps({"track_id": 30}))

    boxes = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    torch.save(
        {
            "tracks": [
                {"track_id": 10, "bbx_xyxy": torch.from_numpy(boxes)},
                {"track_id": 30, "bbx_xyxy": torch.from_numpy(boxes)},
            ]
        },
        detection_dir / "all_tracks.pt",
    )

    captured = {}

    def fake_reprocess_person(**kwargs):
        captured["person_index"] = kwargs["person_index"]
        captured["num_tracks"] = len(kwargs["all_tracks"])
        return {"pt_path": str(person_dir / "demo" / "hmr4d_results.pt")}

    monkeypatch.setitem(
        sys.modules,
        "multi_person_split",
        SimpleNamespace(
            reprocess_person=fake_reprocess_person,
            _merge_duplicate_tracks=lambda tracks: (tracks, []),
            _stitch_fragmented_tracks=lambda tracks: (tracks[1:], [(30, 10, 8)]),
        ),
    )

    session = Session(video_path=tmp_path / "video.mp4", output_dir=output_dir)
    session.person_tracks[30] = PersonTrack(
        person_id=30,
        person_dir=person_dir,
        body_model_type="smplx",
        bboxes=boxes.copy(),
    )

    worker = ReprocessWorker(session=session, person_ids=[30])
    worker.run()

    assert captured["num_tracks"] == 1
    assert captured["person_index"] == 0
