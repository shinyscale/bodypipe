"""Headless integration test: multi-person coordinate pipeline.

Loads GEM-X output for hopeyoudo_60f, applies crop→original camera transform,
runs FK, and verifies positioning, orientation, and stability across all frames.

Run: cd ~/bodypipe && source .venv/bin/activate && python -m pytest tests/test_coordinate_pipeline.py -v
"""
import sys
import json
from pathlib import Path

import numpy as np
import pytest

# Add GVHMR/GEM-X paths for torch + soma imports
GVHMR_ROOT = Path("/home/zacharymandrews/GVHMR")
GEMX_ROOT = Path("/home/zacharymandrews/GEM-X")
for p in [
    str(GVHMR_ROOT),
    str(GVHMR_ROOT / ".venv/lib/python3.12/site-packages"),
    str(GEMX_ROOT / ".venv/lib/python3.12/site-packages"),
    str(GEMX_ROOT),
    str(Path("/home/zacharymandrews/bodypipe")),
]:
    if p not in sys.path:
        sys.path.append(p)

# Skip entire module if torch unavailable (CI without GPU)
torch = pytest.importorskip("torch")

# Patch torch.load for weights_only
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from app_window import AppWindow
from models.session import PersonTrack
from views.mesh_viewport import MeshViewport


VID_W, VID_H = 1920, 1080
BASE = Path("/home/zacharymandrews/GVHMR/outputs/multi_person/hopeyoudo_60f")
BASE_120 = Path("/home/zacharymandrews/GVHMR/outputs/multi_person/hopeyoudo_120f")
BASE_120_02 = Path("/home/zacharymandrews/GVHMR/outputs/multi_person/hopeyoudo_120f_02")

# Skip if test data doesn't exist
pytestmark = pytest.mark.skipif(
    not (BASE / "person_0/isolated_video/hpe_results.pt").exists(),
    reason="GEM-X test data not available",
)


def load_person_params(pid: int) -> dict:
    """Load GEM-X output and apply crop→original camera transform."""
    from workers.gemx_worker import load_gemx_soma_output

    pdir = BASE / f"person_{pid}"
    params = load_gemx_soma_output(pdir)
    assert params is not None, f"Failed to load person {pid}"

    # Apply crop→original transform (same math as app_window._crop_to_original_camera)
    meta = json.loads((pdir / "person_meta.json").read_text())
    crop_bbox = meta["crop_bbox"]
    x1, y1 = float(crop_bbox[0]), float(crop_bbox[1])
    cx_orig, cy_orig = VID_W / 2.0, VID_H / 2.0
    f_orig = float(max(VID_W, VID_H))

    K = np.asarray(params["K_fullimg"])
    if K.ndim == 3:
        K = K[0]
    f_crop_x, f_crop_y = float(K[0, 0]), float(K[1, 1])
    cx_crop, cy_crop = float(K[0, 2]), float(K[1, 2])

    tr = np.asarray(params["transl"], dtype=np.float32).copy()
    X, Y, Z = tr[:, 0], tr[:, 1], tr[:, 2]
    tr[:, 0] = (f_crop_x * X + (cx_crop + x1 - cx_orig) * Z) / f_orig
    tr[:, 1] = (f_crop_y * Y + (cy_crop + y1 - cy_orig) * Z) / f_orig
    params["transl"] = tr

    return params


@pytest.fixture(scope="module")
def person_params():
    """Load and transform both persons (cached for all tests in module)."""
    return {0: load_person_params(0), 1: load_person_params(1)}


class TestMultiPersonPositioning:
    """Verify two persons are correctly separated after crop→original transform."""

    def test_x_separation(self, person_params):
        """Persons should be separated in X (not stacked on top of each other)."""
        t0 = person_params[0]["transl"]
        t1 = person_params[1]["transl"]
        x_sep = abs(t0[0, 0] - t1[0, 0])
        assert x_sep > 0.1, f"X separation too small: {x_sep:.3f}m"

    def test_x_separation_stable(self, person_params):
        """X separation should be roughly constant across frames (same scene)."""
        t0 = person_params[0]["transl"]
        t1 = person_params[1]["transl"]
        x_seps = np.abs(t0[:, 0] - t1[:, 0])
        assert x_seps.std() < 0.1, f"X separation unstable: std={x_seps.std():.3f}"

    def test_z_depth_positive(self, person_params):
        """All Z (depth) values should be positive and reasonable."""
        for pid in [0, 1]:
            Z = person_params[pid]["transl"][:, 2]
            assert np.all(Z > 0), f"Person {pid} has non-positive Z"
            assert Z.min() > 0.5, f"Person {pid} Z too small: {Z.min():.2f}"
            assert Z.max() < 10.0, f"Person {pid} Z too large: {Z.max():.2f}"

    def test_no_nan_or_inf(self, person_params):
        """No NaN or Inf in transformed translations."""
        for pid in [0, 1]:
            tr = person_params[pid]["transl"]
            assert np.all(np.isfinite(tr)), f"Person {pid} has non-finite transl"


class TestSkeletonOrientation:
    """Verify FK produces correctly oriented skeletons in camera space."""

    def test_head_above_pelvis_camera_space(self, person_params):
        """In camera space (Y-down), head Y should be < pelvis Y (higher in image)."""
        from views.mesh_viewport import _forward_kinematics_soma

        for pid in [0, 1]:
            for frame in [0, 15, 30, 45, 59]:
                joints = _forward_kinematics_soma(person_params[pid], frame)
                pelvis_y = joints[0, 1]
                head_y = joints[15, 1]
                assert head_y < pelvis_y, (
                    f"Person {pid} frame {frame}: head not above pelvis "
                    f"(head_y={head_y:.3f}, pelvis_y={pelvis_y:.3f})"
                )

    def test_head_above_pelvis_after_cvgl_flip(self, person_params):
        """After CV→GL flip (Y *= -1), head Y should be > pelvis Y (GL Y-up)."""
        from views.mesh_viewport import _forward_kinematics_soma

        for pid in [0, 1]:
            joints = _forward_kinematics_soma(person_params[pid], 0)
            # CV→GL: Y *= -1
            pelvis_y_gl = -joints[0, 1]
            head_y_gl = -joints[15, 1]
            assert head_y_gl > pelvis_y_gl, (
                f"Person {pid}: head not above pelvis in GL space"
            )

    def test_orientation_stable_across_frames(self, person_params):
        """Global orient should change smoothly (no sudden flips)."""
        for pid in [0, 1]:
            go = np.asarray(person_params[pid]["global_orient"])
            for i in range(1, len(go)):
                delta = np.linalg.norm(go[i] - go[i - 1])
                assert delta < 1.0, (
                    f"Person {pid} frame {i}: orientation jump of {delta:.3f} rad"
                )


class TestConfidence:
    """Verify GEM-X confidence values are reasonable."""

    def test_confidence_range(self, person_params):
        """Confidence should be in [0, 1]."""
        for pid in [0, 1]:
            conf = person_params[pid].get("confidences")
            if conf is None:
                pytest.skip("No confidence data")
            assert np.all(conf >= 0) and np.all(conf <= 1)

    def test_confidence_not_all_low(self, person_params):
        """At least some frames should have confidence > 0.3."""
        for pid in [0, 1]:
            conf = person_params[pid].get("confidences")
            if conf is None:
                pytest.skip("No confidence data")
            pct_above = np.mean(conf > 0.3) * 100
            assert pct_above > 20, (
                f"Person {pid}: only {pct_above:.0f}% frames > 0.3 confidence"
            )


class TestSomaForwardPass:
    """Verify SOMA body model produces valid vertices."""

    @pytest.fixture(scope="class")
    def soma_model(self):
        try:
            from gem.utils.soma_utils.soma_layer import SomaLayer
            from soma.assets import get_assets_dir
            model = SomaLayer(
                data_root=str(get_assets_dir()),
                low_lod=True, device="cpu",
                identity_model_type="soma", mode="dense",
            )
            model.eval()
            return model
        except Exception as e:
            pytest.skip(f"SOMA model unavailable: {e}")

    def test_vertices_finite(self, person_params, soma_model):
        """SOMA forward pass should produce finite vertices."""
        params = person_params[0]
        poses = torch.tensor(params["poses"][:1], dtype=torch.float32)
        transl = torch.tensor(params["transl"][:1], dtype=torch.float32)
        ic = torch.zeros(1, 128)
        sp = torch.ones(1, 1)

        with torch.no_grad():
            out = soma_model.static_forward(poses, ic, sp, transl, pose2rot=True)

        verts = out["vertices"].numpy()
        assert np.all(np.isfinite(verts)), "Non-finite vertices"
        assert verts.shape == (1, 4505, 3) or verts.shape[1] > 1000

    def test_vertices_consistent_across_persons(self, person_params, soma_model):
        """Both persons should produce similar-scale vertices (same body model)."""
        scales = []
        for pid in [0, 1]:
            params = person_params[pid]
            poses = torch.tensor(params["poses"][:1], dtype=torch.float32)
            transl = torch.tensor(params["transl"][:1], dtype=torch.float32)
            ic = torch.zeros(1, 128)
            sp = torch.ones(1, 1)
            with torch.no_grad():
                out = soma_model.static_forward(poses, ic, sp, transl, pose2rot=True)
            verts = out["vertices"][0].numpy()
            extent = np.max(np.abs(verts - verts.mean(axis=0)))
            scales.append(extent)

        ratio = max(scales) / min(scales)
        assert ratio < 2.0, f"Vertex scale mismatch: ratio={ratio:.2f}"


@pytest.mark.skipif(
    not (BASE_120 / "person_0").is_dir(),
    reason="120f viewport regression data not available",
)
class TestViewportGroundingRegression:
    """Regression coverage for the known failing 120-frame multi-person clip."""

    @pytest.fixture(scope="class")
    def viewport_session(self, qapp):
        window = AppWindow()
        window._session.img_width = VID_W
        window._session.img_height = VID_H
        window._session.output_dir = BASE_120
        window._session.person_tracks.clear()

        for pid in [0, 1]:
            person_dir = BASE_120 / f"person_{pid}"
            smplx_params, soma_params, body_model_type = window._load_motion_params(person_dir)
            params = soma_params if soma_params is not None else smplx_params
            assert params is not None, f"Failed to load viewport params for person {pid}"
            window._session.person_tracks[pid] = PersonTrack(
                person_id=pid,
                person_dir=person_dir,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
            )

        yield window._session
        window.close()

    def test_orbit_roots_preserve_multi_person_separation(self, qapp, viewport_session):
        viewport = MeshViewport()
        viewport.resize(960, 540)
        viewport.set_session(viewport_session)
        viewport.set_person(0)
        viewport.set_camera_mode("orbit")
        viewport.on_frame_changed(0)

        assert 0 in viewport._all_joint_positions
        assert 1 in viewport._all_joint_positions

        for frame_idx in [0, 30, 60, 90, 119]:
            viewport.on_frame_changed(frame_idx)
            roots = {
                pid: joints[0]
                for pid, joints in viewport._all_joint_positions.items()
            }
            sep = np.linalg.norm(roots[0][[0, 2]] - roots[1][[0, 2]])
            assert sep > 0.25, f"Frame {frame_idx}: persons too close in orbit view ({sep:.3f}m)"

    def test_orbit_world_grounding_keeps_mesh_near_zero(self, qapp, viewport_session):
        viewport = MeshViewport()
        viewport.resize(960, 540)
        viewport.set_session(viewport_session)
        viewport.set_camera_mode("orbit")

        for pid in [0, 1]:
            viewport.set_person(pid)
            verts_result = viewport._compute_vertices(pid, 0)
            if verts_result is None:
                pytest.skip("Mesh model unavailable for grounding regression")
            verts, _normals = verts_result
            mesh_min_y = float(np.min(verts[:, 1]))
            assert abs(mesh_min_y) < 0.08, (
                f"Person {pid}: mesh not grounded in orbit view (min_y={mesh_min_y:.3f}m)"
            )

    def test_orbit_skeleton_ground_height_tracks_mesh(self, qapp, viewport_session):
        viewport = MeshViewport()
        viewport.resize(960, 540)
        viewport.set_session(viewport_session)
        viewport.set_camera_mode("orbit")

        for pid in [0, 1]:
            viewport.set_person(pid)
            viewport.on_frame_changed(0)
            verts_result = viewport._compute_vertices(pid, 0)
            if verts_result is None:
                pytest.skip("Mesh model unavailable for grounding regression")
            joints = viewport._joint_positions
            assert joints is not None
            verts, _normals = verts_result
            mesh_min_y = float(np.min(verts[:, 1]))
            skel_min_y = float(np.min(joints[:, 1]))
            assert abs(skel_min_y - mesh_min_y) < 0.15, (
                f"Person {pid}: skeleton/mesh grounding mismatch "
                f"(skel_min_y={skel_min_y:.3f}m, mesh_min_y={mesh_min_y:.3f}m)"
            )

    def test_orbit_preserves_head_above_pelvis_without_sign_flip(self, qapp, viewport_session):
        viewport = MeshViewport()
        viewport.resize(960, 540)
        viewport.set_session(viewport_session)
        viewport.set_camera_mode("orbit")

        for pid in [0, 1]:
            viewport.set_person(pid)
            viewport.on_frame_changed(0)
            joints = viewport._joint_positions
            assert joints is not None
            pelvis_y = joints[0, 1]
            head_y = joints[6, 1]
            assert head_y > pelvis_y, f"Person {pid}: head not above pelvis in orbit/world view"


@pytest.mark.skipif(
    not (BASE_120_02 / "session_manifest.json").is_file(),
    reason="120f_02 manifest regression data not available",
)
class TestViewportManifestOffsetRegression:
    """Regression coverage for session-manifest world offsets on the renamed clip."""

    def test_loaded_tracks_apply_manifest_world_offsets(self, qapp):
        window = AppWindow()
        window._session.img_width = VID_W
        window._session.img_height = VID_H
        window._session.output_dir = BASE_120_02

        raw_loaded = {}
        for pid in [0, 1]:
            person_dir = BASE_120_02 / f"person_{pid}"
            smplx_params, soma_params, _body_model_type = window._load_motion_params(person_dir)
            params = soma_params if soma_params is not None else smplx_params
            assert params is not None
            raw_loaded[pid] = np.asarray(params["transl_world"], dtype=np.float32).copy()

        window._session.person_tracks.clear()
        window._load_results_from_output_dir(BASE_120_02)

        manifest = json.loads((BASE_120_02 / "session_manifest.json").read_text())
        expected_offsets = {}
        for binding in manifest["person_bindings"]:
            expected_offsets[int(binding["source_index"])] = np.asarray(
                manifest["offsets"][str(binding["track_id"])], dtype=np.float32
            )

        for pid in [0, 1]:
            track = window._session.person_tracks[pid]
            params = track.soma_params if track.soma_params is not None else track.smplx_params
            assert params is not None
            loaded_world = np.asarray(params["transl_world"], dtype=np.float32)
            np.testing.assert_allclose(
                loaded_world[0] - raw_loaded[pid][0],
                expected_offsets[pid],
                atol=1e-5,
            )

        window.close()
