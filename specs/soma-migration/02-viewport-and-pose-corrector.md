# Handoff B: Viewport SOMA Rendering + Pose Corrector Rewrite (Phases 3 + 4)

**Prerequisite**: Handoff A complete (GEM-X worker + SOMA BVH export).
**This is the largest handoff** — touches the two biggest view files and their tests.

---

## Context

The 3D viewport and pose corrector currently hardcode SMPL-X param layout (separate `global_orient`, `body_pose`, `left_hand_pose`, `right_hand_pose` arrays). SOMA uses a unified `poses` tensor of shape `(N, 77, 3)` where joint 0 is global orient. Both components need dual-path support: detect `body_model_type` on the person track and dispatch to the appropriate param format.

**Key invariant**: All existing SMPL-X tests must pass unchanged. New SOMA tests are added alongside, never replacing.

---

## Step 1: Update `forward_kinematics()` in `views/mesh_viewport.py`

The existing `forward_kinematics()` function (line ~338) handles SMPL-X params with separate body/hand arrays. Add SOMA support by detecting the param format.

Find the function signature:

```python
def forward_kinematics(params: dict, frame_idx: int) -> np.ndarray:
```

Add SOMA dispatch at the top of the function body, right after the docstring, before `from scipy.spatial.transform import Rotation`:

```python
    # SOMA path: unified poses tensor (N, J, 3)
    if "poses" in params and "body_pose" not in params:
        return _forward_kinematics_soma(params, frame_idx)
```

Then add the SOMA FK function right before the existing `forward_kinematics`:

```python
def _forward_kinematics_soma(params: dict, frame_idx: int) -> np.ndarray:
    """Compute joint positions from SOMA unified poses tensor.

    Parameters
    ----------
    params : dict with keys 'poses' (N, J, 3), 'transl' (N, 3)
    frame_idx : which frame to compute

    Returns
    -------
    positions : (J, 3) float64 joint positions
    """
    from scipy.spatial.transform import Rotation
    from models.skeleton import SOMA_SKELETON

    poses = np.asarray(params["poses"])
    transl = np.asarray(params.get("transl", np.zeros((poses.shape[0], 3))))

    n_joints = poses.shape[1]
    skel = SOMA_SKELETON
    parents = skel.joint_parents
    offsets_dict = skel.default_offsets
    names = skel.joint_names

    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    offsets = np.zeros((n_joints, 3))
    for i in range(min(n_joints, len(names))):
        offsets[i] = offsets_dict.get(names[i], [0, 0, 0])

    # Root
    root_aa = poses[frame_idx, 0] if poses.ndim == 3 else poses[0]
    accumulated_R[0] = Rotation.from_rotvec(root_aa.ravel()[:3]).as_matrix()
    positions[0] = transl[frame_idx] if transl.ndim >= 2 else transl

    for j in range(1, n_joints):
        parent = parents[j] if j < len(parents) else 0

        rot_aa = poses[frame_idx, j] if poses.ndim == 3 and frame_idx < poses.shape[0] else np.zeros(3)
        R_local = Rotation.from_rotvec(np.asarray(rot_aa).ravel()[:3]).as_matrix()
        accumulated_R[j] = accumulated_R[parent] @ R_local
        positions[j] = positions[parent] + accumulated_R[parent] @ offsets[j]

    return positions
```

### Update `compute_joint_colors()` for 77 joints

Find `compute_joint_colors` (line ~754). The existing joint_map handles joints up to 51 (+ jaw/eyes mapped to Head). Extend for SOMA face joints (52-76):

Replace the existing mapping loop:

```python
    joint_map = np.zeros(J, dtype=np.int32)
    for j in range(J):
        if j < 22:
            joint_map[j] = j
        elif j < 37:
            joint_map[j] = 20  # L_Wrist
        elif j < 52:
            joint_map[j] = 21  # R_Wrist
        else:
            joint_map[j] = 15  # Head (jaw/eyes)
```

This mapping already handles joints beyond 51 by mapping to Head (15). No change needed — it works for SOMA-77 as-is since face joints (52-76) map to Head color.

---

## Step 2: Update `_get_joint_axis_angle()` in `views/pose_corrector_panel.py`

Find `_get_joint_axis_angle` (line ~215). Add SOMA dispatch after the correction track check, before the raw params fallback.

After this line:

```python
    params = track.smplx_params
```

Add:

```python
    # SOMA path: unified poses tensor
    if track.body_model_type == "soma" and track.soma_params is not None:
        return _get_soma_joint_axis_angle(track.soma_params, frame_idx, joint_idx)

    if params is None:
        return np.zeros(3, dtype=np.float32)
```

And remove the existing `if ... params is None` check since we handle it above. The existing SMPL-X path continues unchanged for `body_model_type == "smplx"`.

Add the SOMA helper function before `_get_joint_axis_angle`:

```python
def _get_soma_joint_axis_angle(soma_params: dict, frame_idx: int, joint_idx: int) -> np.ndarray:
    """Get axis-angle (3,) for a joint from SOMA unified poses tensor."""
    poses = soma_params.get("poses")
    if poses is None:
        return np.zeros(3, dtype=np.float32)

    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim == 3 and frame_idx < poses.shape[0] and joint_idx < poses.shape[1]:
        return poses[frame_idx, joint_idx].ravel()[:3]
    return np.zeros(3, dtype=np.float32)
```

### Update `get_joint_euler()` — already works

`get_joint_euler()` calls `_get_joint_axis_angle()` which now dispatches based on body_model_type. No changes needed.

### Update `compute_pose_issues()` for SOMA

Find `compute_pose_issues` (line ~321). The function reads `track.smplx_params["body_pose"]`. Add SOMA support.

After the existing `params = track.smplx_params` block, add a SOMA branch:

```python
        # SOMA path: use unified poses tensor (all joints)
        if track.body_model_type == "soma" and track.soma_params is not None:
            sp = track.soma_params.get("poses")
            if sp is not None:
                sp = np.asarray(sp, dtype=np.float32)
                if sp.ndim == 3 and sp.shape[0] > 1:
                    # Use all joints (skip global_orient at index 0)
                    body_poses = sp[:, 1:]  # (N, 76, 3)
                    issues.extend(
                        _detect_angular_jumps(body_poses, pid, jump_threshold_deg)
                    )
                    issues.extend(
                        _detect_jitter(body_poses, pid, jitter_window, jitter_threshold_deg)
                    )
```

This should go inside the `for pid, track in ...` loop, right after the existing SMPL-X `params` block (using `elif` or a `continue` after the SOMA block).

### Update `mirror_lr_pose_fallback()` for SOMA

The existing function reads `params["body_pose"]`. For SOMA, it should read from `soma_params["poses"]`. But since this is a fallback function called with raw params, and the caller knows which format it has, the cleanest approach is to add a separate SOMA mirror function:

```python
def mirror_lr_pose_soma_fallback(soma_params: dict, frame: int) -> dict[int, np.ndarray]:
    """Fallback L/R mirror for SOMA unified poses tensor."""
    from models.skeleton import SOMA_SKELETON

    poses = np.asarray(soma_params.get("poses", np.zeros((0, 77, 3))), dtype=np.float32)
    if poses.ndim != 3 or frame >= poses.shape[0]:
        return {}

    frame_poses = poses[frame].copy()
    mirrored: dict[int, np.ndarray] = {}
    for l_idx, r_idx in SOMA_SKELETON.lr_swap_pairs:
        if l_idx < frame_poses.shape[0] and r_idx < frame_poses.shape[0]:
            mirrored[l_idx] = frame_poses[r_idx].copy()
            mirrored[r_idx] = frame_poses[l_idx].copy()
    return mirrored
```

### Update `smooth_joint_rotations()` — already works

The function takes `(N, J, 3)` body_pose — it doesn't hardcode joint count. It works for both `(N, 21, 3)` and `(N, 76, 3)`. No changes needed.

### Update `find_similar_frames()` — already works

Same — takes `(N, J, 3)` and a joint index. No changes needed.

---

## Step 3: Add SOMA tests to `tests/test_mesh_viewport.py`

Add a new test class alongside the existing `TestForwardKinematics`:

```python
class TestForwardKinematicsSoma:
    """forward_kinematics with SOMA unified poses tensor."""

    def _make_soma_params(self, n_frames=10, n_joints=77):
        return {
            "poses": np.zeros((n_frames, n_joints, 3)),
            "transl": np.zeros((n_frames, 3)),
        }

    def test_output_shape(self):
        params = self._make_soma_params()
        joints = forward_kinematics(params, 0)
        assert joints.shape == (77, 3)

    def test_root_at_origin(self):
        params = self._make_soma_params()
        joints = forward_kinematics(params, 0)
        np.testing.assert_allclose(joints[0], [0, 0, 0], atol=1e-6)

    def test_root_with_translation(self):
        params = self._make_soma_params()
        params["transl"][3] = [1.0, 2.0, 3.0]
        joints = forward_kinematics(params, 3)
        np.testing.assert_allclose(joints[0], [1.0, 2.0, 3.0], atol=1e-6)

    def test_nonzero_rotation_moves_joints(self):
        params = self._make_soma_params()
        params["poses"][0, 1] = [0.5, 0, 0]  # rotate L_Hip
        joints = forward_kinematics(params, 0)
        zero_joints = forward_kinematics(self._make_soma_params(), 0)
        # L_Knee (child of L_Hip) should have moved
        assert not np.allclose(joints[4], zero_joints[4], atol=1e-4)

    def test_smplx_params_still_work(self):
        """Existing SMPL-X params format is unaffected."""
        params = {
            "global_orient": np.zeros((10, 3)),
            "body_pose": np.zeros((10, 21, 3)),
            "transl": np.zeros((10, 3)),
        }
        joints = forward_kinematics(params, 0)
        assert joints.shape == (52, 3)  # SMPL-X path
```

---

## Step 4: Add SOMA tests to `tests/test_pose_corrector.py`

Add alongside the existing test helpers at the top of the file:

```python
def _make_soma_session(n_frames=100, n_persons=1):
    """Create a session with SOMA params for testing."""
    session = Session(num_frames=n_frames, fps=30.0, img_width=640, img_height=480)
    for pid in range(n_persons):
        soma_params = {
            "poses": np.zeros((n_frames, 77, 3), dtype=np.float32),
            "transl": np.zeros((n_frames, 3), dtype=np.float32),
            "global_orient": np.zeros((n_frames, 3), dtype=np.float32),
        }
        # Set some non-zero values
        soma_params["poses"][0, 0] = [0.1, 0.2, 0.3]  # global orient
        soma_params["poses"][0, 1] = [0.4, 0.5, 0.6]  # L_Hip

        track = PersonTrack(
            person_id=pid,
            soma_params=soma_params,
            body_model_type="soma",
        )
        session.person_tracks[pid] = track
    return session
```

Then add test classes:

```python
class TestGetJointAxisAngleSoma:
    """_get_joint_axis_angle with SOMA unified poses tensor."""

    def test_global_orient(self):
        session = _make_soma_session()
        aa = _get_joint_axis_angle(session, 0, 0, 0)
        np.testing.assert_allclose(aa, [0.1, 0.2, 0.3], atol=1e-6)

    def test_body_joint(self):
        session = _make_soma_session()
        aa = _get_joint_axis_angle(session, 0, 0, 1)
        np.testing.assert_allclose(aa, [0.4, 0.5, 0.6], atol=1e-6)

    def test_hand_joint(self):
        session = _make_soma_session()
        session.person_tracks[0].soma_params["poses"][0, 22] = [0.7, 0.8, 0.9]
        aa = _get_joint_axis_angle(session, 0, 0, 22)
        np.testing.assert_allclose(aa, [0.7, 0.8, 0.9], atol=1e-6)

    def test_face_joint(self):
        session = _make_soma_session()
        session.person_tracks[0].soma_params["poses"][0, 52] = [0.1, 0.0, 0.0]
        aa = _get_joint_axis_angle(session, 0, 0, 52)
        np.testing.assert_allclose(aa, [0.1, 0.0, 0.0], atol=1e-6)

    def test_zero_for_missing_person(self):
        session = _make_soma_session()
        aa = _get_joint_axis_angle(session, 99, 0, 0)
        np.testing.assert_allclose(aa, [0, 0, 0])

    def test_smplx_session_still_works(self):
        """Existing SMPL-X params path is not broken."""
        session = _make_session_with_params(n_frames=10)
        aa = _get_joint_axis_angle(session, 0, 0, 0)
        np.testing.assert_allclose(aa, [0.1, 0.2, 0.3], atol=1e-6)


class TestComputePoseIssuesSoma:
    """Pose issue detection with SOMA params."""

    def test_detects_angular_jump_soma(self):
        session = _make_soma_session(n_frames=50)
        # Create a large jump at frame 10
        session.person_tracks[0].soma_params["poses"][10, 5] = [3.0, 0, 0]
        issues = compute_pose_issues(session, jump_threshold_deg=30.0)
        jump_issues = [i for i in issues if i.issue_type == "angular_jump"]
        assert len(jump_issues) > 0

    def test_no_issues_on_smooth_motion(self):
        session = _make_soma_session(n_frames=50)
        issues = compute_pose_issues(session)
        jump_issues = [i for i in issues if i.issue_type == "angular_jump"]
        assert len(jump_issues) == 0


class TestMirrorSomaFallback:
    """L/R mirror with SOMA poses."""

    def test_swaps_hip_joints(self):
        soma_params = {
            "poses": np.zeros((10, 77, 3), dtype=np.float32),
        }
        soma_params["poses"][0, 1] = [1.0, 0, 0]  # L_Hip
        soma_params["poses"][0, 2] = [0, 1.0, 0]  # R_Hip
        mirrored = mirror_lr_pose_soma_fallback(soma_params, 0)
        np.testing.assert_allclose(mirrored[1], [0, 1.0, 0])  # was R_Hip
        np.testing.assert_allclose(mirrored[2], [1.0, 0, 0])  # was L_Hip
```

---

## Step 5: Run tests

```bash
python -m pytest tests/ -x -q
```

Expected: All previous tests pass + ~15 new SOMA tests.

---

## Step 6: Commit

```bash
git add views/mesh_viewport.py views/pose_corrector_panel.py tests/test_mesh_viewport.py tests/test_pose_corrector.py
git commit -m "feat: add SOMA rendering and pose correction support (Phases 3 + 4)

Viewport: forward_kinematics() auto-detects SOMA params ('poses' key)
and dispatches to _forward_kinematics_soma() using SOMA_SKELETON hierarchy.
compute_joint_colors() already handles 77+ joints (face → Head color).

Pose corrector: _get_joint_axis_angle() detects body_model_type='soma'
and reads flat index from soma_params['poses'][frame, joint_idx].
compute_pose_issues() scans SOMA poses for angular jumps and jitter.
New mirror_lr_pose_soma_fallback() handles SOMA L/R swap.

All existing SMPL-X tests pass unchanged. ~15 new SOMA-specific tests."
```
