# Physics-Based Mocap Inertia Filter — Implementation Spec

**Target machine**: Powerhouse (RTX PRO 6000 Blackwell 96GB, WSL2, user `shinyscale`)
**Primary approach**: PHC (Perpetual Humanoid Controller) with pretrained weights
**Input**: GVHMR SMPL kinematic sequences
**Output**: Physically-refined SMPL parameters (FBX/BVH export)
**Date**: 2026-04-05

---

## Motivation

Raw markerless mocap (GVHMR, bodypipe) outputs purely kinematic frame-by-frame joint rotations. Without physical context (mass, gravity, momentum), animations feel weightless, exhibit micro-jitters, lack follow-through, and suffer foot skating/ground penetration.

Inspired by Zach Lieberman's spring-system body art (a PD-controller particle system tethered to a pose-tracked dancer), this filter applies the same principle at skeleton scale: a physics-simulated ragdoll follows the kinematic mocap via springs, and the physics engine bakes in weight, inertia, and ground contact for free.

## Why PHC Over Hand-Tuned PD

Research (2026-04-05) showed that naive PD tracking fails in practice — characters fall over on fast motion, gains must be tuned per-joint AND per-motion-type, and there's no recovery from tracking errors. PHC (Zhengyi Luo, CMU/Meta, ICCV 2023) is a universal RL policy trained on 10,000+ AMASS motions that tracks arbitrary SMPL reference sequences through Isaac Gym simulation with 99.9% success rate. Pretrained weights available. An existing fork (Motion-Refinement-using-PHC) demonstrates the exact use case.

---

## Phase 0: Environment Setup on Powerhouse

### Prerequisites
- WSL2 with Ubuntu (already configured, user `shinyscale`)
- NVIDIA driver with CUDA support (RTX PRO 6000 Blackwell — confirm CUDA version)
- Conda or venv for isolation

### Steps

```bash
# SSH into Powerhouse
ssh shinyscale@100.125.206.12

# Create workspace
mkdir -p ~/mocap-filter && cd ~/mocap-filter

# Create conda environment
conda create -n phc python=3.8 -y
conda activate phc

# Install PyTorch (match CUDA version on Powerhouse)
# Blackwell needs nightly cu128 or cu130 — verify:
nvidia-smi  # check CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Isaac Gym
# Download from: https://developer.nvidia.com/isaac-gym
# Extract and:
cd isaacgym/python && pip install -e .

# Verify Isaac Gym
python -c "import isaacgym; print('Isaac Gym OK')"

# Install MuJoCo (for SMPLSim dependency)
pip install mujoco

# Clone PHC and dependencies
git clone https://github.com/ZhengyiLuo/PHC.git
cd PHC
pip install -r requirements.txt

# Install SMPLSim
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master

# Download SMPL body model files
# Requires registration at https://smpl.is.tue.mpg.de/
# Place in PHC/data/smpl/ (SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl)

# Download pretrained PHC weights
# Follow PHC README for checkpoint download links
# Place in PHC/output/phc_*/ directories
```

### Isaac Gym on Blackwell — Risk Flag
Isaac Gym Preview 4 was last updated ~2023 and targets Ampere/Hopper. Blackwell (sm_120) may need:
- CUDA 12.8+ (confirm with `nvidia-smi`)
- Possibly Isaac Lab instead of Isaac Gym (newer, actively maintained)
- If Isaac Gym fails on Blackwell, fallback to Isaac Lab eval scripts (PHC added support Aug 2025)

### Verification Checkpoint
- [ ] `python -c "import isaacgym"` succeeds
- [ ] `python -c "from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot"` succeeds
- [ ] PHC demo script runs with pretrained weights
- [ ] GPU utilization visible during PHC inference

---

## Phase 1: Run PHC on a Test Sequence

### Goal
Prove the pipeline works end-to-end with a canned AMASS motion before introducing GVHMR data.

### Steps

```bash
# PHC ships with test motions from AMASS
cd ~/mocap-filter/PHC

# Run the pretrained PHC on a test sequence
# (exact command depends on PHC version — check README)
python phc/run.py --cfg phc_prim --epoch -1 --test

# Or using the Motion-Refinement fork approach:
git clone https://github.com/Soumyabrata2003/Motion-Refinement-using-PHC.git
cd Motion-Refinement-using-PHC
# Follow their README for BVH-to-AMASS-to-PHC pipeline
```

### What to Look For
- Simulated humanoid tracks reference motion without falling
- Feet don't skate or penetrate ground
- Fast arm movements show natural follow-through/settling
- Output SMPL parameters can be exported

### Verification Checkpoint
- [ ] PHC tracks a walking sequence without falling
- [ ] PHC tracks a fast/dynamic sequence (dance, reach, turn)
- [ ] Output can be visualized (PHC includes rendering tools)
- [ ] Output can be exported as SMPL NPZ or BVH

---

## Phase 2: GVHMR-to-PHC Conversion Pipeline

### Goal
Convert bodypipe/GVHMR output into PHC's expected input format, run physics refinement, convert back.

### Input Format (GVHMR)
GVHMR outputs per-frame:
- `body_pose`: SMPL pose parameters (axis-angle, 23 joints x 3 = 69D)
- `global_orient`: Root orientation (axis-angle, 3D)
- `betas`: Body shape (10D, constant per sequence)
- `transl_world`: World-space root translation (3D)

### Target Format (PHC / AMASS)
PHC expects AMASS-format NPZ:
- `poses`: SMPL pose (156D for SMPL-H, or 72D for SMPL — body_pose + global_orient)
- `betas`: Shape (10D or 16D)
- `trans`: Root translation (Nx3)
- `gender`: String
- `mocap_framerate`: Integer

### Conversion Script: `gvhmr_to_amass.py`

```python
"""Convert GVHMR output to AMASS NPZ format for PHC consumption."""
import numpy as np
import argparse

def convert(gvhmr_path, output_path, fps=30):
    data = np.load(gvhmr_path)
    
    # GVHMR: global_orient (Nx3) + body_pose (Nx69)
    global_orient = data['global_orient']  # (N, 3)
    body_pose = data['body_pose']          # (N, 69)
    
    # AMASS expects: (N, 72) for SMPL = global_orient + body_pose
    poses = np.concatenate([global_orient, body_pose], axis=1)  # (N, 72)
    
    # Pad to 156D if PHC expects SMPL-H (72 body + 90 hands)
    # If SMPL-H: pad hand joints with zeros
    if poses.shape[1] == 72:
        hand_padding = np.zeros((poses.shape[0], 90))
        poses_h = np.concatenate([poses, hand_padding], axis=1)
    else:
        poses_h = poses
    
    np.savez(output_path,
        poses=poses_h,
        betas=data['betas'][0] if data['betas'].ndim > 1 else data['betas'],
        trans=data['transl_world'],
        gender='neutral',
        mocap_framerate=fps
    )
    print(f"Saved {output_path}: {poses_h.shape[0]} frames at {fps}fps")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='GVHMR output NPZ')
    parser.add_argument('output', help='AMASS-format NPZ for PHC')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    convert(args.input, args.output, args.fps)
```

### Conversion Script: `phc_to_smpl.py`

```python
"""Convert PHC simulation output back to SMPL parameters."""
import numpy as np

def convert_back(phc_output_path, output_path):
    data = np.load(phc_output_path)
    
    # PHC outputs simulated qpos per frame
    # Extract SMPL-compatible pose parameters
    # (Exact extraction depends on PHC output format — TBD after Phase 1)
    
    # Reconstruct:
    # - global_orient from root quaternion -> axis-angle
    # - body_pose from joint angles -> axis-angle per joint
    # - transl_world from root position
    
    # scipy for rotation conversion
    from scipy.spatial.transform import Rotation
    
    # TBD: map PHC qpos -> SMPL params
    # PHC may output SMPL params directly if using SMPLSim humanoid
    pass
```

### Verification Checkpoint
- [ ] GVHMR NPZ converts to AMASS format without errors
- [ ] PHC accepts the converted file and runs simulation
- [ ] Output converts back to SMPL params
- [ ] Side-by-side comparison: raw GVHMR vs physics-refined

---

## Phase 3: Quality Evaluation

### Metrics to Compute

Compare raw GVHMR output vs PHC-refined output on the same sequence:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **LDLJ** (Log Dimensionless Jerk) | Smoothness. Lower = smoother. | Natural motion: -6 to -3 |
| **SPARC** (Spectral Arc Length) | Smoothness via frequency. More negative = less smooth. | -1.5 to -3.0 |
| **Jerk Ratio** (refined/raw) | How much jitter was removed | 0.3-0.7 (below 0.3 = oversmoothing) |
| **Foot Skating** | Mean toe displacement during contact | < 2mm/frame |
| **Ground Penetration** | Min foot height during sequence | > -1mm |
| **Spectral Preservation** | Low-freq energy kept, high-freq attenuated | <5Hz preserved >0.9, >15Hz attenuated <0.5 |
| **Follow-Through Score** | Velocity zero-crossing delay (raw vs refined) | Positive = follow-through present |
| **Weight Metric** | Downward accel / upward accel | > 1.0 = weighted feel |

### Evaluation Script: `evaluate_physics_pass.py`

```python
"""Compare raw vs physics-refined mocap on quality metrics."""
import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation

def compute_jerk(positions, fps=30):
    """Third derivative of position."""
    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return jerk

def ldlj(positions, fps=30, duration=None):
    """Log Dimensionless Jerk — lower is smoother."""
    if duration is None:
        duration = len(positions) / fps
    jerk = compute_jerk(positions, fps)
    jerk_magnitude = np.sum(jerk**2) * (1.0 / fps)
    # Normalize by duration and displacement
    displacement = np.max(np.linalg.norm(positions - positions[0], axis=-1))
    if displacement < 1e-6:
        return 0.0
    ldlj_val = -np.log(jerk_magnitude * duration**3 / displacement**2)
    return ldlj_val

def foot_skating(toe_positions, contact_mask, fps=30):
    """Mean horizontal displacement of toes during contact frames."""
    dt = 1.0 / fps
    displacements = np.linalg.norm(np.diff(toe_positions[:, :2], axis=0), axis=1)
    contact_displacements = displacements[contact_mask[1:]]
    return np.mean(contact_displacements) if len(contact_displacements) > 0 else 0.0

def detect_foot_contacts(ankle_positions, fps=30, height_thresh=0.05, vel_thresh=0.2):
    """Simple velocity+height heuristic for foot contact detection."""
    dt = 1.0 / fps
    heights = ankle_positions[:, 2]  # Z-up assumed
    velocities = np.linalg.norm(np.diff(ankle_positions[:, :2], axis=0) / dt, axis=1)
    velocities = np.append(velocities, velocities[-1])  # pad
    contacts = (heights < height_thresh) & (velocities < vel_thresh)
    return contacts

def spectral_preservation(raw_positions, refined_positions, fps=30):
    """Compare frequency content preservation."""
    results = {}
    for label, freq_range in [('low', (0, 5)), ('high', (15, fps/2))]:
        f_raw, psd_raw = welch(raw_positions, fs=fps, axis=0)
        f_ref, psd_ref = welch(refined_positions, fs=fps, axis=0)
        mask = (f_raw >= freq_range[0]) & (f_raw <= freq_range[1])
        ratio = np.sum(psd_ref[mask]) / (np.sum(psd_raw[mask]) + 1e-10)
        results[label] = ratio
    return results

def weight_metric(positions, fps=30):
    """Ratio of downward to upward acceleration magnitude (gravity-aligned)."""
    dt = 1.0 / fps
    vel = np.diff(positions[:, 2], axis=0) / dt  # Z-axis
    acc = np.diff(vel) / dt
    down_acc = np.mean(np.abs(acc[acc < 0])) if np.any(acc < 0) else 0
    up_acc = np.mean(np.abs(acc[acc > 0])) if np.any(acc > 0) else 0
    return down_acc / (up_acc + 1e-10)

# Usage:
# raw = load_joint_positions("raw_gvhmr.npz")
# refined = load_joint_positions("phc_refined.npz")
# print("Raw LDLJ:", ldlj(raw['wrist_r']))
# print("Refined LDLJ:", ldlj(refined['wrist_r']))
# print("Jerk ratio:", np.mean(compute_jerk(refined['wrist_r'])**2) /
#                       np.mean(compute_jerk(raw['wrist_r'])**2))
```

### Verification Checkpoint
- [ ] Metrics compute on both raw and refined sequences
- [ ] LDLJ improves (lower) for extremities
- [ ] Foot skating decreases
- [ ] Low-frequency content preserved (>0.9 ratio)
- [ ] Visual A/B comparison looks better (subjective)

---

## Phase 4: Batch Processing Pipeline

### Goal
Process full GVHMR output directories through the physics filter.

```bash
# Process all sequences in a directory
python batch_refine.py \
    --input ~/bodypipe/output/gvhmr/ \
    --output ~/bodypipe/output/physics_refined/ \
    --checkpoint ~/mocap-filter/PHC/output/phc_prim/epoch_best.pth \
    --fps 30 \
    --export-format npz  # or fbx, bvh
```

### Performance Expectations
- PHC inference is GPU-accelerated (Isaac Gym)
- Powerhouse RTX PRO 6000 96GB should handle long sequences easily
- Expect: 30-minute sequence processed in minutes, not hours
- Batch multiple sequences in parallel if VRAM allows

### Verification Checkpoint
- [ ] Batch script processes a directory of sequences
- [ ] Output files match input count
- [ ] No sequences cause PHC to fail/fall
- [ ] FBX/BVH export works for Unreal import

---

## Phase 5: Integration with Bodypipe

### Goal
Wire the physics filter into bodypipe as an optional post-processing stage.

### Architecture

```
Video → bodypipe (2D pose) → GVHMR (3D SMPL) → [Physics Filter] → Unreal
                                                       ↑
                                                  PHC on Powerhouse
                                                  (SSH/API call)
```

### Options

**Option A: Remote call from f235 to Powerhouse**
- bodypipe on f235 finishes GVHMR pass
- SCP the SMPL NPZ to Powerhouse
- SSH command runs PHC refinement
- SCP results back
- Simple, no code changes to bodypipe

**Option B: REST API on Powerhouse**
- Flask/FastAPI service on Powerhouse wrapping PHC
- bodypipe POSTs SMPL sequence, gets refined sequence back
- Cleaner integration, reusable

**Option C: bodypipe flag**
- `--physics-refine` flag in bodypipe
- Handles the remote call internally
- Best UX, most coupling

Recommend: **Start with Option A** (SCP + SSH), graduate to B or C if it becomes a frequent workflow.

### Verification Checkpoint
- [ ] Full pipeline: video → bodypipe → GVHMR → PHC → FBX
- [ ] Unreal Engine imports refined FBX correctly
- [ ] A/B comparison in Unreal viewport (raw vs refined)

---

## Appendix A: PD Controller Reference (Fallback / Learning Path)

If PHC proves problematic on Blackwell (Isaac Gym compatibility) or you want a lightweight MuJoCo-only version that runs on f235, here are the hand-tuned PD parameters from research.

### Per-Joint Gain Table (75kg humanoid)

| Joint Group | Kp | Kd | Damping Ratio (zeta) | Behavior |
|---|---|---|---|---|
| Pelvis (root) | 600-1000 | 60-100 | 0.7-1.0 | Tight tracking, maintains balance |
| Spine lower/mid | 300-500 | 30-50 | 0.6-0.8 | Core stability |
| Spine upper/chest | 200-400 | 20-40 | 0.5-0.7 | Slight chest sway |
| Neck | 50-100 | 5-10 | 0.4-0.6 | Head drag |
| Head | 30-80 | 3-8 | 0.3-0.5 | Visible follow-through |
| Shoulder | 100-200 | 10-20 | 0.6-0.8 | Arm swing anchor |
| Elbow | 60-120 | 6-12 | 0.5-0.7 | Natural arm bend lag |
| Wrist | 20-50 | 2-5 | 0.3-0.5 | Maximum overlapping action |
| Fingers | 5-20 | 0.5-2 | 0.3-0.5 | Loose appendage feel |
| Hip | 300-500 | 30-50 | 0.7-0.9 | Weight-bearing, stable |
| Knee | 200-400 | 20-40 | 0.7-0.9 | Leg tracking |
| Ankle | 50-150 | 5-15 | 0.6-0.8 | Compliant ground contact |

**Rule of thumb**: Kd = 0.1 * Kp as starting point, then adjust per damping ratio target.
**Stiffness gradient**: Pelvis 1.0x, Spine 0.7x, Shoulder 0.5x, Elbow 0.3x, Wrist 0.15x.
**Animation trick**: Exaggerate hand/foot mass by 15-20% for enhanced overlapping action feel.

### Critical Damping Formula
```
zeta = Kd / (2 * sqrt(Kp * I_eff))
```
Where `I_eff` is effective inertia from MuJoCo's mass matrix diagonal.

To derive Kd from target zeta: `Kd = 2 * zeta * sqrt(Kp * I_eff)`

### Damping Ratio Guide

| zeta | Behavior | Use For |
|---|---|---|
| < 0.3 | Very underdamped, wobbly | Hair, cloth, tails |
| 0.3-0.5 | Visible overshoot + settle | Extremities (wrists, head) |
| 0.5-0.7 | Subtle overshoot | Spine, shoulders |
| 0.7-0.9 | Near-critical, quick settle | Root, weight-bearing joints |
| 1.0 | Critically damped, zero overshoot | Often too "dead" for animation |
| > 1.0 | Overdamped, sluggish | Underwater/exhaustion effects |

### MuJoCo Position Actuator Setup (if building PD from scratch)
```xml
<actuator>
  <position name="L_Hip_x" joint="L_Hip_x" kp="500" kv="50" ctrlrange="-3.14 3.14"/>
  <position name="L_Knee_y" joint="L_Knee_y" kp="300" kv="30" ctrlrange="-3.14 0"/>
  <position name="L_Ankle_x" joint="L_Ankle_x" kp="100" kv="10" ctrlrange="-1.0 1.0"/>
  <!-- Wrist: loose for follow-through -->
  <position name="L_Wrist_x" joint="L_Wrist_x" kp="30" kv="3" ctrlrange="-1.57 1.57"/>
</actuator>
```

### SMPLSim Ragdoll Generation
```python
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
import torch

robot_cfg = {"mesh": False, "model": "smplx"}
smpl_robot = SMPL_Robot(robot_cfg)
smpl_robot.load_from_skeleton(betas=torch.zeros(1, 16), gender=[0])
smpl_robot.write_xml("humanoid.xml")
# Then modify the XML to replace <motor> with <position kp=... kv=...>
```

---

## Appendix B: Biomechanics Reference (de Leva 1996)

### Segment Mass as Percentage of Body Mass (Male)

| Segment | % Body Mass | 75kg (kg) |
|---|---|---|
| Head + Neck | 6.94% | 5.21 |
| Upper Trunk | 15.96% | 11.97 |
| Mid Trunk | 16.33% | 12.25 |
| Pelvis | 11.17% | 8.38 |
| Upper Arm (each) | 2.71% | 2.03 |
| Forearm (each) | 1.62% | 1.22 |
| Hand (each) | 0.61% | 0.46 |
| Thigh (each) | 14.16% | 10.62 |
| Shank (each) | 4.33% | 3.25 |
| Foot (each) | 1.37% | 1.03 |

---

## Appendix C: Animation Principles Mapped to Physics Parameters

| Disney Principle | Physics Implementation |
|---|---|
| **Follow-Through** | Low Kp on distal joints; arms/head continue after torso stops |
| **Overlapping Action** | Proximal-to-distal stiffness gradient creates cascading motion |
| **Slow In / Slow Out** | Velocity-dependent Kp: tighten on accel (2x base), soften on decel (0.4x base) |
| **Arcs** | Natural from rotational joints + inertia — no tuning needed |
| **Secondary Action** | Very low Kp (5-20), zeta ~0.3 on appendages |
| **Timing** | Settle time = 4 / (zeta * sqrt(Kp / I_eff)) |
| **Exaggeration** | Heavier-than-real hands + softer-than-real wrist springs |

---

## Appendix D: Key Repos and Papers

### Primary (use these)
| Repo | Purpose | Link |
|---|---|---|
| **PHC** | Pretrained physics humanoid controller | github.com/ZhengyiLuo/PHC |
| **SMPLSim** | SMPL-to-MuJoCo XML generator | github.com/ZhengyiLuo/SMPLSim |
| **Motion-Refinement-using-PHC** | BVH→AMASS→PHC pipeline | github.com/Soumyabrata2003/Motion-Refinement-using-PHC |
| **UnderPressure** | Foot contact detection + IK fix | github.com/InterDigitalInc/UnderPressure |

### Reference (study these)
| Paper | What | Year |
|---|---|---|
| PHC (Luo et al.) | Universal humanoid tracking via RL | ICCV 2023 |
| DeepMimic (Peng et al.) | PD + RL motion imitation | SIGGRAPH 2018 |
| Stable PD (Tan et al.) | Stability analysis for PD controllers | IEEE CG&A 2011 |
| PhysDiff (NVIDIA) | Physics in diffusion loop | ICCV 2023 |
| The Last Mile (Tao et al.) | Production-ready physics polish | SIGGRAPH Asia 2025 |
| PhysHMR | End-to-end video→physics | SIGGRAPH Asia 2025 |
| MimicKit (Peng) | Multi-algorithm RL tracking | github.com/xbpeng/MimicKit |
| de Leva 1996 | Segment mass/inertia data | J Biomechanics |
| HuMoR (Rempe et al.) | Learned contact + motion prior | ICCV 2021 |

### Fallback (if Isaac Gym fails on Blackwell)
| Repo | Purpose | Link |
|---|---|---|
| **PHC_MJX** | PHC in MuJoCo 3 (WIP) | github.com/ZhengyiLuo/PHC_MJX |
| **UHC** | MuJoCo-native humanoid controller | github.com/ZhengyiLuo/UHC |
| **DRLoco** | DeepMimic in MuJoCo + SB3 | github.com/rgalljamov/DRLoco |
