# Spec: 3D Viewport

## Overview

`QOpenGLWidget` rendering SMPL-X mesh from per-frame params, with camera from K_fullimg, per-vertex coloring, skeleton overlay, and joint selection via ray-cast. This is the centerpiece of the pose corrector — providing what the Gradio app fundamentally cannot.

## Architecture

```
MeshViewport (QOpenGLWidget)
├── SMPL-X body model (loaded once, vertices updated per frame)
├── Skeleton overlay (GL_LINES from FK joint positions)
├── Joint spheres (GL_POINTS or instanced spheres, clickable)
├── Grid floor (reference plane)
└── Camera (from K_fullimg or free-orbit)
```

## Rendering Pipeline

### Frame Update Flow

```
on_frame_changed(frame_idx)
  → load params[frame_idx] from Session
  → apply pose corrections (if any)
  → compute SMPL-X vertices (torch → numpy)
  → update VBO
  → compute FK joints for skeleton
  → update()  # triggers paintGL
```

### SMPL-X Vertex Computation

```python
def _compute_vertices(self, params: dict, frame_idx: int) -> np.ndarray:
    """Compute SMPL-X mesh vertices for a single frame."""
    # Use the SMPL-X body model
    output = self._body_model(
        global_orient=params['global_orient'][frame_idx:frame_idx+1],
        body_pose=params['body_pose'][frame_idx:frame_idx+1],
        betas=params['betas'][:1],
        transl=params['transl'][frame_idx:frame_idx+1],
    )
    return output.vertices[0].detach().cpu().numpy()
```

Load the body model once at init. Use the same model files as the existing pipeline (`smplx` package).

### Vertex Colors

- Default: skin-tone solid color with Phong shading
- Per-joint influence: color vertices by nearest joint (useful for debugging)
- Confidence: color by per-frame confidence (green=good, red=bad)
- Correction highlight: flash modified joints in accent color

### Camera Modes

1. **In-camera** (default): Match the video camera using K_fullimg intrinsics
   - Perspective projection from camera K matrix
   - Mesh appears overlaid on video frame (composite)

2. **Free orbit**: Detached camera, orbit/pan/zoom with mouse
   - Left drag: orbit
   - Middle drag: pan
   - Scroll: zoom
   - Double-click: reset to default view

### Skeleton Overlay

- Draw bones as colored lines (GL_LINES)
- Draw joints as spheres (GL_POINTS with point size, or small icospheres)
- Selected joint: highlighted in accent color, larger size
- Joint names as text labels (QPainter overlay after GL render)

## Joint Selection

### Click-to-Select

Two approaches (implement both, toggle):

1. **Screen-space distance**: Project all joint positions to screen, find nearest to click within threshold
   ```python
   def _pick_joint(self, screen_x: int, screen_y: int) -> int | None:
       projected = self._project_joints_to_screen()
       distances = np.linalg.norm(projected - [screen_x, screen_y], axis=1)
       nearest = np.argmin(distances)
       if distances[nearest] < 20:  # pixel threshold
           return nearest
       return None
   ```

2. **Color-coded render pass**: Render each joint as a unique solid color to an FBO, read pixel at click position
   - More accurate for overlapping joints
   - Implement as optimization later

## OpenGL Setup

### Shaders

**Vertex shader** (`shaders/mesh.vert`):
```glsl
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_normal;
out vec3 frag_color;
out vec3 frag_pos;

void main() {
    frag_pos = vec3(model * vec4(position, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * normal;
    frag_color = color;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
```

**Fragment shader** (`shaders/mesh.frag`):
```glsl
#version 330 core
in vec3 frag_normal;
in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient;

out vec4 out_color;

void main() {
    vec3 norm = normalize(frag_normal);
    float diff = max(dot(norm, -light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    vec3 result = (ambient + diffuse) * frag_color;
    out_color = vec4(result, 1.0);
}
```

### VBO Layout

- Vertex positions: updated every frame (dynamic VBO)
- Normals: recomputed per frame (face normals → vertex normals)
- Colors: updated when color mode changes
- Indices: static (SMPL-X face topology doesn't change)

### Performance

- Target: 30+ FPS for single-person mesh
- SMPL-X has ~10K vertices, ~20K faces — well within GL budget
- Vertex computation is the bottleneck (SMPL-X forward pass) — cache across scrub if params unchanged
- Use `GL_DYNAMIC_DRAW` for vertex/normal VBOs, `GL_STATIC_DRAW` for indices

## Integration

```python
class MeshViewport(QOpenGLWidget):
    joint_clicked = Signal(int)       # joint index
    camera_changed = Signal(object)   # camera state (for syncing)

    def set_session(self, session: Session): ...
    def set_person(self, person_id: int): ...
    def on_frame_changed(self, frame_idx: int): ...
    def set_camera_mode(self, mode: str): ...  # "incam" or "orbit"
    def set_color_mode(self, mode: str): ...   # "solid", "joint", "confidence"
    def highlight_joint(self, joint_idx: int): ...
```

## Source Reference

- `multi_person_split.py:render_multi_person_incam()` — SMPL-X vertex computation, camera transform
- `visualize_skeleton.py:forward_kinematics()` — FK for joint positions
- `visualize_skeleton.py:project_to_2d()` — 3D→2D projection
- `pose_correction.py:compute_skeleton_frame()` — FK + projection for single frame

## Acceptance Criteria

- Renders SMPL-X mesh with Phong shading
- Skeleton overlay with clickable joints
- Camera matches video intrinsics (in-camera mode)
- Free orbit camera with mouse controls
- Smooth frame scrubbing (>20 FPS)
- Joint click emits signal with correct joint index
- Handles missing params gracefully (shows empty viewport)
