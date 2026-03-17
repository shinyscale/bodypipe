# Spec: Video Player

## Overview

QWidget for video frame display with seeking, playback, and LRU cache. This is the primary display widget — used by the identity inspector, pose corrector, and pipeline output preview.

## Components

### VideoPlayer (`views/video_player.py`)

```
┌──────────────────────────────────────────┐
│                                          │
│           Frame Display                  │
│         (QLabel + QPixmap)               │
│                                          │
│         Click events → frame_clicked     │
│                                          │
├──────────────────────────────────────────┤
│  |◄  ◄◄  ◄  ▶/❚❚  ►  ►►  ►|           │
│  Frame slider ═══════════●═══════════    │
│  Frame: 142 / 3600          FPS: 30.0   │
└──────────────────────────────────────────┘
```

### Frame Display

- `QLabel` with `QPixmap` — simple, fast, GPU-composited via Qt
- Aspect-ratio preserved scaling (`Qt.KeepAspectRatio`)
- Frame rendered as RGB numpy array → QImage → QPixmap pipeline
- Mouse click events forwarded as `frame_clicked(x, y)` signal (normalized 0-1 coordinates)
- Right-click context menu: Copy Frame, Save Frame As...

### Transport Controls

Buttons (QToolButton with icons):
- `|◄` First frame
- `◄◄` Back 10 frames
- `◄` Back 1 frame
- `▶/❚❚` Play/Pause toggle
- `►` Forward 1 frame
- `►►` Forward 10 frames
- `►|` Last frame

### Frame Slider

- `QSlider(Qt.Horizontal)` — range 0 to num_frames-1
- Real-time scrubbing: emit `frame_changed` on every value change
- Keyboard: Left/Right = ±1 frame, Ctrl+Left/Right = ±10

### Playback

- `QTimer` at video FPS (default 30) drives frame advance
- Play/Pause toggle
- No audio support needed (mocap video rarely has meaningful audio)

### Frame Cache

Reuse the LRU cache pattern from Gradio app:

```python
class FrameCache:
    """LRU cache with read-ahead for sequential access."""

    READAHEAD = 15
    MAX_SIZE = 200

    def get_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Get frame, populating cache with read-ahead on miss."""
        cached = self._cache.get(frame_idx)
        if cached is not None:
            return cached

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        for i in range(self.READAHEAD + 1):
            ret, frame = cap.read()
            if not ret:
                break
            self._cache.put(frame_idx + i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return self._cache.get(frame_idx)
```

## Signals

```python
class VideoPlayer(QWidget):
    frame_changed = Signal(int)          # current frame index
    frame_clicked = Signal(float, float) # normalized (x, y) click position
    playback_toggled = Signal(bool)      # True = playing
```

## Overlay Support

The VideoPlayer renders a base frame, but overlays (bboxes, skeletons) are composited by the parent widget before setting the pixmap. The player exposes:

```python
def set_frame(self, frame: np.ndarray):
    """Set the displayed frame (already composited with overlays)."""

def set_video(self, path: Path, num_frames: int, fps: float):
    """Load a new video source."""

def current_frame_index(self) -> int:
    """Get current frame position."""

def seek(self, frame_idx: int):
    """Seek to specific frame (clamps to valid range)."""
```

The parent (identity panel, pose corrector) is responsible for:
1. Listening to `frame_changed`
2. Fetching the raw frame from cache
3. Compositing overlays (bboxes, skeleton, etc.)
4. Calling `set_frame()` with the composited result

This keeps the VideoPlayer generic and reusable.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left | Previous frame |
| Right | Next frame |
| Ctrl+Left | Back 10 |
| Ctrl+Right | Forward 10 |
| Home | First frame |
| End | Last frame |

## Source Reference

- `identity_panel.py:_extract_frame` — frame extraction with read-ahead
- `identity_panel.py:_LRUFrameCache` — cache implementation
- `identity_panel.py:update_frame_display` — frame display callback
- `identity_panel.py` — transport buttons (first, back10, back1, play, fwd1, fwd10, last)

## Acceptance Criteria

- Opens video file, displays frames, slider scrubs smoothly
- Play/Pause at correct FPS
- Frame cache prevents re-decoding recently viewed frames
- Click coordinates correctly normalized to image space
- Keyboard shortcuts work when widget has focus
