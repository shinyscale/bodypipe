# sandpipe

WebGPU particle-sim prototype that reads bodypipe SAM2 masks and simulates
sand (and other materials) sloshing inside the body silhouette. Scratchpad
for physics ideas that may eventually inform a "groundedness" cue in the
main Python pipeline.

## Files

- `sandpipe-body-webgpu-v2.html` — WebGPU version. Compute shaders for
  Margolus sand physics, atomic-CAS spiral relocate at frame boundaries,
  trail texture with feedback-zoom / chromatic-aberration VFX, per-side
  sliders for gravity / splash / physics iters. Materials: sand, water,
  smoke, fire, lava, powder.
- `sandpipe-body.html` — CPU reference. Known-good source of the sloshing
  algorithm; used as the porting target when the WebGPU version drifted.

## Running

The HTML expects video masks at a path relative to the HTML file:
`../research/videos/<video-name>/masks/*.png`. Serve whatever parent
directory makes that path resolve. Example, if this repo lives at
`F:\gvhmr\bodypipe\` alongside `F:\research\videos\`:

    python -m http.server 8081 --bind 0.0.0.0 --directory F:\

then open `http://localhost:8081/gvhmr/bodypipe/sandpipe/sandpipe-body-webgpu-v2.html?video=<name>`.

Adjust directory paths to match your layout. No masks needed in-repo — the
prototype is driven by whatever SAM2-processed videos you already have.

## Status

Prototype, not integrated with bodypipe's Python pipeline. The sloshing /
settling physics (per-side gravity, inertia drift, centrifugal fling,
outside-particle bounce off the body surface) are the target for future
integration as a stability signal correlated with skeleton grounding.
