# TRANSMISSIONS Cut-Up Toolkit

A generative video editor for the TRANSMISSIONS installation. The tool analyzes an audio track and assembles synchronized cut-up videos for three tetrahedron faces using per-face logic (signal, infrastructure, noise/error).

## Quick Start
1. Place audio and video assets in the expected structure:
   - `audio/track01.wav`
   - `videos/track01_faceA/*.mp4`
   - `videos/track01_faceB/*.mp4`
   - `videos/track01_faceC/*.mp4`
2. Run the CLI to analyze and render:
   ```bash
   python transmissions_cutup.py --track track01
   ```
3. Find outputs in `output/`:
   - `output/track01_faceA.mp4`
   - `output/track01_faceB.mp4`
   - `output/track01_faceC.mp4`

## Example Workflows
- **First render for a new track**
  ```bash
  python transmissions_cutup.py --track track02
  ```
  This analyzes audio/features and clip metadata, builds timelines for faces A/B/C, renders them with shared audio, and caches metadata in `metadata/` for reuse.

- **Re-render using existing metadata**
  ```bash
  python transmissions_cutup.py --track track01 --skip-audio-analysis --skip-clip-analysis
  ```
  Useful when you only want new random cuts or changed config without reprocessing.

- **Apply triangle masks to outputs**
  ```bash
  python transmissions_cutup.py --track track01 --apply-mask
  ```
  Uses `masks/faceA_mask.png`, `masks/faceB_mask.png`, and `masks/faceC_mask.png` if present.

- **Customize per-face behavior**
  Edit the `FACE_CONFIG` dict in `transmissions_cutup.py` to adjust beat divisors, segment durations, stutter/black probabilities, or threshold tuning for RMS and spectral centroid.

## Making It Better (Ideas & Expansions)
- Add CLI flags for output resolution/fps overrides per face.
- Support LUT-based color grading or OpenCV filters per face.
- Cache per-frame analysis for clips to enable smarter motion/brightness matching.
- Integrate more audio features (e.g., spectral flux, zero-crossing rate) for richer mappings.
- Add deterministic seeding for repeatable renders.
- Export timeline JSON for each face for inspection or alternative renderers.
- Parallelize clip analysis and rendering to speed up long batches.
- Provide small sample assets plus a Makefile to demonstrate end-to-end usage.
