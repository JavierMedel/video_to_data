# Episode 7 controlled translation experiment

Start from the submitted v6 Episode 7 NPZ/GLB pair. Create a fixed 2x2 ablation: unchanged control, seven-frame quadratic Savitzky–Golay smoothing alone, +0.07 m camera-ray shift alone, and both. No fitting or candidate-specific contact corrections are allowed: they would confound the experiment. The combined branch must reproduce submitted v7 within 1 micrometre, allowing for its intermediate float32 rounding.

Every NPZ field other than `object_translation` must remain exactly equal to v6. Mesh hashes must match. Inputs and outputs are fingerprinted; a rerun verifies existing candidate arrays before reusing them. Submitted v6/v7 artifacts are read-only inputs. No packing, promotion or upload is performed.

Run from the workspace root:

```bash
docker run --rm --gpus all --user 1000:1000 --entrypoint python \
  -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/episode7-ablation-cache \
  -v /home/alien51/video_to_data/reconstruction:/repo \
  v2d_cari4d:blackwell /repo/scripts/track1_episode7_ablation.py --validate --mask-sheets
```

Without `--validate`, the script generates candidates and motion diagnostics only. `--mask-sheets` renders six matched source-video overlays per changed branch. The run is scoped to Episode 7's verified identity camera. Five focused tests are in `scripts/test_track1_episode7_ablation.py` and run in the same image.

Validation uses all 597 scored frames for visible silhouette IoU, common-pixel cached depth, full anatomical-hand penetration, FPS512 penetration and grasp-proximity checks. Motion uses all 777 video frames. The hand reviewer reuses the existing v6 control summary only after matching NPZ/mesh hashes, exact frame list, full-frame scope and hand/sample counts. Candidate penetration and both proximity trajectories are recalculated. Cached control summaries do not contain vertex-level depths, so pointwise penetration differences are explicitly unavailable.

The unchanged branch is a sanity control, so its meaningful-improvement flag should be false. A candidate can pass alignment while failing the separate meaningful-gain or depth checks. Report these flags separately. Depth support is shared between v6 and each candidate, but may differ between candidate comparisons.

The inherited 11-frame split is for historical comparison; these frames have already been used in v7 evaluation and are not new validation evidence. Cached depth is a monocular estimate. Local checks cannot determine whether smoothing or depth correction caused the hidden-reference CD-O regression. Official controlled scores would be needed to resolve that question.

Outputs: `data/v2d_challenge/track_1/episode7_translation_ablation/`, with aggregate diagnostics in `comparison.json` and `results.md` beside this plan. Preserve all branches even when a gate fails. Select no candidate automatically.
