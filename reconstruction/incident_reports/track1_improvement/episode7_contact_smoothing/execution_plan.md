# Contact-constrained Episode 7 smoothing

Continue from the completed 2x2 translation ablation. Pure smoothing reduced motion noise but increased hand penetration. This experiment fits a trajectory from physically screened per-frame options, using the submitted v6 Episode 7 as the control.

Each of the 777 video frames has two options: its unchanged v6 translation, and its seven-frame quadratic Savitzky–Golay translation from the preserved ablation. There is no constant camera-ray depth correction. Smoothing can still change individual frame depths; the actual mean depth change is reported explicitly. Human arrays, object rotations, mesh and object scale remain unchanged.

A smoothed option is feasible only if both full anatomical-hand and FPS512 mean penetration do not increase at that frame (numerical tolerance 1e-10 cm). On frames where the control's nearest sampled hand-to-mesh-vertex gap is at most 1.5 cm, the new gap may increase by at most 0.5 cm. The translation displacement must be at most 3 cm, camera depth must exceed 0.1 m, and the first/last two frames remain unchanged. These are local predicted-hand diagnostics, not host hand-role labels or hidden-reference metrics.

A second-order dynamic program chooses the complete feasible sequence that minimizes squared sampled-surface acceleration plus a small fidelity penalty (weight 0.001). With fixed rotations, the path-dependent surface term is exactly the transformed sampled-mesh-centroid acceleration; the residual rotational variance is constant. The unchanged sequence remains feasible, providing an objective upper bound. The familiar mean-norm acceleration proxies are reported separately and need not rank trajectories identically to the squared objective.

The fit computes exact-mesh hand penetration on all frames and caches small per-frame summaries after every eight frames. Cache inputs include the control/proposal/mesh hashes, MHR model and hand-spec hashes, the checker and converter source hashes, calibration, and scored-frame metadata. Resume rejects changed inputs. The final candidate must also pass the existing scored-frame image/depth and hand/proximity validators and visual review. These checks use the same video that supplied constraints and do not constitute held-out evaluation.

Run from the workspace root:

```bash
docker run --rm --gpus all --user 1000:1000 --entrypoint python \
  -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/episode7-contact-cache \
  -v /home/alien51/video_to_data/reconstruction:/repo \
  v2d_cari4d:blackwell /repo/scripts/track1_episode7_contact_smoothing.py --validate --render
```

Focused solver tests: `reconstruction/.venv/bin/python reconstruction/scripts/test_track1_contact_smoothing.py`. The exhaustive oracle, contact/displacement rejection, fixed-rotation surface identity, feasible-control bound, and infeasible-frame cases pass.

Outputs are isolated under `data/v2d_challenge/track_1/episode7_contact_smoothing/episode_000007/`. Progress and results are in `status.json` beside this plan. No package is assembled, no artifact is promoted, and no leaderboard upload is performed.
