# Submission complete

All 27 reruns and final review are complete; the validated third version was uploaded and scored on all five leaderboards. See submission_results.md. Do not restart the queue or repeat the upload.

## Historical recovery notes

# Rollout resumed on 2026-10-13

Docker Desktop was restarted. The queue resumed Episode 20 from 59 saved frames and advanced to 62 frames, confirming recovery. A fresh FoundationPose estimator now initializes its saved pose tensor using its mesh point dtype/device before tracking. Completed candidates 7, 8, and 19 remain preserved; Episode 19 still needs promotion review.

# Rollout paused at user request

Queue and active GPU container are stopped. Completed candidates: Episodes 7, 8, and 19. Episodes 7 and 8 are approved; Episode 19 still needs visual promotion review. Resume Episode 20 from its saved checkpoint, then finish the remaining queue. No third submission has been uploaded.

Resume command, from repository root:

```bash
reconstruction/.venv/bin/python reconstruction/scripts/track1_rollout_all.py
```

The runner preserves completed candidates and uses input-bound frame checkpoints. Review all candidates before packing or uploading.

Paused state:
```json
{
  "state": "PAUSED",
  "episode": 20,
  "stage": "pipeline",
  "completed": [
    7,
    8,
    19
  ],
  "requested": [
    7,
    8,
    19,
    20,
    27,
    29,
    0,
    1,
    2,
    3,
    4,
    5,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    21,
    22,
    23,
    24,
    25,
    26
  ],
  "updated": "2026-10-13T04:32:32.856718+00:00",
  "reason": "User requested stop now and resume tomorrow",
  "checkpoint": {
    "path": "reconstruction/data/v2d_challenge/track_1/improvement_rollout_v1/episode_000020/run/foundationpose/episode_000020.progress.pkl",
    "completed_frames": 59,
    "total_frames": 549
  }
}
```
