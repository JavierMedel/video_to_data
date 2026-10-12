# Track 1 improvement pilot

Baseline submission and Episode 16 remain preserved. User authorized finishing the interrupted pilot and submitting one new Track 1 version on 2026-10-12. Second version uploaded and scored COMPLETE on all five competitions; see the final results below.

Visual inspection confirmed wrong-object segmentation: Episode 6 masks target the support cart rather than iron; Episode 28 masks target a background desk rather than the rolling cart. Episode 18 masks fragment/drift onto support geometry. Existing scale estimates from those masks are unsuitable and will not be used to calibrate the corrected objects.

Corrected prompts were selected from the released video images (not hidden reference labels). Isolated outputs: data/v2d_challenge/track_1/improvement_pilot_v1/. Episode 6 corrected SAM2 masks are running. After mask verification, regenerate affected SAM3D meshes, calibrate using cached human-aligned depth, and rerun only downstream stages into isolated candidate runs. Human/depth caches will be reused when compatible.

START episode_000028 corrected masks
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam2/lib:/workspace/v2d_sam2/lib:ro v2d_sam2:blackwell -m v2d.sam2.lib.video_to_masks --video_path /repo/data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000028.mp4 --prompts_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/object_prompts.json --masks_dir /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/object_masks --weights_dir /repo/data/weights/sam2`

episode_000028 masks exit 0

START episode_000018 corrected masks
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam2/lib:/workspace/v2d_sam2/lib:ro v2d_sam2:blackwell -m v2d.sam2.lib.video_to_masks --video_path /repo/data/v2d_challenge/track_1/videos/chunk-000/observation.images.exo_camera/episode_000018.mp4 --prompts_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/object_prompts.json --masks_dir /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/object_masks --weights_dir /repo/data/weights/sam2`

episode_000018 masks exit 0

START episode_000006 depth-grounded SAM3D mesh
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/repo/data/weights/sam3d/torch_home -e HF_HOME=/repo/data/weights/sam3d/hf_home -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro v2d_sam3d:blackwell -m v2d.sam3d.lib.image_to_mesh --image_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/rgb.png --mask_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/mask.png --mesh_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/object.glb --transform_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/transform.json --intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/intrinsics.json --weights_dir /repo/data/weights/sam3d --pointmap_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/pointmap.npy --pointmap_intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/intrinsics.json --seed 42`

FAILED episode_000006 SAM3D exit 1
```
    from kaolin.utils.testing import check_tensor
  File "/opt/conda/lib/python3.11/site-packages/kaolin/__init__.py", line 9, in <module>
    from . import physics
  File "/opt/conda/lib/python3.11/site-packages/kaolin/physics/__init__.py", line 3, in <module>
    from . import simplicits
  File "/opt/conda/lib/python3.11/site-packages/kaolin/physics/simplicits/__init__.py", line 5, in <module>
    from .losses_warp import *
  File "/opt/conda/lib/python3.11/site-packages/kaolin/physics/simplicits/losses_warp.py", line 27, in <module>
    wp.init()
  File "/opt/conda/lib/python3.11/site-packages/warp/_src/context.py", line 16199, in init
    runtime = Runtime()
              ^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/warp/_src/context.py", line 8535, in __init__
    warp._src.build.init_kernel_cache(warp.config.kernel_cache_dir)
  File "/opt/conda/lib/python3.11/site-packages/warp/_src/build.py", line 257, in init_kernel_cache
    os.makedirs(warp.config.kernel_cache_dir, exist_ok=True)
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 215, in makedirs
  File "<frozen os>", line 225, in makedirs
PermissionError: [Errno 13] Permission denied: '/.cache'
```

START episode_000006 depth-grounded SAM3D mesh
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/repo/data/weights/sam3d/torch_home -e HF_HOME=/repo/data/weights/sam3d/hf_home -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro v2d_sam3d:blackwell -m v2d.sam3d.lib.image_to_mesh --image_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/rgb.png --mask_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/mask.png --mesh_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/object.glb --transform_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/transform.json --intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh/intrinsics.json --weights_dir /repo/data/weights/sam3d --pointmap_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/pointmap.npy --pointmap_intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/mesh_inputs/intrinsics.json --seed 42`

COMPLETE episode_000006 mesh; applied saved scale [0.32635390758514404, 0.32635390758514404, 0.32635390758514404], extents metres [0.10620760076474056, 0.1616322928015279, 0.3258423533998922]

START episode_000028 depth-grounded SAM3D mesh
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/repo/data/weights/sam3d/torch_home -e HF_HOME=/repo/data/weights/sam3d/hf_home -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro v2d_sam3d:blackwell -m v2d.sam3d.lib.image_to_mesh --image_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh_inputs/rgb.png --mask_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh_inputs/mask.png --mesh_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh/object.glb --transform_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh/transform.json --intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh/intrinsics.json --weights_dir /repo/data/weights/sam3d --pointmap_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh_inputs/pointmap.npy --pointmap_intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/mesh_inputs/intrinsics.json --seed 42`

COMPLETE episode_000028 mesh; applied saved scale [0.8246045112609863, 0.8246045112609863, 0.8246045112609863], extents metres [0.5799031150734493, 0.8222981107835068, 0.5811734772420181]

START episode_000018 depth-grounded SAM3D mesh
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -e PYTHONUNBUFFERED=1 -e TORCH_HOME=/repo/data/weights/sam3d/torch_home -e HF_HOME=/repo/data/weights/sam3d/hf_home -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_sam3d/lib:/workspace/v2d_sam3d/lib:ro v2d_sam3d:blackwell -m v2d.sam3d.lib.image_to_mesh --image_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh_inputs/rgb.png --mask_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh_inputs/mask.png --mesh_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh/object.glb --transform_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh/transform.json --intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh/intrinsics.json --weights_dir /repo/data/weights/sam3d --pointmap_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh_inputs/pointmap.npy --pointmap_intrinsics_path /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000018/mesh_inputs/intrinsics.json --seed 42`

Corrected masks visually reviewed on nine frames spanning each video. Coverage: iron 816 frames, cart 366 frames, roller 535 frames. None of the reviewed masks selects the background desk or support cart. New iron mesh generated with cached aligned-depth pointmap; dimensions 0.106 x 0.162 x 0.326 m. Quality gates have four passing regression checks for depth divergence, zero overlap, negative depth, and a motion jump requiring reacquisition.

COMPLETE episode_000018 mesh; applied saved scale [0.3271527886390686, 0.3271527886390686, 0.3271527886390686], extents metres [0.3254674966984439, 0.09734908248632923, 0.32596909071631686]

START episode_000006 pipeline
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_reconstruct_container.py --episode 6`

COMPLETE episode_000006 pipeline

START episode_000006 convert
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_convert_refined.py --root /repo --episode 6 --run /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/run --out-dir /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000006/submission_episodes --progress-report /repo/incident_reports/track1_improvement/progress.md`

Converted episode_000006 from preserved refined output: 816 frames. Official converter report: {"device": "cuda", "fitted_frames": 816, "frames": 816, "invalid_input_frames": [], "keyframes": [0, 102, 204, 306, 408, 509, 611, 713, 815], "precision": "float32", "refits": {"backward_resolves": 40, "cold_refits": 40, "identity_keyframes": [0, 102, 204, 306, 408, 509, 611, 713, 815], "identity_refinement_max_change": 0.006605802202813815}, "seconds": 423.2, "seconds_per_frame": 0.519, "vertex_error_mm": {"max": 11.469670866806748, "mean": 0.7490629146537714, "p99": 3.659029783514116, "worst_frame": 257, "worst_frame_mean": 1.9294382583922232}}

COMPLETE episode_000006 convert

START episode_000006 compare
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_compare.py /repo 6`

FAILED episode_000006 compare: exit 1
```
mkdir -p failed for path /.config/matplotlib: [Errno 13] Permission denied: '/.config'
Matplotlib created a temporary cache directory at /tmp/matplotlib-byor_5xj because there was an issue with the default path (/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Warp 1.18.0 initialized:
   CUDA Toolkit 13.4, Driver 13.4
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA GeForce RTX 5090 Laptop GPU" (24 GiB, sm_120, mempool enabled)
   Kernel cache:
     /tmp/v2d-pilot-cache/warp/1.18.0
Traceback (most recent call last):
  File "/repo/scripts/track1_pilot_compare.py", line 13, in <module>
    from prep.run_foundationpose_mhr_export import _render_candidate_masks,_scaled_image_grid
  File "/repo/modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py", line 27, in <module>
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, cluster_poses
ImportError: cannot import name 'cluster_poses' from 'estimater' (/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py)
```

START episode_000006 pipeline
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_reconstruct_container.py --episode 6`

COMPLETE episode_000006 pipeline

START episode_000006 compare
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_compare.py /repo 6`

Pilot comparison episode_000006
```json
{
  "baseline": {
    "visible_iou_mean": 0.055877046786587344,
    "visible_iou_median": 0.0,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.8605079650878906,
      27.2790470123291
    ],
    "translation_acceleration_cm_proxy": 33.22622579618296
  },
  "gated_tracker": {
    "visible_iou_mean": 0.8095412019527319,
    "visible_iou_median": 0.8271604776382446,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.9255776405334473,
      3.4147579669952393
    ],
    "translation_acceleration_cm_proxy": 2.8741851449012756
  },
  "coconet": {
    "visible_iou_mean": 0.837647613250848,
    "visible_iou_median": 0.8469656705856323,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.8665671348571777,
      3.263476610183716
    ],
    "translation_acceleration_cm_proxy": 1.7728226259350777
  },
  "refined": {
    "visible_iou_mean": 0.837767891811602,
    "visible_iou_median": 0.8490259647369385,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.8731729984283447,
      3.2472572326660156
    ],
    "translation_acceleration_cm_proxy": 0.7627159357070923
  }
}
```

COMPLETE episode_000006 compare

START episode_000028 pipeline
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_reconstruct_container.py --episode 28`

Baseline semantic review found additional wrong-object masks in Episodes 8 (tripod instead of iron) and 9 (background desks instead of moved desk). Episodes 6 and 28 were already repaired in the pilot. Single sampled frames are insufficient to certify remaining full trajectories. Shared baseline iron mesh was generated from the wrong Episode 6 mask, so Episodes 7 and 8 also require corrected iron geometry even if their own masks are repaired. No expansion has yet been run.

Iron pilot passed local comparison against corrected masks on 33 scored frames: mean visible silhouette IoU 0.05588 baseline -> 0.83777 refined; max depth 27.279 -> 3.247 m; translation second-difference proxy 33.226 -> 0.763 cm/frame^2. This is observed-input consistency, not held-out Kaggle performance. No new upload. Comparison dependency error was module search order; fixed without rerunning completed iron GPU stages or overwriting converted arrays.

COMPLETE episode_000028 pipeline

START episode_000028 convert
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_convert_refined.py --root /repo --episode 28 --run /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/run --out-dir /repo/data/v2d_challenge/track_1/improvement_pilot_v1/episode_000028/submission_episodes --progress-report /repo/incident_reports/track1_improvement/progress.md`

Converted episode_000028 from preserved refined output: 366 frames. Official converter report: {"device": "cuda", "fitted_frames": 366, "frames": 366, "invalid_input_frames": [], "keyframes": [0, 122, 243, 365], "precision": "float32", "refits": {"backward_resolves": 17, "cold_refits": 17, "identity_keyframes": [0, 122, 243, 365], "identity_refinement_max_change": 0.0048258533911482715}, "seconds": 215.4, "seconds_per_frame": 0.589, "vertex_error_mm": {"max": 11.189560619552768, "mean": 0.953382125790084, "p99": 4.139386697638538, "worst_frame": 287, "worst_frame_mean": 2.0985919662480397}}

COMPLETE episode_000028 convert

START episode_000028 compare
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_compare.py /repo 28`

Pilot comparison episode_000028
```json
{
  "baseline": {
    "visible_iou_mean": 0.0,
    "visible_iou_median": 0.0,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.4195356369018555,
      8.388994216918945
    ],
    "translation_acceleration_cm_proxy": 14.173429235170012
  },
  "gated_tracker": {
    "visible_iou_mean": 0.6962520371783864,
    "visible_iou_median": 0.705633819103241,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      3.4366531372070312,
      4.15506649017334
    ],
    "translation_acceleration_cm_proxy": 2.5589486584067345
  },
  "coconet": {
    "visible_iou_mean": 0.6686006043896531,
    "visible_iou_median": 0.6694484949111938,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      3.382169723510742,
      3.9275903701782227
    ],
    "translation_acceleration_cm_proxy": 2.0942553877830505
  },
  "refined": {
    "visible_iou_mean": 0.6698492819612677,
    "visible_iou_median": 0.6769784092903137,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      3.4019343852996826,
      3.917264699935913
    ],
    "translation_acceleration_cm_proxy": 0.9663893841207027
  }
}
```

COMPLETE episode_000028 compare

START episode_000018 pipeline

USER STOP: Episode 18 container `7f1466ec99d5` was stopped at the user's request because runtime was too long. Docker stop returned exit code 0; verified no containers remain running. Preserve completed masks, mesh, preparation stages, logs, and completed Episode 6/28 candidates. Interrupted object tracking has no saved mid-stage checkpoint and is not counted complete. No restart or Kaggle submission is authorized by this stop request.
`docker run --rm --gpus all --user 1000:1000 --entrypoint python -e PYTHONUNBUFFERED=1 -e XDG_CACHE_HOME=/tmp/v2d-pilot-cache -v /home/alien51/video_to_data/reconstruction:/repo -v /home/alien51/video_to_data/reconstruction/modules/v2d_cari4d/lib:/workspace/v2d_cari4d/lib:ro v2d_cari4d:blackwell /repo/scripts/track1_pilot_reconstruct_container.py --episode 18`

2026-10-12T01:03:11.147220+00:00 Local candidate builder waiting for all three pilot comparisons. No Kaggle uploader is called.

Cart pilot comparison passed: mean visible silhouette IoU 0.000 baseline -> 0.66985 refined; translation acceleration proxy 14.173 -> 0.966 cm/frame^2; maximum depth 8.389 -> 3.917 m. Accepted tracker frames 362/366. Roller tracking is slower because failed quality checks trigger repeated registration; GPU telemetry confirms active computation at 100% utilization, not an idle stalled process.

FAILED episode_000018 pipeline: exit 137
```
        [0.5490, 0.5686, 0.5922]], device='cuda:0')
[reset_object()] reset done
Module Utils 65dffc6 load on device 'cuda:0' took 430.81 ms  (compiled)
/opt/conda/lib/python3.11/site-packages/torch/__init__.py:1264: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at /pytorch/torch/csrc/tensor/python_tensor.cpp:434.)
  _C._set_default_tensor_type(t)
/repo/modules/v2d_cari4d/lib/cari4d/learning/training/predict_pose_refine.py:177: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.amp):
/repo/modules/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:181: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=self.amp):
QUALITY_TRACKER_PROGRESS 0 535 {"frame": "000000", "reliable": true, "source": "register", "iou": 0.4897959232330322, "expected_depth_m": 3.2890000343322754, "rejection": ""}
QUALITY_TRACKER_PROGRESS 25 535 {"frame": "000025", "reliable": false, "source": "carry", "iou": 0.4965035021305084, "expected_depth_m": 3.2899999618530273, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 50 535 {"frame": "000050", "reliable": true, "source": "track", "iou": 0.5461538434028625, "expected_depth_m": 3.2300000190734863, "rejection": ""}
QUALITY_TRACKER_PROGRESS 75 535 {"frame": "000075", "reliable": true, "source": "track", "iou": 0.4834437072277069, "expected_depth_m": 3.0920000076293945, "rejection": ""}
QUALITY_TRACKER_PROGRESS 100 535 {"frame": "000100", "reliable": false, "source": "carry", "iou": 0.27559053897857666, "expected_depth_m": 3.2219998836517334, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 125 535 {"frame": "000125", "reliable": false, "source": "carry", "iou": 0.0, "expected_depth_m": 3.1610000133514404, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 150 535 {"frame": "000150", "reliable": true, "source": "track", "iou": 0.5643564462661743, "expected_depth_m": 3.2785000801086426, "rejection": ""}
QUALITY_TRACKER_PROGRESS 175 535 {"frame": "000175", "reliable": false, "source": "carry", "iou": 0.529411792755127, "expected_depth_m": 3.0160000324249268, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 200 535 {"frame": "000200", "reliable": false, "source": "carry", "iou": 0.47333332896232605, "expected_depth_m": 2.9700000286102295, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 225 535 {"frame": "000225", "reliable": false, "source": "carry", "iou": 0.4303797483444214, "expected_depth_m": 3.078000068664551, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 250 535 {"frame": "000250", "reliable": false, "source": "carry", "iou": 0.3333333432674408, "expected_depth_m": 3.124000072479248, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 275 535 {"frame": "000275", "reliable": false, "source": "carry", "iou": 0.4166666567325592, "expected_depth_m": 3.193000078201294, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 300 535 {"frame": "000300", "reliable": true, "source": "register", "iou": 0.36734694242477417, "expected_depth_m": 3.1640000343322754, "rejection": ""}
QUALITY_TRACKER_PROGRESS 325 535 {"frame": "000325", "reliable": false, "source": "carry", "iou": 0.39024388790130615, "expected_depth_m": 2.924999952316284, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 350 535 {"frame": "000350", "reliable": false, "source": "carry", "iou": 0.4873417615890503, "expected_depth_m": 2.8459999561309814, "rejection": "registration_failed_mask_or_depth_gate"}
QUALITY_TRACKER_PROGRESS 375 535 {"frame": "000375", "reliable": false, "source": "carry", "iou": 0.515625, "expected_depth_m": 3.1384999752044678, "rejection": "registration_failed_mask_or_depth_gate"}
```

2026-10-12T02:44:41.207048+00:00 BLOCKED: pilot comparison missing for episode 18. No candidate packed.

2026-10-12T15:31:14.315359+00:00 USER AUTHORIZATION: finish the interrupted run and submit one new Track 1 version across all five metric competitions. Episode 18 tracking restarted; completed Episode 6/28 candidates and all baseline artifacts preserved. This supersedes the previous stop request.

2026-10-12T17:29:47.587968+00:00 Episode 18 restarted tracking reached frame 300/535. Container remains active; CPU time advances and GPU utilization reaches 100%. Public producing-code commit 811b00735b793033d48047217ab1a41872d7637f verified on remote branch. Kaggle check-entry passed all five competitions. Conversion/comparison queued after successful reconstruction.

RESUMED START episode_000018 convert

Converted episode_000018 from preserved refined output: 535 frames. Official converter report: {"device": "cuda", "fitted_frames": 535, "frames": 535, "invalid_input_frames": [], "keyframes": [0, 107, 214, 320, 427, 534], "precision": "float32", "refits": {"backward_resolves": 0, "cold_refits": 0, "identity_keyframes": [0, 107, 214, 320, 427, 534], "identity_refinement_max_change": 0.01727859037437396}, "seconds": 251.8, "seconds_per_frame": 0.471, "vertex_error_mm": {"max": 9.7611331480309, "mean": 0.9451215854013296, "p99": 4.719228296546921, "worst_frame": 51, "worst_frame_mean": 1.4019028006000827}}

RESUMED COMPLETE episode_000018 convert

RESUMED START episode_000018 compare

Pilot comparison episode_000018
```json
{
  "baseline": {
    "visible_iou_mean": 0.0035473549682082553,
    "visible_iou_median": 0.0,
    "negative_center_depth_frames": 2,
    "depth_range_m": [
      -0.060736894607543945,
      3.3084523677825928
    ],
    "translation_acceleration_cm_proxy": 12.63507777055149
  },
  "gated_tracker": {
    "visible_iou_mean": 0.3367904688824307,
    "visible_iou_median": 0.3609898090362549,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.8577308654785156,
      3.5067930221557617
    ],
    "translation_acceleration_cm_proxy": 3.021949902176857
  },
  "coconet": {
    "visible_iou_mean": 0.41360377723520453,
    "visible_iou_median": 0.4216216206550598,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.87520694732666,
      3.7352375984191895
    ],
    "translation_acceleration_cm_proxy": 2.774227410554886
  },
  "refined": {
    "visible_iou_mean": 0.4121479378505187,
    "visible_iou_median": 0.4000000059604645,
    "negative_center_depth_frames": 0,
    "depth_range_m": [
      2.884840965270996,
      3.720088481903076
    ],
    "translation_acceleration_cm_proxy": 1.0897526517510414
  }
}
```

RESUMED COMPLETE episode_000018 compare

2026-10-12T20:18:59.260930+00:00 Local candidate builder waiting for all three pilot comparisons. No Kaggle uploader is called.

2026-10-12T20:19:13.794624+00:00 LOCAL CANDIDATE READY: {"status": "LOCAL_VALIDATION_PASS", "rows": 740780, "episodes": 30, "improved_episodes": [6, 18, 28], "baseline_episodes_preserved": 27, "archive": "/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_improvement_pilot_v1.parquet", "commit": "811b00735b793033d48047217ab1a41872d7637f", "uploaded": false, "comparisons": {"6": {"baseline": {"visible_iou_mean": 0.055877046786587344, "visible_iou_median": 0.0, "negative_center_depth_frames": 0, "depth_range_m": [2.8605079650878906, 27.2790470123291], "translation_acceleration_cm_proxy": 33.22622579618296}, "gated_tracker": {"visible_iou_mean": 0.8095412019527319, "visible_iou_median": 0.8271604776382446, "negative_center_depth_frames": 0, "depth_range_m": [2.9255776405334473, 3.4147579669952393], "translation_acceleration_cm_proxy": 2.8741851449012756}, "coconet": {"visible_iou_mean": 0.837647613250848, "visible_iou_median": 0.8469656705856323, "negative_center_depth_frames": 0, "depth_range_m": [2.8665671348571777, 3.263476610183716], "translation_acceleration_cm_proxy": 1.7728226259350777}, "refined": {"visible_iou_mean": 0.837767891811602, "visible_iou_median": 0.8490259647369385, "negative_center_depth_frames": 0, "depth_range_m": [2.8731729984283447, 3.2472572326660156], "translation_acceleration_cm_proxy": 0.7627159357070923}}, "18": {"baseline": {"visible_iou_mean": 0.0035473549682082553, "visible_iou_median": 0.0, "negative_center_depth_frames": 2, "depth_range_m": [-0.060736894607543945, 3.3084523677825928], "translation_acceleration_cm_proxy": 12.63507777055149}, "gated_tracker": {"visible_iou_mean": 0.3367904688824307, "visible_iou_median": 0.3609898090362549, "negative_center_depth_frames": 0, "depth_range_m": [2.8577308654785156, 3.5067930221557617], "translation_acceleration_cm_proxy": 3.021949902176857}, "coconet": {"visible_iou_mean": 0.41360377723520453, "visible_iou_median": 0.4216216206550598, "negative_center_depth_frames": 0, "depth_range_m": [2.87520694732666, 3.7352375984191895], "translation_acceleration_cm_proxy": 2.774227410554886}, "refined": {"visible_iou_mean": 0.4121479378505187, "visible_iou_median": 0.4000000059604645, "negative_center_depth_frames": 0, "depth_range_m": [2.884840965270996, 3.720088481903076], "translation_acceleration_cm_proxy": 1.0897526517510414}}, "28": {"baseline": {"visible_iou_mean": 0.0, "visible_iou_median": 0.0, "negative_center_depth_frames": 0, "depth_range_m": [2.4195356369018555, 8.388994216918945], "translation_acceleration_cm_proxy": 14.173429235170012}, "gated_tracker": {"visible_iou_mean": 0.6962520371783864, "visible_iou_median": 0.705633819103241, "negative_center_depth_frames": 0, "depth_range_m": [3.4366531372070312, 4.15506649017334], "translation_acceleration_cm_proxy": 2.5589486584067345}, "coconet": {"visible_iou_mean": 0.6686006043896531, "visible_iou_median": 0.6694484949111938, "negative_center_depth_frames": 0, "depth_range_m": [3.382169723510742, 3.9275903701782227], "translation_acceleration_cm_proxy": 2.0942553877830505}, "refined": {"visible_iou_mean": 0.6698492819612677, "visible_iou_median": 0.6769784092903137, "negative_center_depth_frames": 0, "depth_range_m": [3.4019343852996826, 3.917264699935913], "translation_acceleration_cm_proxy": 0.9663893841207027}}}}

## Second submission completed

Updated 2026-10-12T20:25:02.678421+00:00

Completed interrupted Episode 18 tracking, 100-step refinement, official conversion, and comparison. All three pilots passed the conservative local promotion checks. Official packer, 30-pair integrity checks, exact scored-row coverage, baseline copy hashes, environment preflight, and authenticated Kaggle dry-run passed. New package has 740,780 rows over 30 episodes; only Episodes 6, 18, and 28 were replaced. Episode 16 and all first-submission artifacts remain preserved. One new version uploaded to all five competitions; all COMPLETE. No remaining jobs or submission blockers.

Archive: `data/v2d_challenge/track_1/submission_packages/track1_improvement_pilot_v1.parquet`

Receipt: `data/v2d_challenge/track_1/submission_packages/track1_improvement_pilot_v1.kaggle-receipt.json`

Producing-code commit: https://github.com/JavierMedel/video_to_data/commit/811b00735b793033d48047217ab1a41872d7637f

| Metric | Baseline | New | Submission ID | Status |
|---|---:|---:|---|---|
| cd_h_cm | 15.11851 | 14.99887 | 57118072 | COMPLETE |
| cd_o_cm | 152.72099 | 75.48250 | 57118073 | COMPLETE |
| acc_h_cm | 1.62217 | 1.55253 | 57118075 | COMPLETE |
| acc_o_cm | 7.10147 | 5.43410 | 57118076 | COMPLETE |
| interpenetration_cm | 0.00213 | 0.00233 | 57118077 | COMPLETE |

Four errors improved. Object CD fell by 50.57%; object acceleration error fell by 23.48%. Interpenetration increased slightly by 0.00020 cm. Local image/motion diagnostics remain distinct from official hidden-reference leaderboard metrics.
