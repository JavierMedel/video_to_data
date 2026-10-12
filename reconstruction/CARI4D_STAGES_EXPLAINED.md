# CARI4D pipeline stages in plain language

This guide explains the eight numbered stages recorded by this repository's
CARI4D video reconstruction pipeline. In simple terms, the pipeline uses a
video, an object mask, and an object mesh to estimate how a person and an object
move through the scene.

## Inputs used by the whole pipeline

For each episode, the runner starts with these main inputs:

- **Color video:** the episode video, named like
  `episode_000000.0.color.mp4`.
- **Object masks:** an HDF5 (`.h5`) file marking the object in video frames.
- **Object mesh:** a 3D model of the object, such as a `.glb` file.
- **Model files:** already-installed model weights and configuration files for
  depth estimation, human-body fitting, object tracking, and CARI4D.

The output root below means the per-episode directory created by the runner,
for example `.../episode_000000/episode_000000/`. Exact parent directories vary
with the run command and chosen output location.

## What each stage does

| Stage | Name in the run | In plain words | Main inputs | Main outputs |
|---|---|---|---|---|
| 01 | `depth` | Estimates how far visible parts of each video frame are from the camera and records camera information for later steps. | Color video; MoGe2 model files. | `depth/moge2/raw.npy` (depth maps); `depth/moge2/intrinsics.pkl` (camera information); `depth/moge2/depth_report.json` (run details). |
| 02 | `export` | Prepares the video, object mask, and object mesh in a consistent scene coordinate system. | Color video; object-mask `.h5`; input object mesh; camera intrinsics from Stage 01. | `export/<episode>/wild_export.json` (prepared scene data); `export/<episode>/object_mesh/output_aligned.glb` (aligned mesh). |
| 03 | `human_initialization` | Makes an initial estimate of the person's 3D body and pose from the video. This is a starting point for later fitting. | Prepared scene from Stage 02; SAM3D checkpoint; MHR body model files. | `mhr_init/<episode>.pkl` (initial body estimate); `mhr_init/<episode>.sam3d.h5` (cached SAM3D data). |
| 04 | `depth_alignment` | Brings the depth maps into agreement with the prepared scene and initial human estimate, so later steps use compatible coordinates. | Prepared scene; raw depth and report from Stage 01; human initialization from Stage 03. | `depth/moge2/aligned.h5` (depth aligned to the scene). |
| 05 | `object_pose` | Uses FoundationPose to estimate where the supplied object mesh is, and how it is oriented, in each frame. | Prepared scene; aligned depth from Stage 04; FoundationPose model files. | `foundationpose/<episode>.pkl` (initial object poses across frames). |
| 06 | `coconet` | Combines the initial human estimate, object tracking, depth, and video evidence to estimate the human-and-object motion sequence. | Prepared scene; aligned depth; human initialization; object poses; CARI4D configuration and checkpoint. | `inference/coconet.pth` (initial reconstruction); `inference/materialized_inputs.h5` (cached model inputs). |
| 07 | `refinement` | Improves the initial sequence by adjusting estimates to fit the observations and make motion more consistent over time. | Initial reconstruction from Stage 06; aligned object mesh from Stage 02; MHR model files. | `inference/refined.pth` (refined reconstruction); `inference/refinement_checkpoint.pth` (refinement state). |
| 08 | `render` | Draws videos from the reconstruction so the result can be inspected visually. | Color video; prepared scene; aligned object mesh; initial bundle from Stage 06; refined bundle from Stage 07. | `inference/<episode>_step200000_reconstruction.mp4`; `inference/<episode>_step200000_before_after.mp4` (comparison video). |

## How to read a stage status

- A completed stage means the pipeline recorded the expected output for that
  step. It does not guarantee that every estimate is accurate.
- A stage marked as **reused** means its existing output was accepted, so the
  pipeline did not need to run that step again.
- A stage with no completion marker is **not verified as complete**. There may
  still be useful intermediate files, but their presence alone does not prove
  the stage finished successfully.
- Stage 08 is mainly for visual review. The Track 1 submission still needs the
  official per-episode parameter file and object mesh in the required format.
  A successful CARI4D run by itself is not a complete Kaggle submission.

## How this relates to a Track 1 submission

The eight stages produce reconstruction estimates and supporting files. The
submission preparation step must then convert the relevant human estimates into
the official MHR parameter layout, provide the object's per-frame pose and mesh,
and pass the official packer for **every required episode**. The official
validator and final upload are separate steps after reconstruction.

In short: stages 01–08 create and refine the reconstruction; packaging converts
it into the competition's required format; validation checks the package; and
Kaggle submission uploads it.
