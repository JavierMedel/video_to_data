from pathlib import Path
import v2d.cari4d.docker.run_inference as runner

root = Path.cwd()
runner.IMAGE_NAME = "v2d_cari4d:blackwell"

runner.run_inference(
    video_path=str(root / "data/v2d_challenge/track_1/episode_000015/2026-09-10_16-14-06_foam_grass_block_rotate_02.0.color.mp4"),
    mask_h5_path=str(root / "data/v2d_challenge/track_1/episode_000015/cari4d_masks.h5"),
    object_mesh_path=str(root / "data/v2d_challenge/track_1/episode_000015/sam3d/object.glb"),
    weights_path=str(root / "data/weights/cari4d_episode15_overlay"),
    output_dir=str(root / "data/v2d_challenge/track_1/cari4d_runs/episode_000015_test"),
    download_models=False,
    expected_frames=501,
    device="cuda",
    postopt_num_steps=30,
    overwrite=False,
)
