# Track 1 submission recovery

Updated 2026-10-11T22:46:29.772059+00:00

Branch: v2d-track1-blackwell. No inference or upload performed during recovery. Episode 16 untouched.

Verified pairs: 30/30. Pipelines with all eight marker output identities verified: 6/30.

| Episode | Object | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | Mesh | NPZ | GLB | Next Action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| episode_000000 | hula_hoop | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000001 | hula_hoop | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000002 | hula_hoop | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000003 | big_red_bowl | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000004 | big_red_bowl | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000005 | big_red_bowl | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000006 | iron | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000007 | iron | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000008 | iron | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000009 | white_desk | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000010 | white_desk | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000011 | white_desk | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000012 | black_pan | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000013 | black_pan | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000014 | black_pan | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000015 | foam_grass_block | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000016 | foam_grass_block | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | COMPLETE | COMPLETE | COMPLETE | Protected; never rerun without approval |
| episode_000017 | foam_grass_block | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | MISSING | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000018 | paint_roller | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000019 | paint_roller | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000020 | paint_roller | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000021 | pink_foam_roll | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000022 | pink_foam_roll | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000023 | pink_foam_roll | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000024 | short_wood_stool | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000025 | short_wood_stool | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000026 | short_wood_stool | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000027 | white_laptop_cart | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000028 | white_laptop_cart | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |
| episode_000029 | white_laptop_cart | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | COMPLETE | MISSING | COMPLETE | COMPLETE | COMPLETE | Preserve; pair ready |

COMPLETE stages require matching output sizes and nanosecond timestamps from stage signatures; NPY headers were opened. This verifies preserved checkpoint identities, not full tensor contents. Episode 16 uses legacy artifacts without stage signatures and is UNKNOWN for stage completion. Mesh COMPLETE means loadable finite triangle geometry; metre scale still requires checking against source transform/calibration before inference. Candidate meshes are shared only within their same-object episode triplet.

## Blockers and next execution

- Docker CLI reports missing WSL integration both inside and outside sandbox. `/var/run/docker.sock`, Docker Desktop CLI mount, and Windows Docker executable are unavailable. GPU is visible outside sandbox: RTX 5090 Laptop, 24463 MiB. No host configuration changed.
- Kaggle authenticated successfully outside sandbox; all five Track 1 competitions entered. No upload issued.
- Host Python lacks torch and h5py. Existing refined PTH conversion cannot run here without the normal Docker environment or added dependencies.
- Finish converting episodes 3–5, resume episode 15 at S07 with its existing stage signatures and original command settings, then process remaining missing episodes sequentially using Blackwell images. Preserve every existing stage.
- Official format is one Parquet file for five metric competitions. No ZIP required. One authorized submission version only.
- Producing-code GitHub commit must include actual conversion/pipeline changes and be accessible to organizers; current producing worktree is dirty. Do not invent a provenance URL.

## Recorded failure

Docker availability command: `docker ps --format ...`; CLI prints "The command docker could not be found in this WSL 2 distro". GPU command in sandbox exited 255; escalation succeeded. Kaggle entry check in sandbox exited 1 with DNS failure; escalated check exited 0 with all five competitions entered. No inference stage failed in this recovery turn.
