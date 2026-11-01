import results from './results.json';
import provenance from './provenance.json';
import coverage from './coverage.json';

export type Track = 'track_1' | 'track_2_tier1' | 'track_2_tier2' | 'track_2_tier3' | 'track_3';
export interface Metric {
  id: string; key: string; label: string; name: string; description: string; track: Track;
  value: number | null; unit: string | null; direction: 'higher' | 'lower';
  classification: 'official' | 'local'; scope: string; aggregation: string;
  run: string | null; submissionId: number | null; timestamp: string | null;
  source: string; definition: string;
}
export interface Media {
  id: string; type: 'video' | 'image' | 'mesh'; title: string; episode: string;
  stage: string; src: string | null; poster: string | null; caption: string; alt: string;
  availability: 'available' | 'unavailable'; status: 'Verified artifact' | 'Historical diagnostic' | 'Preview unavailable';
  sourcePath: string; attribution: string; permission: string;
  frameRange: [number, number] | null; sourceFps: number | null; playbackFps?: number; offset: number | null;
  run: string; gallery?: boolean;
}
export interface Episode {
  id: string; title: string; track: string; object: string; frames: number | null;
  fps: number | null; resolution: [number, number] | null; status: string;
  media: string[]; mesh: string | null; officialMetrics: string[];
  localDiagnostics: { label: string; value: number | null; unit: string | null; scope: string; source: string }[];
  issues: string[]; sourcePath: string; run: string; verificationDate: string;
}
export interface Stage {
  id: string; name: string; tool: string; purpose: string; input: string; output: string;
  uncertainty: string; branch: string; source: string;
}
export interface Observation {
  id: string; title: string; observation: string; evidence: string;
  hypothesis: string; experiment: string; media: string | null; source: string;
}

export const project = { title: 'Video to Data', subtitle: 'Reconstructing human–object interaction', author: 'Javier Medel, MSc.', ...provenance };
export const metrics = results as Metric[];
export { coverage };
export const asset = (path: string) => `${import.meta.env?.BASE_URL ?? '/video_to_data/'}${path}`;
// Research caches are ignored by Git. Point to the published provenance ledger,
// rather than inventing a GitHub URL for a file that was never committed.
export const repoFile = (path: string) => path.startsWith('reconstruction/data/')
  ? asset('evidence/asset-inventory.json')
  : path === 'website/README.md' ? asset('evidence/README.md')
  : `${project.links.project}/blob/${project.commit}/${path}`;
const ep15 = 'reconstruction/data/v2d_challenge/track_1/episode_000015';
const permission = 'Display derivative of challenge research output. Dataset: NVIDIA, CC BY 4.0. Upstream model and code licenses apply separately.';
const attribution = 'NVIDIA V2D Challenge dataset; output generated in Javier Medel’s project checkout.';
const mediaBase = { attribution, permission, availability: 'available' as const, status: 'Verified artifact' as const, offset: null };

export const media: Media[] = [
  { ...mediaBase, id: 't1-comparison', type: 'video', title: 'Human + object reconstruction', episode: 't1-e01', stage: 'Reconstruction', src: 'media/t1-comparison.mp4', poster: 'media/t1-comparison.webp', caption: 'CARI4D’s saved three-panel comparison: initialization, CoCoNet prediction, and contact-guided refinement. This early stability run is separate from the later v8 scored package. All panels are preserved; rendering is evidence of a generated reconstruction, not ground-truth accuracy.', alt: 'Three saved CARI4D views show a reconstructed human stepping through a hula hoop.', sourcePath: 'reconstruction/data/v2d_challenge/track_1/cari4d_runs/episode_000001_stability/episode_000001/inference/episode_000001_step200000_before_after.mp4', frameRange: [0, 667], sourceFps: 30, run: 'episode_000001_stability · step200000', gallery: true },
  { ...mediaBase, id: 'e15-input', type: 'video', title: 'The source demonstration', episode: 't1-e15', stage: 'RGB input', src: 'media/e15-input.mp4', poster: 'media/e15-input.webp', caption: 'Original third-person demonstration, reduced to 960 px width for playback. Full frame range and source timing preserved. Source-video frame rate is not inference throughput.', alt: 'A person rotates a green foam grass block in a laboratory.', sourcePath: `${ep15}/2026-09-10_16-14-06_foam_grass_block_rotate_02.0.color.mp4`, frameRange: [0, 500], sourceFps: 30, run: 'foam_grass_block_rotate_02' },
  { ...mediaBase, id: 'e15-segmentation', type: 'image', title: 'Isolating the object', episode: 't1-e15', stage: 'Segmentation', src: 'media/e15-segmentation.webp', poster: 'media/e15-segmentation.webp', caption: 'SAM2 object mask at frame 154, composited in translucent green over that exact RGB frame. Identifies object pixels; does not reconstruct the human body.', alt: 'Frame 154 with the SAM2 block mask highlighted in green.', sourcePath: `${ep15}/object_masks/1/000154.png`, frameRange: [154, 154], sourceFps: 30, run: 'Episode 15 cached SAM2 masks' },
  { ...mediaBase, id: 'e15-depth', type: 'image', title: 'Estimating scene depth', episode: 't1-e15', stage: 'Depth', src: 'media/e15-depth.webp', poster: 'media/e15-depth.webp', caption: 'MoGe depth at frame 154. Grayscale normalized between the positive-value 2nd and 98th percentiles: dark is nearer, light is farther; invalid pixels are black. Estimated relative depth, not calibrated ground truth.', alt: 'Grayscale relative-depth visualization of the laboratory at frame 154.', sourcePath: `${ep15}/moge/depth/000154.png`, frameRange: [154, 154], sourceFps: 30, run: 'Episode 15 cached MoGe depth' },
  { ...mediaBase, status: 'Historical diagnostic', id: 'e15-scale', type: 'image', title: 'Inspecting the mesh projection', episode: 't1-e15', stage: 'Pose overlay', src: 'media/e15-scale.webp', poster: 'media/e15-scale.webp', caption: 'Saved scale-diagnostic mesh projection on source RGB. The stored scale multiplier is 0.38555270635198524. It is a fitting parameter, not an official metric or proof of metric scale.', alt: 'Saved green mesh projection over the foam block in the original scene.', sourcePath: `${ep15}/sam3d/scale_diagnostic_overlay.png`, frameRange: null, sourceFps: null, run: 'Episode 15 scale diagnostic · October 9 context', gallery: true },
  { ...mediaBase, id: 'e15-mesh', type: 'mesh', title: 'A surface from a single view', episode: 't1-e15', stage: 'Object mesh', src: 'media/e15-display.glb', poster: 'media/e15-mesh.webp', caption: 'Real SAM3D object mesh. The viewer uses the preserved simplified backup: 3,156 vertices / 5,316 faces. Original research mesh: 254,734 vertices / 381,436 faces. Poster uses neutral flat shading. Mesh units have not been validated as metres.', alt: 'Flat-shaded rendering of the reconstructed foam-block surface.', sourcePath: `${ep15}/sam3d/object_simplified_backup.glb`, frameRange: null, sourceFps: null, run: 'Episode 15 SAM3D cached output', gallery: true },
  { ...mediaBase, id: 't1-masks', type: 'image', title: 'Reviewing segmentation quality', episode: 't1-e18', stage: 'Segmentation', src: 'media/t1-masks.webp', poster: 'media/t1-masks.webp', caption: 'Saved Episode 18 corrected-mask review sheet from the reconstruction improvement pilot. This is a diagnostic contact sheet; it is not a sequence-wide accuracy score.', alt: 'Saved contact sheet reviewing corrected person and object masks for Episode 18.', sourcePath: 'reconstruction/incident_reports/track1_improvement/episode_000018_corrected_masks.jpg', frameRange: null, sourceFps: null, run: 'improvement_pilot_v1', gallery: true },
  { ...mediaBase, id: 't3-hands', type: 'video', title: 'Hands from an egocentric view', episode: 't3-e38', stage: 'Hand reconstruction', src: 'media/t3-hands.mp4', poster: 'media/t3-hands.webp', caption: 'Saved HaMeR depth-aligned hand overlay from the Episode 38 pilot. An intermediate reconstruction diagnostic, separate from the scored end-to-end policy packet. It does not show robot-policy success. The input video is 20 FPS; this saved overlay is encoded at 30 FPS. Player time follows the saved output, not the source-video clock.', alt: 'Egocentric video with depth-aligned reconstructed hand overlays.', sourcePath: 'reconstruction/data/track3/pilot/episode_000038/hamer_depth_aligned_overlay.mp4', frameRange: [0, 262], sourceFps: 20, playbackFps: 30, run: 'Track 3 Episode 38 HaMeR pilot', gallery: true },
  { ...mediaBase, id: 't2-rollout', type: 'video', title: 'Robot policy rollout', episode: 't2-cohort', stage: 'Policy evaluation', src: null, poster: null, caption: 'Scored trajectory exports and policy receipts are present. A browser-ready robot rollout video was not found in this checkout. Add an attributable, matched rollout from the scored campaign to enable this preview.', alt: 'Preview not yet available.', sourcePath: 'robotic_grounding/incident_reports/track2_tier2/v1_campaign', frameRange: null, sourceFps: null, run: 'track2_tier2.csv', availability: 'unavailable', status: 'Preview unavailable', gallery: true },
  { ...mediaBase, status: 'Historical diagnostic', id: 'e15-failure', type: 'image', title: 'When the projection drifts', episode: 't1-e15', stage: 'Pose diagnostic', src: 'media/e15-failure.webp', poster: 'media/e15-failure.webp', caption: 'Original saved pose-check visualization at frame 212. Colored silhouettes separate in this diagnostic. The independently registered frame 212 file is a separate experiment and was excluded from the motion chart.', alt: 'Saved frame 212 diagnostic shows displaced green and red object silhouettes on black.', sourcePath: `${ep15}/sam3d/pose_check_000212.png`, frameRange: [212, 212], sourceFps: 30, run: 'Original FoundationPose scale-diagnostic poses' },
];

export const episodes: Episode[] = [
  { id: 't1-e01', title: 'Track 1 · Episode 01', track: 'Track 1', object: 'hula_hoop', frames: 668, fps: 30, resolution: [1536, 1152], status: 'Saved CARI4D render', media: ['t1-comparison'], mesh: null, officialMetrics: [], localDiagnostics: [], issues: ['This early stability render is not the v8 scored packet.'], sourcePath: 'reconstruction/data/v2d_challenge/track_1/meta/episodes_metadata.jsonl', run: 'episode_000001_stability', verificationDate: '2026-11-01' },
  { id: 't1-e15', title: 'Track 1 · Episode 15', track: 'Track 1', object: 'foam_grass_block', frames: 501, fps: 30, resolution: [1536, 1152], status: 'Cached object workflow + diagnostics', media: ['e15-input', 'e15-segmentation', 'e15-depth', 'e15-scale', 'e15-mesh', 'e15-failure'], mesh: 'e15-mesh', officialMetrics: [], localDiagnostics: [{ label: 'Pose files', value: coverage.poseFiles, unit: 'files', scope: 'Processing coverage, not tracking accuracy', source: `${ep15}/foundationpose_scale_diagnostic` }], issues: ['Pose discontinuities', 'Unvalidated metric scale', 'Single-frame fit does not establish full-sequence accuracy'], sourcePath: `${ep15}/sam3d/scale_diagnostic.json`, run: 'Original scale-diagnostic workflow', verificationDate: '2026-11-01' },
  { id: 't1-e18', title: 'Track 1 · Episode 18', track: 'Track 1', object: 'paint_roller', frames: null, fps: null, resolution: null, status: 'Mask review', media: ['t1-masks'], mesh: null, officialMetrics: [], localDiagnostics: [], issues: ['Local review only'], sourcePath: 'reconstruction/incident_reports/track1_improvement/progress.md', run: 'improvement_pilot_v1', verificationDate: '2026-11-01' },
  { id: 't2-cohort', title: 'Track 2 · Tier 2 campaign', track: 'Track 2', object: 'Multiple objects', frames: null, fps: null, resolution: null, status: 'Scored trajectory export; video unavailable', media: ['t2-rollout'], mesh: null, officialMetrics: metrics.filter(m => m.track === 'track_2_tier2').map(m => m.id), localDiagnostics: [], issues: ['Tier 3 has no scored submission in the saved snapshot.'], sourcePath: 'reconstruction/incident_reports/cross_track_validation_plan/completion_20261025_v1.json', run: 'track2_tier2.csv', verificationDate: '2026-11-01' },
  { id: 't3-e38', title: 'Track 3 · Episode 38 pilot', track: 'Track 3', object: 'See source scene', frames: 263, fps: 20, resolution: [2028, 1520], status: 'Saved intermediate hand overlay', media: ['t3-hands'], mesh: null, officialMetrics: [], localDiagnostics: [], issues: ['Hand-overlay diagnostic does not establish policy success.'], sourcePath: 'reconstruction/data/track3/pilot/episode_000038/hamer_depth_aligned_overlay.mp4', run: 'HaMeR pilot', verificationDate: '2026-11-01' },
];

export const tracks = [
  { number: '01', title: 'Reconstruction', input: 'Third-person video → 4D interaction', body: 'Estimate the human, object surface, and motion from a single camera. Inspect generated meshes and poses through time.', status: 'Scored · v8 package', metric: 'CD-O', value: '23.48324', unit: 'cm ↓', link: 'track_1' as Track },
  { number: '02', title: 'Robotic grounding', input: 'Human motion → robot policy', body: 'Retarget demonstrations to G1 + Dex3 and evaluate learned object tracking under clean and corrupted reference trajectories.', status: 'Tiers 1 + 2 scored · Tier 3 unavailable', metric: 'AUC · Tier 2', value: '0.22330', unit: '↑', link: 'track_2_tier2' as Track },
  { number: '03', title: 'Egocentric transfer', input: 'First-person video → robot behavior', body: 'Explore the path from hand and object reconstruction to retargeting and policy learning. Measure the final behavior separately.', status: 'Scored · end-to-end packet', metric: 'AUC', value: '0.16307', unit: '↑', link: 'track_3' as Track },
];

export const stages: Stage[] = [
  { id: 'detect', name: 'Find people + objects', tool: 'Grounding DINO · Detectron2', purpose: 'Locate the target object and establish person tracks.', input: 'RGB frames + object prompt', output: 'Bounding boxes and person tracks', uncertainty: 'Missed or switched detections propagate downstream.', branch: 'Shared RGB input', source: 'reconstruction/README.md' },
  { id: 'segment', name: 'Separate the pixels', tool: 'SAM2', purpose: 'Identify pixels belonging to the human or object.', input: 'RGB + detection prompts', output: 'Human and object masks', uncertainty: 'Occlusion and distractors can corrupt the mask. SAM2 performs segmentation, not 3D reconstruction.', branch: 'Detection → masks', source: 'reconstruction/modules/v2d_sam2' },
  { id: 'depth', name: 'Estimate scene depth', tool: 'MoGe', purpose: 'Estimate depth and camera intrinsics: the parameters describing how the camera projects 3D points.', input: 'RGB frames, in parallel with masks', output: 'Depth maps + estimated intrinsics', uncertainty: 'Monocular depth can have scale ambiguity and frame-to-frame changes.', branch: 'Parallel RGB → depth', source: 'reconstruction/modules/v2d_moge' },
  { id: 'mesh', name: 'Recover a surface', tool: 'SAM3D', purpose: 'Reconstruct the object’s surface from a selected masked image.', input: 'Reference RGB + object mask', output: 'Textured GLB mesh', uncertainty: 'Hidden surfaces are inferred. Silhouette agreement alone does not validate geometry.', branch: 'Masks + RGB → object geometry', source: 'reconstruction/modules/v2d_sam3d' },
  { id: 'pose', name: 'Track the object', tool: 'FoundationPose', purpose: 'Estimate the object’s position and orientation over time.', input: 'Mesh + RGB + masks + depth + intrinsics', output: 'Per-frame object poses', uncertainty: 'Symmetry, occlusion, and depth changes can cause discontinuities.', branch: 'Object branches converge', source: 'reconstruction/modules/v2d_foundation_pose' },
  { id: 'human', name: 'Reconstruct interaction', tool: 'CARI4D · CoCoNet', purpose: 'Generate and refine human–object interaction. The saved CARI4D render demonstrates a human body output independently of human masks.', input: 'Video, masks, geometry and initial estimates', output: 'Human–object reconstruction + rendered comparisons', uncertainty: 'Contact refinement and coherent renders do not establish ground-truth accuracy.', branch: 'Human–object interaction branch', source: 'reconstruction/modules/v2d_cari4d/README.md' },
  { id: 'export', name: 'Review + export', tool: 'Local validation · submission kit', purpose: 'Inspect image agreement, pose continuity, contact constraints and artifact integrity before packaging.', input: 'Reconstructed sequences', output: 'Versioned packets, review sheets and diagnostics', uncertainty: 'Local proxy improvements can fail to improve official scores.', branch: 'Track 1 output', source: 'reconstruction/incident_reports/track1_improvement/results.md' },
  { id: 'policy', name: 'Transfer + evaluate', tool: 'Retargeting · FlashCHORD', purpose: 'Track 2 starts from supplied trajectories. Track 3 starts from egocentric reconstruction. Both continue to robot-policy evaluation.', input: 'Track-specific motion references', output: 'Policy rollouts + scored trajectory packets', uncertainty: 'Reference quality, contact and simulation repeatability limit transfer. Track 2 Tier 3 remains unfinished.', branch: 'Tracks 2 + 3', source: 'robotic_grounding/flash_chord/README.md' },
];

export const observations: Observation[] = [
  { id: 'jumps', title: 'A pose file can still be wrong.', observation: 'Episode 15 contains saved poses for every source frame, yet adjacent-pose differences expose abrupt changes around frames 212–213 and 401–402.', evidence: 'The motion plot reads the original numbered JSON poses. The frame 212 projection check is shown here.', hypothesis: 'Occlusion, ambiguous orientation or depth changes may destabilize tracking. The saved outputs do not isolate one cause.', experiment: 'Evaluate visible-mask overlap and depth support around each jump; test reinitialization separately from the original trajectory.', media: 'e15-failure', source: `${ep15}/foundationpose_scale_diagnostic` },
  { id: 'proxy', title: 'Smoother does not mean more accurate.', observation: 'The v7 review reports lower object acceleration alongside worse official object reconstruction distance. Later v8 scores are reported separately above.', evidence: 'The preserved v7 results compare the same submitted cohorts and document rejected contact corrections.', hypothesis: 'A correction can reduce motion variation while shifting an object away from its true depth.', experiment: 'Separate translation smoothing from depth offsets and retain contact and image-agreement gates.', media: null, source: 'reconstruction/incident_reports/track1_improvement/results.md' },
  { id: 'repeat', title: 'Policy selection needs repeatability.', observation: 'The Tier 1 checkpoint reselection pilot failed its repeatability gate. The existing scored packet was retained.', evidence: 'The October 22 execution record documents diverging repeated rollouts. No improved packet was promoted.', hypothesis: 'The experiment did not identify the root cause of simulation divergence.', experiment: 'Measure repeated-rollout variance before selecting checkpoints or claiming gains. Tier 3 still needs genuine reconstruction and a validated adapter.', media: null, source: 'V2D_NEXT_EXECUTION_AND_VALIDATION_PLAN.md' },
];

export const acknowledgments = [
  'NVIDIA for the V2D toolkit, challenge, and research dataset.',
  'The authors and maintainers of Grounding DINO, Detectron2, SAM2, MoGe, SAM3D, FoundationPose, CARI4D, CoCoNet, HaMeR, and the upstream robotic-grounding tools.',
];
