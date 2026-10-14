# Why the v4 penetration gate did not match official PEN

The released `metric_code/track_1/PEN.py` computes hand vertices inside the submitted object mesh. Its `_episode_penetration` transforms predicted hand vertices into object-local coordinates, classifies containment with absolute generalized winding number > 0.5, computes distance to object triangles, and averages penetration over hand vertices and scored frames. It multiplies by object scale and the first-frame human similarity scale.

The prior v4 local gate measured sampled object-surface points inside a predicted full-body collision proxy. That is the opposite containment direction with a different sampled surface. It can improve while fingers penetrate farther into the object. The official PEN increase from 0.00318 to 0.00331 therefore was not ruled out by the old gate.

Correction to the earlier v4 interpretation: official PEN uses the predicted hands and submitted object; hidden reference geometry supplies the positive human alignment scale. It does not directly test the submitted object against a hidden reference body surface.

New review: generalized winding and triangle distance on the actual submitted human decoder and object mesh. No watertightness assumption or ray-parity shortcut is substituted for winding on open objects. Numerical tests compare directly with the released metric on synthetic closed/open/reversed/degenerate meshes.

The participant copy omits the exact `role_left_hand` and `role_right_hand` IDs, and they are absent from the public Track 1 dataset metadata. We therefore report all 4636 anatomically selected hand vertices and an explicitly local 512-point FPS subset. Neither is labelled an official score. Both mean depths must be non-increasing; all-scored-frame review follows the initial screening. A pointwise-increase diagnostic is also retained.

Because all human arrays are unchanged, the hidden positive alignment scale is unchanged per episode. With the exact host hand selection, non-increasing unaligned penetration within an episode would imply non-increasing official PEN. The remaining subset mismatch prevents claiming this guarantee from our local means alone.
