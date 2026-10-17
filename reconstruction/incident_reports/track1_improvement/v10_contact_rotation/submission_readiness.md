# v10 submission readiness

**Ready locally; not uploaded.** All preparation authorized by the user is complete. The remaining issue is challenge submission eligibility, not another request for permission.

- Archive: `/home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_v10_contact_rotation.parquet`.
- SHA256: `78d56ea2d8bbc63784aff01e4642625d6ddbee8d2aca560a91a4173509c8e87a`.
- Complete 30 episodes / 740,780 rows / 15,708,468 bytes.
- Only Episode 18 object rotations differ from v8: 657 packed rows. All other numeric values are exactly identical to v8.
- Producing commit (verified publicly accessible): https://github.com/JavierMedel/video_to_data/commit/8feab58e16031461639d7e3a1ec7dbef1705c3a3 .
- Official packing, exact parent row comparison, environment preflight and authenticated uploader dry-run: **PASS**.
- 24 unit tests pass; GPU mask/grid parity matches source-extracted upstream helpers at both test sizes.
- Two 535-frame source-RGB actual-object projection/crop videos decoded completely; sampled visual review and independent 33-frame hand recomputation pass. No reconstructed-human video or exhaustive visual-review claim.

## Why upload is held

The [official challenge website](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#submission) states five submissions per week. The authenticated history check found **seven versions this calendar week**. The dry-run reports five remaining **daily** submissions per metric; this does not supersede the published weekly policy. No new upload was made and no duplicate receipt/guard was created.

The user already authorized a reviewed submission. Before uploading, establish an organizer-confirmed exception/revised limit or the next eligible submission window. A clarification was requested during preparation; no confirming answer has arrived. Recheck the current policy and actual history at that time rather than assuming that an elapsed time or daily reset grants eligibility. Do not contact organizers or send external messages without user authorization.

## Resume after eligibility is established

1. Recheck archive SHA256, public producing code, all required review evidence and absence of a v10 receipt/guard.
2. Run the official dry-run again and verify the applicable policy/quota.
3. Reserve one exclusive upload-start guard, then use the official uploader with this exact archive and a new exclusive receipt:

```bash
cd /home/alien51/video_to_data/reconstruction/v2d_submission_kit/v2d_submission_kit
../../.venv/bin/python submit/v2d_submit.py \
  /home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_v10_contact_rotation.parquet \
  --track track_1 \
  -m 'v10: contact-constrained Episode18 rotations; complete30 v8-parent package' \
  --wait \
  --receipt /home/alien51/video_to_data/reconstruction/data/v2d_challenge/track_1/submission_packages/track1_v10_contact_rotation.kaggle-receipt.json
```

4. Poll only saved exact IDs. Never re-upload an accepted or ambiguous metric blindly.
5. Record all five real official scores versus v8 and update the handoff. Until then v8 remains the latest submitted version; v10 has no official score.
