# Official v8 submission results

The complete 30-episode campaign and official submission are complete. All five uploads were accepted and scored successfully, using one candidate version. Twelve episodes use reviewed contact-constrained translations and eighteen retain v6. All other numeric packed values remain identical to v6.

| Official metric (cm; lower is better) | v6 | v8 |
|---|---:|---:|
| Human distance | 15.43955 | 15.43955 |
| Object distance | 23.48489 | 23.48324 |
| Human acceleration | 1.15894 | 1.15894 |
| Object acceleration | 0.81013 | 0.71908 |
| Penetration | 0.00325 | 0.00322 |

Object acceleration improves by 11.24%. Object distance improves by only 0.00165 cm (about 0.007%), and penetration improves by 0.00003 cm. Both human metrics are unchanged. None of the five reported metrics regress against v6. The main gain is smoother object motion; reconstruction-distance improvement is small. The recorded CARI4D object-distance reference of 23.16246 remains better than this candidate.

All scores are recorded with the exact submission IDs in `submission_receipt.json`. No duplicate uploads were made while collecting the final score. The live receipt is `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.kaggle-receipt.json`.

Producing code: https://github.com/JavierMedel/video_to_data/commit/b29d19010a14c39c5402ae7a8d0420bf6e0c925e . The 740,780-row archive SHA256 is `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`.
