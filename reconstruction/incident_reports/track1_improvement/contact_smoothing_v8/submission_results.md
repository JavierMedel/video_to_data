# Official v8 submission results

The complete 30-episode campaign and official submission finished. All five uploads were accepted, using one candidate version. Twelve episodes use reviewed contact-constrained translations and eighteen retain v6. All other numeric packed values remain identical to v6.

| Official metric (cm; lower is better) | v6 | v8 |
|---|---:|---:|
| Human distance | 15.43955 | 15.43955 |
| Object distance | 23.48489 | 23.48324 |
| Human acceleration | 1.15894 | 1.15894 |
| Object acceleration | 0.81013 | Pending |
| Penetration | 0.00325 | 0.00322 |

Object distance improves by only 0.00165 cm (about 0.007%); this is a small gain. Penetration improves by 0.00003 cm. Human metrics are unchanged. The recorded CARI4D object-distance reference of 23.16246 remains better than this candidate. Local smoothing proxies did not predict a large distance improvement.

Object acceleration is still processing. No claim of overall non-regression is made until that score is available. The existing official uploader continues polling the saved exact submission IDs; do not upload this version again. `submission_receipt.json` is a snapshot; the live receipt is `reconstruction/data/v2d_challenge/track_1/submission_packages/track1_contact_smoothing_v8.kaggle-receipt.json`.

Producing code: https://github.com/JavierMedel/video_to_data/commit/b29d19010a14c39c5402ae7a8d0420bf6e0c925e . The 740,780-row archive SHA256 is `bd73a931e9f9ec85b920884415c5dc2123d86e2d3d91395c7820ebf564fd3198`.
