# v8 geometry risk audit

Diagnostics only; no geometry modified or promoted. Cached image evidence is hash-matched to the exact submitted v8 pairs. Equal episode weighting does not reveal hidden per-episode distance.

| Episode | Object | IoU | Extents m | Area outside largest component | >45 degree jumps |
|---:|---|---:|---|---:|---:|
| 20 | paint_roller | 0.4130 | 0.092, 0.323, 0.326 | 0.7187 | 3 |
| 18 | paint_roller | 0.4148 | 0.092, 0.323, 0.326 | 0.7187 | 37 |
| 19 | paint_roller | 0.4534 | 0.092, 0.323, 0.326 | 0.7187 | 4 |
| 29 | white_laptop_cart | 0.4933 | 0.576, 0.584, 0.822 | 0.1381 | 0 |
| 26 | short_wood_stool | 0.4980 | 0.342, 0.467, 0.532 | 0.0899 | 2 |
| 10 | white_desk | 0.5048 | 0.760, 0.990, 1.149 | 0.1841 | 0 |
| 11 | white_desk | 0.5658 | 0.760, 0.990, 1.149 | 0.1841 | 2 |
| 1 | hula_hoop | 0.5922 | 0.042, 0.809, 0.811 | 0.2930 | 0 |
| 27 | white_laptop_cart | 0.6468 | 0.576, 0.584, 0.822 | 0.1381 | 4 |
| 13 | black_pan | 0.6572 | 0.057, 0.220, 0.394 | 0.0486 | 7 |
| 24 | short_wood_stool | 0.6643 | 0.342, 0.467, 0.532 | 0.0899 | 0 |
| 2 | hula_hoop | 0.6721 | 0.042, 0.809, 0.811 | 0.2930 | 0 |
| 28 | white_laptop_cart | 0.6812 | 0.576, 0.584, 0.822 | 0.1381 | 4 |
| 25 | short_wood_stool | 0.6859 | 0.342, 0.467, 0.532 | 0.0899 | 10 |
| 7 | iron | 0.7055 | 0.106, 0.161, 0.326 | 0.0404 | 48 |
| 8 | iron | 0.7068 | 0.106, 0.161, 0.326 | 0.0404 | 31 |
| 12 | black_pan | 0.7266 | 0.057, 0.220, 0.394 | 0.0486 | 7 |
| 0 | hula_hoop | 0.7288 | 0.042, 0.809, 0.811 | 0.2930 | 0 |
| 16 | foam_grass_block | 0.7337 | 0.210, 0.211, 0.212 | 0.4805 | 2 |
| 9 | white_desk | 0.7365 | 0.760, 0.990, 1.149 | 0.1841 | 0 |
| 14 | black_pan | 0.7407 | 0.057, 0.220, 0.394 | 0.0486 | 1 |
| 21 | pink_foam_roll | 0.7757 | 0.140, 0.141, 0.595 | 0.0000 | 14 |
| 22 | pink_foam_roll | 0.7765 | 0.156, 0.157, 0.661 | 0.0000 | 33 |
| 23 | pink_foam_roll | 0.7991 | 0.140, 0.141, 0.595 | 0.0000 | 13 |
| 6 | iron | 0.8320 | 0.106, 0.161, 0.326 | 0.0404 | 20 |
| 17 | foam_grass_block | 0.8546 | 0.210, 0.211, 0.212 | 0.4805 | 2 |
| 4 | big_red_bowl | 0.8569 | 0.133, 0.318, 0.319 | 0.0099 | 9 |
| 3 | big_red_bowl | 0.8976 | 0.133, 0.318, 0.319 | 0.0099 | 20 |
| 15 | foam_grass_block | 0.8986 | 0.210, 0.211, 0.212 | 0.4805 | 7 |
| 5 | big_red_bowl | 0.9054 | 0.133, 0.318, 0.319 | 0.0099 | 30 |

## Hoop precheck

Bounding extents are not a fitted ring diameter. The same-label metric hoop must be checked for physical identity, local pose/origin, thin-tube geometry and image agreement before substitution.
- Episode 0: IoU 0.7288, scaled extents [0.041666413832585084, 0.8088169701279165, 0.8105755007189505], mesh SHA `ef2f782683e412e74eabe62273abb7ebe6eb26b28fc4fdf6f5c92f445330687d`.
- Episode 1: IoU 0.5922, scaled extents [0.041666413832585084, 0.8088169701279165, 0.8105755007189505], mesh SHA `ef2f782683e412e74eabe62273abb7ebe6eb26b28fc4fdf6f5c92f445330687d`.
- Episode 2: IoU 0.6721, scaled extents [0.041666413832585084, 0.8088169701279165, 0.8105755007189505], mesh SHA `ef2f782683e412e74eabe62273abb7ebe6eb26b28fc4fdf6f5c92f445330687d`.
