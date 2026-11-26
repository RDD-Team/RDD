# RoadSight — Synthesis v0.4

## Status

**REVIEW ONLY — NOT APPROVED FOR TRAINING**

This report documents the validation of the
object-centric road debris synthesis pilot.

It does not demonstrate model performance improvement.

## Pilot Dataset

| Metric | Result |
|---|---:|
| Synthetic images | 27 |
| YOLO label files | 27 |
| Unique backgrounds | 9 |
| Unique foreground objects | 13 |
| Original object labels | 12 |
| Synthetic object labels | 27 |
| Total object labels | 39 |

## Validation Results

| Check | Result |
|---|---|
| Image-label pairing | PASS |
| Image SHA-256 | PASS |
| Label SHA-256 | PASS |
| Original-label preservation recorded | 27/27 |
| Team-pack provenance | 27/27 PASS |
| ZIP integrity | PASS |
| Independent B1 split validation | PENDING |
| Final class mapping | PROVISIONAL |
| Training approval | False |
| Evaluation approval | False |

## Package Information

- Filename: `RDD_synth_v0.4_review_only.zip`
- Files: 61
- Size: 22.31 MiB
- SHA-256: `f5cd59820c45440c618655756d5bd09d4d7d52c8170d36303657f4fe0656ff3f`

The package is intended for authorized project
team members only and is not distributed through
this public repository.

## Synthesis Method

The pilot uses object-centric synthesis:

1. Object extraction using SAM-derived masks
2. Road ROI-based placement
3. Perspective-aware object sizing and positioning
4. Appearance harmonization
5. YOLO annotation generation
6. Source provenance and integrity validation

## Remaining Validation

- Obtain the official B1 split lists
- Validate foreground and background source membership
- Check potential sequence-level data leakage
- Confirm the detector class mapping
- Complete visual quality review
- Record training approval separately

## Limitations

This is a small development pilot containing
27 synthetic images from 9 backgrounds and
13 distinct foreground objects.

No improvement in recall or mAP is claimed.

## Data and Licensing

AI Hub-derived images, source filenames,
private storage paths, masks and restricted
dataset metadata are excluded from this repository.

Dataset use remains subject to source terms.
