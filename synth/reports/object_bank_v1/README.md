# RDD Training Object Pool v1

Pool version: training_pool_v1_20261127_180657_5e79d3ab

## Final selection

| Category | Objects |
|---|---:|
| Full Object Bank | 692 |
| Pilot candidates | 618 |
| RGBA review PASS | 490 |
| Source review PASS | 12 |
| Approved training pool | 502 |
| Source review HOLD | 93 |
| Source review EXCLUDE | 23 |
| Not in pilot | 74 |

## Approved class distribution

| Class | Objects |
|---|---:|
| Box | 112 |
| Cone | 112 |
| Garbage | 109 |
| Sign | 107 |
| Stone | 62 |

Unique foreground source images: 495.

## Quality and leakage validation

- Physical foreground file validation: 502/502 PASS
- Invalid foreground files: 0
- Exact file-content duplicates: 0
- Exact decoded RGBA duplicates: 0
- B1 training split membership: PASS
- Sequence-level audit: PASS
- Validation/test overlap: 0

## Label policy

Original subclasses are preserved for synthesis analysis.
All detector training labels use class ID 0 (debris).

## Usage

This object pool is approved for controlled synthesis
experiments. It does not establish synthetic image
realism or downstream detection performance.

## Data management

Foreground images and object-level manifests remain
in the private team storage.

Only aggregate reports and public-safe documentation
should be uploaded to GitHub.

Original manifests and review records are preserved.
