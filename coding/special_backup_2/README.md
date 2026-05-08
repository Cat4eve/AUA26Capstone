# Matrix Group Holography Pipeline

This package contains a complete testable pipeline:

```text
matrix folder
  -> grouped VTP
  -> hologram PNG files
  -> decoded CSV matrix layers
  -> decoded grouped VTP
```

Main script:

```text
matrix_group_holography_pipeline.py
```

## Main full roundtrip command

```bash
python matrix_group_holography_pipeline.py roundtrip test_matrices roundtrip_output --holo_height 256 --holo_width 256 --block_gap 2
```

## Separate commands

### 1. Matrices to grouped VTP

```bash
python matrix_group_holography_pipeline.py matrices-to-vtp test_matrices original_grouped.vtp --block_gap 2
```

### 2. Grouped VTP to hologram

```bash
python matrix_group_holography_pipeline.py encode original_grouped.vtp encoded_hologram.png --holo_height 256 --holo_width 256
```

### 3. Decode hologram

```bash
python matrix_group_holography_pipeline.py decode encoded_hologram.png decoded_output --block_gap 2
```

## VTP geometry rule

```text
x = col_index * x_spacing + layer_x_offset
y = row_index * y_spacing + layer_y_offset
z = matrix[row, col]
```

So each matrix becomes a height map, and each matrix block is placed near the others in the x-y plane.

## Files created by roundtrip

```text
original_grouped.vtp
encoded_hologram.png
encoded_hologram_real.png
encoded_hologram_imag.png
encoded_hologram_meta.png
decoded_grouped.vtp
decoded_csv/
reconstruction_metrics.csv
```

## Important note about quality

When the hologram tile for a layer is at least as large as the original matrix,
the reconstruction should be nearly exact except for PNG quantization noise.
If the matrix is larger than its allocated hologram tile, the script keeps only
low-frequency FFT content, so reconstruction becomes lossy.

## Test output

```text
Loaded layer_0_small.csv: shape=(3, 4), min=0, max=1
Loaded layer_1_tall.csv: shape=(5, 3), min=0, max=4
Loaded layer_2_bell.csv: shape=(7, 7), min=0.011109, max=1
========================================================================
STEP 1: matrices -> grouped VTP
========================================================================
========================================================================
STEP 2: grouped VTP -> hologram
========================================================================
Encoded grouped VTP into hologram
Input VTP:      /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/original_grouped.vtp
Preview image:  /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram.png
Real image:     /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_real.png
Imag image:     /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_imag.png
Metadata image: /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_meta.png
Layers:         3
Hologram size:  256 x 256
========================================================================
STEP 3: hologram -> decoded matrices + grouped VTP
========================================================================
Decoded hologram into matrices and grouped VTP
Decoded CSV folder: /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/decoded_csv
Decoded VTP:        /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/decoded_grouped.vtp
========================================================================
DONE: full roundtrip
========================================================================
Original grouped VTP: /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/original_grouped.vtp
Hologram preview:     /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram.png
Hologram real:        /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_real.png
Hologram imag:        /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_imag.png
Hologram metadata:    /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/encoded_hologram_meta.png
Decoded VTP:          /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/decoded_grouped.vtp
Decoded CSV folder:   /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/decoded_csv
Metrics CSV:          /mnt/data/matrix_group_holography_pipeline_v2/roundtrip_test_output/reconstruction_metrics.csv

Reconstruction metrics:
  layer 0 shape=(3, 4) MAE=5.13425e-07 RMSE=6.16864e-07 max_abs=1.10246e-06 rel_RMSE=1.35692e-06
  layer 1 shape=(5, 3) MAE=1.56031e-06 RMSE=2.0151e-06 max_abs=4.92845e-06 rel_RMSE=1.05235e-06
  layer 2 shape=(7, 7) MAE=1.67138e-07 RMSE=2.42294e-07 max_abs=7.44901e-07 rel_RMSE=6.76811e-07

```

## Test errors/warnings

```text
Spreadsheet runtime warmup failed during python startup
Traceback (most recent call last):
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/patches/warm_spreadsheet_runtime_on_startup.py", line 26, in warm_spreadsheet_runtime_on_startup
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 785, in warm_spreadsheet_runtime
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 720, in _warm_feature_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/spreadsheet_warmup.py", line 704, in _warm_collaboration_flows
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/generated/interface/models.py", line 48821, in hydrate_crdt_from_proto
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/remote.py", line 747, in __call__
  File "/tmp/tmp.9eeVjt35CN/artifact_tool_v2-2.7.5/artifact_tool/rpc/client.py", line 150, in call
artifact_tool.rpc.client.RemoteError: hydrateCrdtFromProto requires an empty collaborative document.
/mnt/data/matrix_group_holography_pipeline_v2/matrix_group_holography_pipeline.py:433: DeprecationWarning: 'mode' parameter for changing data types is deprecated and will be removed in Pillow 13 (2026-10-15)
  Image.fromarray(np.asarray(arr, dtype=np.uint16), mode="I;16").save(path)

```
