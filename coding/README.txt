IWCH: blockwise SVD + sparse residual compressive hologram
==========================================================

This package handles mixed matrix dimensions, for example:
  layer_00.csv = 768x768
  layer_01.csv = 768x768
  layer_02.csv = 3072x768
  layer_03.csv = 768x3072

The encoder creates a fixed-size 2D complex hologram representation:
  output.png        phase preview
  output_real.png   real part of encoded 2D complex field
  output_imag.png   imaginary part of encoded 2D complex field
  output_meta.png   compressed metadata image

Install
-------
  pip install numpy pillow

Encode
------
Basic mixed-shape run:
  python encode_iwch.py ./matrix_layers output.png --rank 8 --residual_k 32 --block_rows 128 --block_cols 128

Target about 9x storage reduction for 12 layers of 768x768:
  python encode_iwch.py ./matrix_layers output.png --rank 8 --residual_k 32 --block_rows 128 --block_cols 128 --holo_height 887 --holo_width 887

Higher accuracy, less compression:
  python encode_iwch.py ./matrix_layers output.png --rank 16 --residual_k 64 --block_rows 128 --block_cols 128 --holo_height 1536 --holo_width 1536

Decode
------
  python decode_iwch.py output.png decoded_layers --vtp reconstructed.vtp

Compare
-------
  python compare_reconstruction.py ./matrix_layers decoded_layers

Parameter meaning
-----------------
--rank:
  SVD rank kept per block. Higher rank preserves more structure but uses more space.

--residual_k:
  Number of Fourier residual coefficients kept per block. Higher values preserve more details.

--block_rows / --block_cols:
  Block size. 128x128 is a good start for 768-scale matrices.
  Smaller blocks can improve local reconstruction but increase metadata and overhead.

--holo_height / --holo_width:
  Fixed 2D hologram size. If omitted, the encoder chooses the minimum size required to store the selected coefficients.

Notes
-----
- This is approximate reconstruction. 80%+ depends on matrix structure and parameter budget.
- For true 9x compression, the real+imag 16-bit images should total around original_float32_bytes/9.
- The metadata image stores shapes, block positions, ranks, and residual index positions. It does not store the original full matrices.
