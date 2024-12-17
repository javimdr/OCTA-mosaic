# OCTA-mosaic

`octa-mosaic` is a Python library for building, optimizing, and blending mosaics using OCTA (Optical Coherence Tomography Angiography) images. It provides a fully automatic method for generating high-resolution, wide-field OCTA mosaics from overlapping scans, addressing the need for wider fields of view without requiring advanced OCTA equipment or manual mosaicking.

The proposed approach consists of a three-stage pipeline:
1. **Build an Initial Mosaic**: Constructs an initial mosaic using correlation-based template matching.
1. **Optimize Mosaic**: Refines the mosaic with an evolutionary algorithm to optimize vascular continuity at seams.
1. **Blend Seamlines**: Finalizes the mosaic with blending techniques to improve overall quality.

Unlike existing methods, this approach avoids keypoint extraction or input image preprocessing, making it robust against noise and artifacts typically present in clinical OCTA images. It employs a correlation-based metric to measure the degree of vascular continuity at the seams in each mosaic.

## Features

- **Mosaic Construction:** Build an initial mosaic using a template-matching approach.
- **Optimization**: Optimize image alignments using different optimization algorithms.
- **Seamless Blending:** Apply alpha blending to create smooth transitions between images.
- **Load OCTA Dataset**: Easily download and use the [OCTA-Mosaicking Dataset](https://doi.org/10.5281/zenodo.14333858).

## Getting started

- [Create mosaic example](./examples/create_mosaics.ipynb)
