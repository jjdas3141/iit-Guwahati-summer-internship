# Eigenfaces Using Singular Value Decomposition (SVD)

A MATLAB implementation of eigenfaces for face representation, reconstruction, and recognition using Singular Value Decomposition (SVD).

This project presents a small-scale computational study of methods explored during my summer internship at IIT Guwahati.

## Objectives

- Construct a face data matrix from grayscale images.
- Compute the mean face and center the data.
- Apply SVD to obtain eigenfaces.
- Study dimensionality reduction using cumulative singular-value energy.
- Reconstruct faces using different numbers of eigenfaces.
- Evaluate face recognition using nearest-neighbour classification.
- Explore rejection criteria based on distance and reconstruction error.

## Mathematical Background

Let the vectorized face images form the data matrix

`P = [x_1, x_2, ..., x_n]`, where `P` is a `d × n` matrix.

The **mean face** is computed as

`mu = (1/n) * sum(x_i)` for `i = 1, ..., n`.

The **centered data matrix** is

`A = P - mu * 1^T`

where `1` is a vector of ones.

We then compute the **Singular Value Decomposition (SVD)**:

`A = U * Sigma * V^T`

Here:

- `U` contains the left singular vectors, which form the eigenface basis.
- `Sigma` contains the singular values, ordered from largest to smallest.
- `V` contains the right singular vectors.

The leading columns of `U` form a lower-dimensional eigenface subspace. A face can be projected onto this subspace and reconstructed using a selected number of components.

## Project Structure

| Script | Description |
|---|---|
| `O1_check_dataset.m` | Validate the dataset and image files |
| `O2_build_data_matrix.m` | Construct the data matrix and mean face |
| `O3_svd_eigenfaces.m` | Compute SVD and visualize eigenfaces |
| `O4_select_components.m` | Analyze cumulative singular-value energy |
| `O5_face_reconstruction.m` | Reconstruct faces using selected components |
| `O6_face_recognition.m` | Evaluate nearest-neighbour recognition |
| `O7_interactive_eigenface_demo.m` | Interactive recognition and rejection demo |

## Dataset

The project uses the AT&T/ORL Face Database, consisting of 400 grayscale images from 40 subjects, with 10 images per subject. Each image is 112 × 92 pixels.

The dataset is not included in this repository. Obtain it from an authorized source and arrange the images in folders named `s1` through `s40`.

Update the dataset path in the MATLAB scripts before running them.

## Recognition and Rejection

The recognition stage compares image coefficient vectors in eigenface space using Euclidean distance.

The interactive demonstration uses two rejection criteria:

1. **Nearest-neighbour distance:** checks whether the query is sufficiently close to a training image.
2. **Relative reconstruction error:** measures how well the centered query can be represented by the selected eigenface subspace.

A query is accepted only when both criteria pass. These empirical thresholds do not guarantee correct identity recognition.

## How to Run

1. Install MATLAB with the Image Processing Toolbox.
2. Download and organize the ORL dataset.
3. Update the dataset path in the scripts.
4. Run the scripts in numerical order, as applicable.
5. Inspect the reconstruction plots, recognition accuracy, and interactive demo.

## Limitations

- Results depend on image alignment, pose, lighting, and the number of eigenfaces.
- Nearest-neighbour classification always finds a closest candidate, even for an unfamiliar face.
- Rejection thresholds require calibration and independent evaluation.

## Reference

*Singular Value Decomposition, Eigenfaces, and 3D Reconstructions.*

This repository is an educational implementation for studying SVD, dimensionality reduction, image reconstruction, and face recognition.
