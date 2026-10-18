
%% 05_face_reconstruction.m
% Reconstruct a face using different numbers of eigenfaces.

clear;
clc;
close all;

%% 1. Load dataset and SVD results

if ~isfile('face_data.mat')
    error('face_data.mat not found. Run 02_build_data_matrix.m first.');
end

if ~isfile('eigenfaces_svd.mat')
    error('eigenfaces_svd.mat not found. Run 03_svd_eigenfaces.m first.');
end

D = load('face_data.mat', 'P', 'meanFace');
E = load('eigenfaces_svd.mat', 'U');

P = D.P;
meanFace = D.meanFace;
U = E.U;

%% 2. Select an example image

imageIndex = 1;

if imageIndex > size(P, 2)
    error('Selected image index exceeds the dataset size.');
end

x = P(:, imageIndex);

% Centre the image using the mean face.
v = x - meanFace;

%% 3. Choose reconstruction dimensions

rValues = [20, 50, 100, 190];

% Keep only values supported by the available SVD.
rValues = rValues(rValues <= size(U, 2));

%% 4. Display original and reconstructed faces

figure('Name', 'Eigenface Reconstruction', ...
    'Color', 'w');

tiledlayout(1, numel(rValues) + 1);

nexttile;
imshow(reshape(x, 112, 92), []);
title('Original');

for k = 1:numel(rValues)

    r = rValues(k);
    Ur = U(:, 1:r);

    % Project into the eigenface subspace.
    coefficients = Ur' * v;

    % Reconstruct and add the mean face back.
    xReconstructed = meanFace + Ur * coefficients;

    nexttile;
    imshow(reshape(xReconstructed, 112, 92), []);
    title(sprintf('r = %d', r));

    relativeError = norm(x - xReconstructed) / ...
        max(norm(v), eps);

    fprintf('r = %d, relative reconstruction error = %.4f\n', ...
        r, relativeError);
end

sgtitle('Face Reconstruction Using Eigenfaces');

fprintf('\nReconstruction completed successfully.\n');
