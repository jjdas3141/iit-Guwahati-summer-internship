
%% 03_svd_eigenfaces.m
% Compute the SVD of the centered face data matrix.

clear;
clc;
close all;

%% Load the data built by 02_build_data_matrix.m

dataFile = 'face_data.mat';

if ~isfile(dataFile)
    error('face_data.mat not found. Run 02_build_data_matrix.m first.');
end

D = load(dataFile);

if ~isfield(D, 'A')
    error(['The file face_data.mat does not contain A. ' ...
        'Run 02_build_data_matrix.m again.']);
end

A = D.A;

fprintf('Loaded centered data matrix A: %d x %d\n', ...
    size(A, 1), size(A, 2));

%% Compute economy SVD

fprintf('Computing SVD...\n');

[U, S, V] = svd(A, 'econ');

singularValues = diag(S);

fprintf('SVD completed.\n');
fprintf('U: %d x %d\n', size(U, 1), size(U, 2));
fprintf('S: %d x %d\n', size(S, 1), size(S, 2));
fprintf('V: %d x %d\n', size(V, 1), size(V, 2));

%% Display the first 10 eigenfaces

numToShow = min(10, size(U, 2));

figure('Name', 'First 10 Eigenfaces', 'Color', 'w');

for j = 1:numToShow
    subplot(2, 5, j);

    eigenface = reshape(U(:, j), 112, 92);

    imshow(eigenface, []);
    title(sprintf('Eigenface %d', j));
end

%% Save SVD results

save('eigenfaces_svd.mat', ...
    'U', 'S', 'V', 'singularValues', '-v7.3');

fprintf('Saved SVD results to eigenfaces_svd.mat\n');
