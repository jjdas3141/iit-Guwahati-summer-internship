
%% 02_build_data_matrix.m
% Build the face data matrix from mixed image formats.
% Supported: PGM, JPG, JPEG, PNG, BMP, TIF, TIFF

clear;
clc;
close all;

%% Dataset path
dataPath = 'C:\Users\JEEBANJYOTI\OneDrive\Desktop\Summer Internship IIT Guwahati\DATA set';

imageSize = [112, 92];
numSubjects = 40;

supportedExtensions = { ...
    '*.pgm', '*.jpg', '*.jpeg', '*.png', ...
    '*.bmp', '*.tif', '*.tiff'};

%% Read all image paths
imagePaths = {};
labels = [];

for subject = 1:numSubjects

    subjectFolder = fullfile(dataPath, sprintf('s%d', subject));

    if ~isfolder(subjectFolder)
        error('Subject folder not found: %s', subjectFolder);
    end

    subjectFiles = [];

    for extIndex = 1:numel(supportedExtensions)
        files = dir(fullfile(subjectFolder, ...
            supportedExtensions{extIndex}));

        subjectFiles = [subjectFiles; files]; %#ok<AGROW>
    end

    % Sort filenames for repeatable image ordering.
    if isempty(subjectFiles)
        error('No supported images found in %s', subjectFolder);
    end

    fileNames = {subjectFiles.name};
    [~, order] = sort(lower(string(fileNames)));
    subjectFiles = subjectFiles(order);

    for j = 1:numel(subjectFiles)
        imagePaths{end+1} = fullfile(subjectFolder, ...
            subjectFiles(j).name); %#ok<SAGROW>
        labels(end+1) = subject; %#ok<SAGROW>
    end
end

%% Load and preprocess images
numImages = numel(imagePaths);
numPixels = prod(imageSize);

P = zeros(numPixels, numImages);

fprintf('Loading %d images...\n', numImages);

for i = 1:numImages

    imagePath = imagePaths{i};

    try
        I = imread(imagePath);
    catch ME
        error('Cannot read image "%s": %s', imagePath, ME.message);
    end

    % Convert RGB or indexed images to grayscale.
    if ndims(I) == 3
        I = rgb2gray(I);
    end

    % Resize all images to the same dimensions.
    I = imresize(I, imageSize);

    % Convert to double and store as a column vector.
    P(:, i) = double(I(:));

    if mod(i, 50) == 0 || i == numImages
        fprintf('Processed %d / %d images\n', i, numImages);
    end
end

%% Compute mean face and centered data
meanFace = mean(P, 2);
A = P - meanFace;

%% Save dataset
save('face_data.mat', ...
    'P', 'labels', 'meanFace', 'A', ...
    'dataPath', 'imagePaths', 'imageSize', '-v7.3');

fprintf('\nDataset created successfully.\n');
fprintf('Number of images: %d\n', numImages);
fprintf('Image vector dimension: %d x 1\n', numPixels);
fprintf('Data matrix dimension: %d x %d\n', size(P, 1), size(P, 2));
fprintf('Saved as face_data.mat\n');
