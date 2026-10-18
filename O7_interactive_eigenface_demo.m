
%% O7_interactive_eigenface_demo.m
% Interactive Eigenfaces Recognition with Unknown-Face Rejection
%
% Four-panel display:
% 1. Selected Image
% 2. Grayscale and Resized
% 3. Eigenface Reconstruction
% 4. Accepted Match or NO CLOSE MATCH FOUND
%
% Model: eigenface_model.mat
% Training: images 1-5 per subject
% Calibration: images 6-7 per subject
% Final test: images 8-10 per subject
%
% Acceptance requires BOTH:
% 1. Nearest-neighbour distance below threshold
% 2. Relative reconstruction error below threshold

clear;
clc;
close all;

%% 1. Load the trained model

modelFile = 'eigenface_model.mat';

if ~isfile(modelFile)
    error('Model file not found. Run O6_face_recognition.m first.');
end

load(modelFile, ...
    'meanFace', 'U', 'Xtrain', 'trainLabels', ...
    'trainImagePaths', 'Xcal', 'calLabels', ...
    'calibrationImagePaths', 'imageSize', 'bestR');

fprintf('Eigenfaces model loaded successfully.\n');

%% 2. Select the number of eigenfaces

r = min(bestR, size(U, 2));
Ur = U(:, 1:r);

fprintf('Number of eigenfaces used: %d\n', r);

%% 3. Calculate training coefficients

Atrain = Xtrain - meanFace;
trainingCoeff = Ur' * Atrain;

%% 4. Calibrate rejection thresholds using images 6-7

fprintf('\nCalibrating rejection thresholds...\n');

numCalibration = size(Xcal, 2);

if numCalibration == 0
    error('No calibration images found in eigenface_model.mat.');
end

knownDistances = zeros(numCalibration, 1);
knownErrors = zeros(numCalibration, 1);

for k = 1:numCalibration

    % Calibration images are already loaded by O6.
    xKnown = Xcal(:, k);
    vKnown = xKnown - meanFace;

    % Project into the eigenface space.
    yKnown = Ur' * vKnown;

    % Reconstruct the calibration image.
    vRecon = Ur * yKnown;

    % Relative reconstruction error.
    knownErrors(k) = ...
        norm(vKnown - vRecon) / max(norm(vKnown), eps);

    % Distance to the nearest training image.
    distances = sqrt(sum( ...
        (trainingCoeff - yKnown).^2, 1));

    knownDistances(k) = min(distances);

end

% Set thresholds using the 95th percentile of known-face scores.
distanceThreshold = prctile(knownDistances, 95);
errorThreshold = prctile(knownErrors, 95);

fprintf('Distance threshold: %.4f\n', distanceThreshold);
fprintf('Reconstruction error threshold: %.4f\n', errorThreshold);

%% 5. Choose a query image

[fileName, folderName] = uigetfile( ...
    {'*.pgm;*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', ...
     'Supported image files'}, ...
    'Select a face image');

if isequal(fileName, 0)
    fprintf('\nDemo cancelled: no image selected.\n');
    return;
end

queryPath = fullfile(folderName, fileName);

fprintf('\nSelected image: %s\n', queryPath);

%% 6. Read and preprocess the query image

I = imread(queryPath);

% Convert the original image to grayscale if necessary.
if ndims(I) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% Resize to the dimensions used by the ORL dataset.
Iresized = imresize(Igray, imageSize);

% Convert the image into a column vector.
x = double(Iresized(:));

% Centre using the training mean face.
v = x - meanFace;

%% 7. Project and reconstruct the query

queryCoeff = Ur' * v;

vReconstructed = Ur * queryCoeff;

% Restore the mean face to the reconstruction.
reconstructedFace = reshape( ...
    vReconstructed + meanFace, imageSize);

% Relative reconstruction error.
reconstructionError = ...
    norm(v - vReconstructed) / max(norm(v), eps);

%% 8. Find the nearest training image

distances = sqrt(sum( ...
    (trainingCoeff - queryCoeff).^2, 1));

[nearestDistance, nearestIndex] = min(distances);

predictedSubject = trainLabels(nearestIndex);

% Use the saved path of the nearest training image.
matchedPath = trainImagePaths{nearestIndex};

%% 9. Apply both rejection criteria

distanceAccepted = ...
    nearestDistance <= distanceThreshold;

reconstructionAccepted = ...
    reconstructionError <= errorThreshold;

isAccepted = distanceAccepted && reconstructionAccepted;

if isAccepted
    decisionText = sprintf( ...
        'ACCEPTED: Subject %d', predictedSubject);
else
    decisionText = 'REJECTED: No close match found';
end

%% 10. Display the four panels

figure( ...
    'Name', 'Interactive Eigenfaces Recognition', ...
    'NumberTitle', 'off', ...
    'Color', 'w', ...
    'Position', [100, 100, 1100, 760]);

% Panel 1: Original selected image.
subplot(2, 2, 1);
imshow(I);
title('Selected Image');

% Panel 2: Grayscale and resized image.
subplot(2, 2, 2);
imshow(Iresized, []);
title('Grayscale and Resized');

% Panel 3: Eigenface reconstruction.
subplot(2, 2, 3);
imshow(reconstructedFace, []);
title(sprintf('Eigenface Reconstruction (r = %d)', r));

% Panel 4: Show the match only if accepted.
subplot(2, 2, 4);

if isAccepted

    if isfile(matchedPath)

        matchedImage = imread(matchedPath);

        if ndims(matchedImage) == 3
            matchedImage = rgb2gray(matchedImage);
        end

        imshow(matchedImage, []);

        title(sprintf('Accepted Match: Subject %d', ...
            predictedSubject));

    else

        axis off;

        text(0.5, 0.5, 'Matched image file not found', ...
            'Units', 'normalized', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 12);

        title('Recognition Result');

    end

else

    axis off;

    text(0.5, 0.58, 'NO CLOSE MATCH FOUND', ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 14, ...
        'FontWeight', 'bold', ...
        'Color', [0.75, 0, 0]);

    text(0.5, 0.42, 'No identity assigned', ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 11);

    title('Recognition Result');

end

%% 11. Show decision and measurements beneath the four panels

if isAccepted
    decisionColor = [0, 0.45, 0];
else
    decisionColor = [0.75, 0, 0];
end

resultText = sprintf([ ...
    '%s\n' ...
    'Nearest distance: %.4f (threshold: %.4f) | ' ...
    'Relative reconstruction error: %.4f (threshold: %.4f)'], ...
    decisionText, ...
    nearestDistance, distanceThreshold, ...
    reconstructionError, errorThreshold);

annotation('textbox', [0.04, 0.015, 0.92, 0.09], ...
    'String', resultText, ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 10, ...
    'Color', decisionColor, ...
    'Interpreter', 'none');

%% 12. Print the full result in the Command Window

fprintf('\n========================================\n');
fprintf('      EIGENFACES RECOGNITION RESULT\n');
fprintf('========================================\n');

fprintf('Query image: %s\n', queryPath);
fprintf('Eigenfaces used: %d\n', r);

fprintf('\nNearest training candidate:\n');
fprintf('Subject: %d\n', predictedSubject);
fprintf('Image path: %s\n', matchedPath);

fprintf('\nDistance test:\n');
fprintf('Measured distance: %.4f\n', nearestDistance);
fprintf('Threshold: %.4f\n', distanceThreshold);
fprintf('Passed: %s\n', string(distanceAccepted));

fprintf('\nReconstruction test:\n');
fprintf('Relative error: %.4f\n', reconstructionError);
fprintf('Threshold: %.4f\n', errorThreshold);
fprintf('Passed: %s\n', string(reconstructionAccepted));

fprintf('\nFINAL DECISION: %s\n', decisionText);
fprintf('========================================\n');
