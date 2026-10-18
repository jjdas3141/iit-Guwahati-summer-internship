
clc;
clear;
close all;

%% O6: Train, calibrate, and independently evaluate Eigenfaces

dataPath = 'C:\Users\JEEBANJYOTI\OneDrive\Desktop\Summer Internship IIT Guwahati\DATA set';

numSubjects = 40;
imagesPerSubject = 10;
numTrainPerSubject = 5;
imageSize = [112, 92];

% Images 1-5: training
% Images 6-7: threshold calibration
% Images 8-10: independent final evaluation

Xtrain = [];
Xcal = [];
Xtest = [];

trainLabels = [];
calLabels = [];
testLabels = [];

trainImagePaths = {};
calibrationImagePaths = {};
testImagePaths = {};

%% Step 1: Load the ORL dataset using the fixed split

for subject = 1:numSubjects

    subjectFolder = fullfile(dataPath, sprintf('s%d', subject));

    if ~isfolder(subjectFolder)
        error('Subject folder not found: %s', subjectFolder);
    end

    for imageNum = 1:imagesPerSubject

        imagePath = fullfile(subjectFolder, sprintf('%d.pgm', imageNum));

        if ~isfile(imagePath)
            error('Expected image not found: %s', imagePath);
        end

        I = imread(imagePath);

        if ndims(I) == 3
            I = rgb2gray(I);
        end

        if ~isequal(size(I), imageSize)
            error('Unexpected image dimensions: %s', imagePath);
        end

        imageVector = double(I(:));

        if imageNum <= 5

            Xtrain = [Xtrain, imageVector];
            trainLabels = [trainLabels, subject];
            trainImagePaths{end+1} = imagePath;

        elseif imageNum <= 7

            Xcal = [Xcal, imageVector];
            calLabels = [calLabels, subject];
            calibrationImagePaths{end+1} = imagePath;

        else

            Xtest = [Xtest, imageVector];
            testLabels = [testLabels, subject];
            testImagePaths{end+1} = imagePath;

        end
    end
end

fprintf('Training images: %d\n', size(Xtrain, 2));
fprintf('Calibration images: %d\n', size(Xcal, 2));
fprintf('Final test images: %d\n\n', size(Xtest, 2));

assert(size(Xtrain, 2) == 200, 'Expected 200 training images.');
assert(size(Xcal, 2) == 80, 'Expected 80 calibration images.');
assert(size(Xtest, 2) == 120, 'Expected 120 final test images.');

%% Step 2: Compute the mean face using training images only

meanFace = mean(Xtrain, 2);

Atrain = Xtrain - meanFace;

%% Step 3: Compute the eigenface basis using training images only

[U, S, ~] = svd(Atrain, 'econ');

singularValues = diag(S);

fprintf('Eigenface basis computed from training images only.\n');
fprintf('Number of available components: %d\n\n', size(U, 2));

%% Step 4: Select the number of components using calibration images

rValues = [20, 50, 100, 190];

rValues = rValues(rValues <= size(U, 2));

calibrationAccuracy = zeros(size(rValues));
finalAccuracy = zeros(size(rValues));
finalCorrectCounts = zeros(size(rValues));

for k = 1:numel(rValues)

    r = rValues(k);
    Ur = U(:, 1:r);

    % Project training images into the r-dimensional space
    trainCoefficients = Ur' * Atrain;

    % Evaluate candidate r using calibration images
    correctCal = 0;

    for j = 1:size(Xcal, 2)

        query = Xcal(:, j) - meanFace;
        queryCoefficients = Ur' * query;

        differences = trainCoefficients - queryCoefficients;
        distances = sqrt(sum(differences.^2, 1));

        [~, nearestIndex] = min(distances);

        predictedLabel = trainLabels(nearestIndex);

        if predictedLabel == calLabels(j)
            correctCal = correctCal + 1;
        end
    end

    calibrationAccuracy(k) = ...
        100 * correctCal / size(Xcal, 2);

    % Independently evaluate the final test images
    correctTest = 0;

    for j = 1:size(Xtest, 2)

        query = Xtest(:, j) - meanFace;
        queryCoefficients = Ur' * query;

        differences = trainCoefficients - queryCoefficients;
        distances = sqrt(sum(differences.^2, 1));

        [~, nearestIndex] = min(distances);

        predictedLabel = trainLabels(nearestIndex);

        if predictedLabel == testLabels(j)
            correctTest = correctTest + 1;
        end
    end

    finalCorrectCounts(k) = correctTest;

    finalAccuracy(k) = ...
        100 * correctTest / size(Xtest, 2);

end

%% Step 5: Choose r using calibration accuracy, not final test accuracy

[~, bestIndex] = max(calibrationAccuracy);
bestR = rValues(bestIndex);

fprintf('Results for each number of components:\n\n');

resultsTable = table( ...
    rValues(:), ...
    calibrationAccuracy(:), ...
    finalAccuracy(:), ...
    finalCorrectCounts(:), ...
    'VariableNames', ...
    {'Components', 'CalibrationAccuracyPercent', ...
     'FinalTestAccuracyPercent', 'CorrectFinalTests'});

disp(resultsTable);

fprintf('Selected number of components: %d\n', bestR);
fprintf('Final independent test accuracy at selected r: %.2f%%\n', ...
    finalAccuracy(bestIndex));

%% Step 6: Save the model and data splits

save('eigenface_model.mat', ...
    'meanFace', 'U', 'S', 'singularValues', ...
    'Xtrain', 'trainLabels', 'trainImagePaths', ...
    'Xcal', 'calLabels', 'calibrationImagePaths', ...
    'Xtest', 'testLabels', 'testImagePaths', ...
    'rValues', 'calibrationAccuracy', ...
    'finalAccuracy', 'finalCorrectCounts', ...
    'bestR', 'dataPath', 'imageSize');

writetable(resultsTable, 'recognition_results.csv');

fprintf('\nSaved eigenface_model.mat\n');
fprintf('Saved recognition_results.csv\n');
