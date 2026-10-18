
%% 01_check_dataset.m
% Check dataset folders and validate image files.
% Supports PGM, JPG, JPEG, PNG, BMP, TIF, TIFF.

clear;
clc;
close all;

%% 1. Dataset path

dataPath = 'C:\Users\JEEBANJYOTI\OneDrive\Desktop\Summer Internship IIT Guwahati\DATA set';

numSubjects = 40;
expectedImagesPerSubject = 10;
imageSize = [112, 92];

supportedExtensions = { ...
    '.pgm', '.jpg', '.jpeg', '.png', ...
    '.bmp', '.tif', '.tiff'};

%% 2. Check dataset directory

if ~isfolder(dataPath)
    error('Dataset folder not found:\n%s', dataPath);
end

fprintf('========================================\n');
fprintf('       EIGENFACES DATASET CHECK\n');
fprintf('========================================\n');
fprintf('Dataset path: %s\n\n', dataPath);

%% 3. Check each subject folder

totalImages = 0;
totalUnreadable = 0;
totalInvalidSize = 0;
allValid = true;

for subject = 1:numSubjects

    subjectFolder = fullfile(dataPath, sprintf('s%d', subject));

    if ~isfolder(subjectFolder)
        fprintf('ERROR: Missing folder s%d\n', subject);
        allValid = false;
        continue;
    end

    entries = dir(subjectFolder);
    entries = entries(~[entries.isdir]);

    imageFiles = [];

    % Filter by supported file extensions.
    for j = 1:numel(entries)

        [~, ~, ext] = fileparts(entries(j).name);

        if any(strcmpi(ext, supportedExtensions))
            imageFiles = [imageFiles; entries(j)]; %#ok<AGROW>
        end
    end

    numImages = numel(imageFiles);
    totalImages = totalImages + numImages;

    if numImages ~= expectedImagesPerSubject
        fprintf('WARNING: s%d contains %d images; expected %d.\n', ...
            subject, numImages, expectedImagesPerSubject);
        allValid = false;
    else
        fprintf('s%d: %d images found\n', subject, numImages);
    end

    % Check whether each image can be read.
    for j = 1:numImages

        imagePath = fullfile(subjectFolder, imageFiles(j).name);

        try
            I = imread(imagePath);

            if isempty(I)
                error('Image is empty.');
            end

            % Confirm the image can be converted and resized.
            if ndims(I) == 3
                I = rgb2gray(I);
            end

            I = imresize(I, imageSize);

            if ~isequal(size(I), imageSize)
                totalInvalidSize = totalInvalidSize + 1;
                fprintf('SIZE ERROR: %s\n', imagePath);
                allValid = false;
            end

        catch ME
            totalUnreadable = totalUnreadable + 1;
            allValid = false;

            fprintf('UNREADABLE: %s\n', imagePath);
            fprintf('  Reason: %s\n', ME.message);
        end
    end
end

%% 4. Summary

fprintf('\n========================================\n');
fprintf('              SUMMARY\n');
fprintf('========================================\n');

fprintf('Expected subjects: %d\n', numSubjects);
fprintf('Expected total images: %d\n', ...
    numSubjects * expectedImagesPerSubject);
fprintf('Total supported images found: %d\n', totalImages);
fprintf('Unreadable images: %d\n', totalUnreadable);
fprintf('Invalid resized dimensions: %d\n', totalInvalidSize);

if allValid && totalImages == ...
        numSubjects * expectedImagesPerSubject

    fprintf('\nDATASET CHECK PASSED.\n');
    fprintf('All subjects contain 10 readable images.\n');
    fprintf('You can run 02_build_data_matrix.m next.\n');

else

    fprintf('\nDATASET CHECK NEEDS ATTENTION.\n');
    fprintf('Review the warnings above before continuing.\n');

end

fprintf('========================================\n');
