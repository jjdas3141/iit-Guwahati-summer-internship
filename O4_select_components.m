
clc;
clear;
close all;

load('eigenfaces_svd.mat', 'singularValues');

% Variance represented by each singular value
energy = singularValues.^2;

% Cumulative proportion of total squared singular values
cumulativeEnergy = cumsum(energy) / sum(energy);

% Plot cumulative energy
figure;
plot(1:length(cumulativeEnergy), cumulativeEnergy, 'LineWidth', 1.5);
hold on;
yline(0.90, '--r', '90%');
yline(0.95, '--g', '95%');
yline(0.98, '--k', '98%');
xlabel('Number of components');
ylabel('Cumulative proportion of squared singular values');
title('Cumulative Energy of the SVD');
ylim([0 1.02]);
grid on;

% Find component counts for selected thresholds
thresholds = [0.90, 0.95, 0.98];

for t = thresholds
    r = find(cumulativeEnergy >= t, 1, 'first');
    fprintf('%2.0f%% threshold: %d components\n', 100*t, r);
end
