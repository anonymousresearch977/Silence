%% COMPLETE LSTM VOWEL A CLASSIFIER 
clc; clear; close all;
rng(42, 'twister');
fprintf('Random seed set to 42\n\n');
%% CONFIGURATION
FREQ_MIN = 0.1;
FREQ_MAX = 450;
SAMPLE_RATE = 12500;
%% LOAD DATA
fprintf(' Loading Data \n');
tic;

trainData = load('trainv2.mat');
FinalTableTrain = trainData.FinalTable;
clear trainData

testData = load('testv2.mat');
FinalTableTest = testData.FinalTable;
clear testData

fprintf(' Loaded in %.1f sec\n', toc);
fprintf('  Train: %d rows\n', height(FinalTableTrain));
fprintf('  Test: %d rows\n\n', height(FinalTableTest));
%% PROCESS LABELS
fprintf(' Processing Labels \n');

% Normalize training labels
v = upper(strtrim(FinalTableTrain.Vowels));
v(v == "" | v == "N" | v == "NONE") = "";
FinalTableTrain.Vowels = v;

% Normalize test labels
v_test = upper(strtrim(FinalTableTest.Vowels));
v_test(v_test == "" | v_test == "N" | v_test == "NONE") = "";
FinalTableTest.Vowels = v_test;

% Add multi-hot encoding
FinalTableTrain = addVowelsMultiHot(FinalTableTrain);
FinalTableTest = addVowelsMultiHot(FinalTableTest);

% Extract Vowel A labels 
Ytrain_A = double(FinalTableTrain.isA);  
Ytest_A = double(FinalTableTest.isA);   

fprintf('Training distribution:\n');
fprintf('  Present (1): %d (%.1f%%)\n', sum(Ytrain_A==1), 100*sum(Ytrain_A==1)/length(Ytrain_A));
fprintf('  Absent (0):  %d (%.1f%%)\n', sum(Ytrain_A==0), 100*sum(Ytrain_A==0)/length(Ytrain_A));

fprintf('\nTest distribution:\n');
fprintf('  Present (1): %d (%.1f%%)\n', sum(Ytest_A==1), 100*sum(Ytest_A==1)/length(Ytest_A));
fprintf('  Absent (0):  %d (%.1f%%)\n\n', sum(Ytest_A==0), 100*sum(Ytest_A==0)/length(Ytest_A));
%% EXTRACT TRAINING FEATURES
fprintf(' Extracting Training Features \n');

N = height(FinalTableTrain);
Xtrain_cell = cell(N, 1);

tic;
for i = 1:N
    Xtrain_cell{i} = extractLSTMFeatures(FinalTableTrain, i, FREQ_MIN, FREQ_MAX, SAMPLE_RATE);
    
    if mod(i, 100) == 0 || i == N
        elapsed = toc;
        pct = 100 * i / N;
        eta = elapsed * (N - i) / i;
        
        fprintf('  [%6d/%6d] %.1f%% | Elapsed: %.1fm | ETA: %.1fm\n', ...
            i, N, pct, elapsed/60, eta/60);
    end
end

totalTime = toc;
fprintf('\n Training features ready (%.1f min)\n', totalTime/60);

[numFeatures, seqLength] = size(Xtrain_cell{1});
fprintf('  Feature shape: %d features × %d timesteps\n\n', numFeatures, seqLength);

clear FinalTableTrain
%% EXTRACT TEST FEATURES
fprintf(' Extracting Test Features \n');

M = height(FinalTableTest);
Xtest_cell = cell(M, 1);

tic;
for i = 1:M
    Xtest_cell{i} = extractLSTMFeatures(FinalTableTest, i, FREQ_MIN, FREQ_MAX, SAMPLE_RATE);
    
    if mod(i, 50) == 0 || i == M
        elapsed = toc;
        pct = 100 * i / M;
        eta = elapsed * (M - i) / i;
        
        fprintf('  [%4d/%4d] %.1f%% | ETA: %.1fm\n', i, M, pct, eta/60);
    end
end

fprintf('\n Test features ready (%.1f min)\n\n', toc/60);

clear FinalTableTest
%% TRAIN/VALIDATION SPLIT
fprintf(' Creating Train/Val Split \n');

valRatio = 0.2;
numSamples = length(Ytrain_A);
shuffleIdx = randperm(numSamples);
valCount = round(numSamples * valRatio);

trainIdx = shuffleIdx(valCount+1:end);
valIdx = shuffleIdx(1:valCount);
% Calculate the split point (No shuffling)
%trainCount = floor(numSamples * (1 - valRatio));


%trainIdx = 1:trainCount;                  % First 80% (0 to T)
%valIdx   = (trainCount + 1):numSamples;   % Last 20%  (T to End)

% Split data - keep as numeric 0/1
Xtrain = Xtrain_cell(trainIdx);
Ytrain = Ytrain_A(trainIdx);

Xval = Xtrain_cell(valIdx);
Yval = Ytrain_A(valIdx);

fprintf('Split completed:\n');
fprintf('  Training:   %d samples (%.0f%%)\n', length(trainIdx), (1-valRatio)*100);
fprintf('  Validation: %d samples (%.0f%%)\n\n', length(valIdx ), valRatio*100);

original_counts = [sum(Ytrain==0), sum(Ytrain==1)];
fprintf('Training set distribution (before augmentation):\n');
fprintf('  Absent (0):  %d\n', original_counts(1));
fprintf('  Present (1): %d\n\n', original_counts(2));
%%  DATA AUGMENTATION
fprintf(' Augmenting Training Data \n');

idx0 = find(Ytrain == 0);  % Absent
idx1 = find(Ytrain == 1);  % Present

fprintf('Current distribution:\n');
fprintf('  Absent (0):  %d\n', length(idx0));
fprintf('  Present (1): %d\n', length(idx1));

if length(idx0) < length(idx1)
    minority_idx = idx0;
    minority_label = 0;
    majority_idx = idx1;
    fprintf('   Minority class: Absent (0)\n');
else
    minority_idx = idx1;
    minority_label = 1;
    majority_idx = idx0;
    fprintf('   Minority class: Present (1)\n');
end
%%
targetN = length(majority_idx);
numToGenerate = targetN - length(minority_idx);

fprintf('\nGenerating %d augmented samples...\n', numToGenerate);

noise_std = 0.05;
augmented_features = cell(numToGenerate, 1);
augmented_labels = zeros(numToGenerate, 1);  % Numeric 0/1

tic;
for i = 1:numToGenerate
    source_idx = minority_idx(randi(length(minority_idx)));
    original = Xtrain{source_idx};
    
    noise = noise_std * randn(size(original), 'single');
    augmented_features{i} = original + noise;
    augmented_labels(i) = minority_label;
    
    if mod(i, 100) == 0 || i == numToGenerate
        fprintf('  Generated: %d/%d\n', i, numToGenerate);
    end
end

fprintf(' Augmentation done (%.1f sec)\n\n', toc);
%% Combine original + augmented
XtrainFinal = [Xtrain; augmented_features];
YtrainFinal_numeric = [Ytrain; augmented_labels];

% NOW convert to categorical with EXPLICIT ordering
YtrainFinal = categorical(YtrainFinal_numeric, [0, 1], {'Absent', 'Present'});
Yval_cat = categorical(Yval, [0, 1], {'Absent', 'Present'});

balanced_counts = [sum(YtrainFinal_numeric==0), sum(YtrainFinal_numeric==1)];
fprintf('Balanced training set:\n');
fprintf('  Absent (0):  %d (%.1f%%)\n', balanced_counts(1), 100*balanced_counts(1)/sum(balanced_counts));
fprintf('  Present (1): %d (%.1f%%)\n\n', balanced_counts(2), 100*balanced_counts(2)/sum(balanced_counts));

clear Xtrain augmented_features Xtrain_cell
%% DATA NORMALIZATION (Standard Scaler)
fprintf(' Normalizing Features (Standard Scaler) \n');

%  Concatenate all training samples to calculate global stats
allTrainData = horzcat(XtrainFinal{:}); 

mu = mean(allTrainData, 2);
sigma = (std(allTrainData, 0, 2)+eps);


fprintf('  Calculated stats for %d features\n', length(mu));

%  Apply normalization to Training Set
for i = 1:length(XtrainFinal)
    XtrainFinal{i} = (XtrainFinal{i} - mu) ./ sigma;
end

%  Apply SAME normalization to Validation Set (using train mu/sigma)
for i = 1:length(Xval)
    Xval{i} = (Xval{i} - mu) ./ sigma;
end

%  Apply SAME normalization to Test Set (using train mu/sigma)
for i = 1:length(Xtest_cell)
    Xtest_cell{i} = (Xtest_cell{i} - mu) ./ sigma;
end

fprintf('  Normalization complete.\n\n');

clear allTrainData % Free up memory
%% BUILD LSTM MODEL
fprintf('Building LSTM Architecture \n');

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    
    bilstmLayer(64, 'OutputMode', 'last', 'Name', 'bilstm1')
    batchNormalizationLayer('Name', 'bn1')
    dropoutLayer(0.5, 'Name', 'dropout1')
    
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% TRAINING CONFIGURATION
fprintf(' Training Configuration \n');
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xval, Yval_cat}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 15, ...
    'ExecutionEnvironment', 'auto', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'GradientThreshold', 1, ...
    'SequenceLength', 'longest', ...
    'SequencePaddingDirection', 'right');

fprintf('Settings:\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Learning rate: %.0e\n', options.InitialLearnRate);
fprintf('  Epochs: %d\n', options.MaxEpochs);
fprintf('  Batch size: %d\n', options.MiniBatchSize);
fprintf('  Validation check: every %d iterations\n', options.ValidationFrequency);
fprintf('  Early stopping patience: %d checks\n\n', options.ValidationPatience);
%% TRAIN MODEL
tic;
net_A = trainNetwork(XtrainFinal, YtrainFinal, layers, options);
trainTime = toc;

fprintf('\n');
fprintf(' Time: %.1f minutes \n', trainTime/60);
%% EVALUATE TRAINING SET
fprintf(' Evaluating Training Set \n');

YPred_train = classify(net_A, XtrainFinal);

Ytrue_train_num = double(YtrainFinal == "Present");
YPred_train_num = double(YPred_train == "Present");

TP_train = sum((YPred_train_num == 1) & (Ytrue_train_num == 1));
FP_train = sum((YPred_train_num == 1) & (Ytrue_train_num == 0));
FN_train = sum((YPred_train_num == 0) & (Ytrue_train_num == 1));
TN_train = sum((YPred_train_num == 0) & (Ytrue_train_num == 0));

acc_train = sum(YPred_train_num == Ytrue_train_num) / length(Ytrue_train_num);
prec_train = TP_train / (TP_train + FP_train + eps);
rec_train = TP_train / (TP_train + FN_train + eps);
F1_train = 2 * prec_train * rec_train / (prec_train + rec_train + eps);

fprintf('Results:\n');
fprintf('  Accuracy:  %.2f%%\n', acc_train*100);
fprintf('  Precision: %.2f%%\n', prec_train*100);
fprintf('  Recall:    %.2f%%\n', rec_train*100);
fprintf('  F1-Score:  %.2f%%\n', F1_train*100);
fprintf('\nConfusion Matrix:\n');
fprintf('  TN=%d  FP=%d\n', TN_train, FP_train);
fprintf('  FN=%d  TP=%d\n\n', FN_train, TP_train);
%% EVALUATE VALIDATION SET
fprintf(' Evaluating Validation Set \n');

YPred_val = classify(net_A, Xval);

Ytrue_val_num  = double(Yval_cat == "Present");
YPred_val_num = double(YPred_val  == "Present");


TP_val = sum((YPred_val_num == 1) & (Ytrue_val_num == 1));
FP_val = sum((YPred_val_num == 1) & (Ytrue_val_num == 0));
FN_val = sum((YPred_val_num == 0) & (Ytrue_val_num == 1));
TN_val = sum((YPred_val_num == 0) & (Ytrue_val_num == 0));

acc_val = sum(YPred_val_num == Ytrue_val_num) / length(Ytrue_val_num);
prec_val = TP_val / (TP_val + FP_val + eps);
rec_val = TP_val / (TP_val + FN_val + eps);
F1_val = 2 * prec_val * rec_val / (prec_val + rec_val + eps);

fprintf('Results:\n');
fprintf('  Accuracy:  %.2f%%\n', acc_val*100);
fprintf('  Precision: %.2f%%\n', prec_val*100);
fprintf('  Recall:    %.2f%%\n', rec_val*100);
fprintf('  F1-Score:  %.2f%%\n', F1_val*100);
fprintf('\nConfusion Matrix:\n');
fprintf('  TN=%d  FP=%d\n', TN_val, FP_val);
fprintf('  FN=%d  TP=%d\n\n', FN_val, TP_val);
%% EVALUATE TEST SET 
fprintf(' Evaluating Test Set \n');

Ytest_cat = categorical(Ytest_A, [0, 1], {'Absent', 'Present'});
YPred_test = classify(net_A, Xtest_cell);

Ytest_num  = double(Ytest_cat == "Present");
YPred_test_num= double(YPred_test  == "Present");


TP = sum((YPred_test_num == 1) & (Ytest_num == 1));
FP = sum((YPred_test_num == 1) & (Ytest_num == 0));
FN = sum((YPred_test_num == 0) & (Ytest_num == 1));
TN = sum((YPred_test_num == 0) & (Ytest_num == 0));

acc_test = sum(YPred_test_num == Ytest_num) / length(Ytest_num);
prec_test = TP / (TP + FP + eps);
rec_test = TP / (TP + FN + eps);
F1_test = 2 * prec_test * rec_test / (prec_test + rec_test + eps);

fprintf('Results:\n');
fprintf('  Accuracy:  %.2f%%\n', acc_test*100);
fprintf('  Precision: %.2f%%\n', prec_test*100);
fprintf('  Recall:    %.2f%%\n', rec_test*100);
fprintf('  F1-Score:  %.2f%%\n', F1_test*100);
fprintf('\nConfusion Matrix:\n');
fprintf('  TN=%d  FP=%d\n', TN, FP);
fprintf('  FN=%d  TP=%d\n\n', FN, TP);
%% COMPARATIVE SUMMARY
fprintf('\n');
fprintf('╔═══════════════════════════════════════════════════════════════════╗\n');
fprintf('║                   COMPARATIVE SUMMARY                             ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Metric      │   Training  │  Validation │    Test     │  T-T Gap ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Accuracy    │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', ...
    acc_train*100, acc_val*100, acc_test*100, (acc_train-acc_test)*100);
fprintf('║  Precision   │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', ...
    prec_train*100, prec_val*100, prec_test*100, (prec_train-prec_test)*100);
fprintf('║  Recall      │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', ...
    rec_train*100, rec_val*100, rec_test*100, (rec_train-rec_test)*100);
fprintf('║  F1-Score    │   %6.2f%%   │   %6.2f%%   │  %6.2f%%   │  %.1f%%    ║\n', ...
    F1_train*100, F1_val*100, F1_test*100, (F1_train-F1_test)*100);
fprintf('╚═══════════════════════════════════════════════════════════════════╝\n\n');
%% VISUALIZATIONS
fprintf(' Creating Visualizations \n');

%  CONFUSION MATRICES
figure('Position', [50 50 1500 500], 'Name', 'Confusion Matrices');

% Training Matrix
subplot(1,3,1);
cmTrain = confusionchart(YtrainFinal, YPred_train);
title(sprintf('Training (Acc: %.1f%%)', acc_train*100));

% Validation Matrix
subplot(1,3,2);
cmVal = confusionchart(Yval_cat, YPred_val);
title(sprintf('Validation (Acc: %.1f%%)', acc_val*100));

% Test Matrix
% Note: We use the variables created in Block 13
subplot(1,3,3);
cmTest = confusionchart(Ytest_cat, YPred_test);
title(sprintf('Test (Acc: %.1f%%)', acc_test*100));

fprintf(' Confusion matrices created\n');

% 2. CLASS DISTRIBUTION
figure('Position', [100 100 1000 500], 'Name', 'Class Distribution');

subplot(1,2,1);
b1 = bar(categorical({'Absent','Present'}), original_counts);
title('Before Augmentation');
ylabel('Count');
b1.FaceColor = 'flat';
b1.CData(1,:) = [0.8 0.3 0.3]; % Red
b1.CData(2,:) = [0.3 0.8 0.3]; % Green

subplot(1,2,2);
b2 = bar(categorical({'Absent','Present'}), balanced_counts);
title('After Augmentation');
ylabel('Count');
b2.FaceColor = 'flat';
b2.CData(1,:) = [0.8 0.3 0.3];
b2.CData(2,:) = [0.3 0.8 0.3];

fprintf(' Distribution charts created\n\n');
%% ROC & AUC CALCULATION AND PLOTTING
fprintf('Calculating ROC and AUC \n');

% GET PROBABILITY SCORES
% We need the second column of 'scores' which corresponds to the "Present" class
[~, scores_train] = classify(net_A, XtrainFinal);
[~, scores_val]   = classify(net_A, Xval);
[~, scores_test]  = classify(net_A, Xtest_cell);

%  CALCULATE PERFCURVE (ROC COORDINATES)
% The 3rd output '~' is thresholds , 4th output is AUC
[Xroc_train, Yroc_train, ~, AUC_train] = perfcurve(YtrainFinal, scores_train(:,2), 'Present');
[Xroc_val,   Yroc_val,   ~, AUC_val]   = perfcurve(Yval_cat,    scores_val(:,2),   'Present');
[Xroc_test,  Yroc_test,  ~, AUC_test]  = perfcurve(Ytest_cat,   scores_test(:,2),  'Present');

fprintf('AUC Results:\n');
fprintf('  Training AUC:   %.4f\n', AUC_train);
fprintf('  Validation AUC: %.4f\n', AUC_val);
fprintf('  Test AUC:       %.4f\n', AUC_test);

% PLOT ROC CURVES
figure('Name', 'ROC Curves', 'Color', 'w', 'Position', [100 100 600 500]);
hold on;

% Plot lines
plot(Xroc_train, Yroc_train, 'LineWidth', 2, 'Color', 'b', 'DisplayName', sprintf('Train (AUC=%.2f)', AUC_train));
plot(Xroc_val,   Yroc_val,   'LineWidth', 2, 'Color', 'g', 'LineStyle', '--', 'DisplayName', sprintf('Val (AUC=%.2f)', AUC_val));
plot(Xroc_test,  Yroc_test,  'LineWidth', 2, 'Color', 'r', 'LineStyle', '-.', 'DisplayName', sprintf('Test (AUC=%.2f)', AUC_test));

% Plot diagonal "Random Chance" line
plot([0 1], [0 1], 'k:', 'LineWidth', 1, 'DisplayName', 'Random Chance');

% Formatting
xlabel('False Positive Rate (1 - Specificity)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('True Positive Rate (Sensitivity)', 'FontWeight', 'bold', 'FontSize', 14);
title('ROC Curves - Vowel A Detection', 'FontSize', 18);

% Make axes tick labels larger and bold
ax = gca;
ax.FontSize = 16;          % Tick labels size
ax.FontWeight = 'bold';    % Tick labels bold

% Legend formatting
lgd = legend('Location', 'SouthEast');
lgd.FontSize = 12;         % Legend font size
lgd.FontWeight = 'bold';   % Legend bold

grid on;
axis square; % Makes the plot square-shaped
hold off;

fprintf(' ROC Curves plotted successfully.\n\n');
%% 
%% EMPIRICAL CHANCE-LEVEL AUC (Permutation Test)
fprintf(' Computing Empirical Chance-Level AUC \n');

numPermutations = 1000;
permAUC_test = zeros(numPermutations, 1);

tic;
for p = 1:numPermutations
    % Shuffle the true test labels randomly
    permLabels = Ytest_cat(randperm(length(Ytest_cat)));
    
    % Compute AUC against same model scores (no re-inference needed)
    [~, ~, ~, auc_perm] = perfcurve(permLabels, scores_test(:,2), 'Present');
    permAUC_test(p) = auc_perm;
    
    if mod(p, 200) == 0
        fprintf('  Permutation %d/%d\n', p, numPermutations);
    end
end

chanceAUC_mean = mean(permAUC_test);
chanceAUC_std  = std(permAUC_test);
chanceAUC_95   = prctile(permAUC_test, 95);  % 95th percentile = significance threshold

fprintf('\nPermutation Results (%d runs):\n', numPermutations);
fprintf('  Chance AUC mean:  %.4f\n', chanceAUC_mean);
fprintf('  Chance AUC std:   %.4f\n', chanceAUC_std);
fprintf('  95th percentile:  %.4f  (significance threshold)\n', chanceAUC_95);
fprintf('  Model Test AUC:   %.4f\n', AUC_test);
fprintf('  Above chance by:  %.4f\n\n', AUC_test - chanceAUC_mean);

% p-value: proportion of permutations that exceeded the real AUC
pValue = mean(permAUC_test >= AUC_test);
if pValue == 0
    fprintf('  p-value (one-tailed): <0.001', pValue);
else
    fprintf('  p-value (one-tailed): %.4f', pValue);
end
if pValue < 0.05
    fprintf('  --> Significant (p < 0.05)\n\n');
else
    fprintf('  --> Not significant\n\n');
end

%% PERMUTATION DISTRIBUTION FIGURE
figure('Name', 'Permutation Test Distribution', 'Color', 'w', 'Position', [100 100 600 450]);
hold on;

% Histogram of permutation AUCs
histogram(permAUC_test, 40, ...
    'FaceColor', [0.6 0.75 0.9], ...
    'EdgeColor', 'none', ...
    'DisplayName', 'Permutation AUCs');

% Vertical lines
xline(chanceAUC_mean, '--', 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Chance mean = %.3f', chanceAUC_mean));
xline(chanceAUC_95, '-.', 'Color', [0.8 0.5 0.1], 'LineWidth', 1.5, ...
    'DisplayName', sprintf('95th pct = %.3f', chanceAUC_95));
xline(AUC_test, '-', 'Color', [0.1 0.6 0.3], 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Model AUC = %.3f', AUC_test));

xlabel('AUC', 'FontWeight', 'bold', 'FontSize', 13);
ylabel('Count', 'FontWeight', 'bold', 'FontSize', 13);
title('Permutation Test — Empirical Chance Distribution', 'FontSize', 14);

ax = gca;
ax.FontSize = 12;
ax.FontWeight = 'bold';

lgd = legend('Location', 'NorthWest');
lgd.FontSize = 10;
grid on;
hold off;

fprintf(' Permutation distribution figure created.\n\n');
%% UPDATED ROC PLOT WITH EMPIRICAL CHANCE BAND
figure('Name', 'ROC Curves with Empirical Chance', 'Color', 'w', 'Position', [100 100 600 500]);
hold on;

% Shade the permutation distribution as a band
% Build a shaded region: mean ± 2*std across FPR axis

% We only have scalar AUC values per permutation, so shade using
% the 5th–95th percentile bounds as horizontal reference lines
yl1 = yline(chanceAUC_95,   '--', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.2);
yl2 = yline(chanceAUC_mean, ':',  'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
yl1.DisplayName = sprintf('Chance 95th pct = %.3f', chanceAUC_95);
yl2.DisplayName = sprintf('Chance mean = %.3f',     chanceAUC_mean);

if pValue == 0
    pStr = '<0.001';
else
    pStr = sprintf('%.3f', pValue);
end

% Plot model ROC curves
plot(Xroc_train, Yroc_train, 'LineWidth', 2, 'Color', 'b', ...
    'DisplayName', sprintf('Train (AUC=%.3f)', AUC_train));
plot(Xroc_val,   Yroc_val,   'LineWidth', 2, 'Color', 'g', 'LineStyle', '--', ...
    'DisplayName', sprintf('Val   (AUC=%.3f)', AUC_val));
plot(Xroc_test,  Yroc_test,  'LineWidth', 2, 'Color', 'r', 'LineStyle', '-.', ...
    'DisplayName', sprintf('Test  (AUC=%.3f, p=%s)', AUC_test, pStr));

% Fixed diagonal for reference
plot([0 1], [0 1], 'k:', 'LineWidth', 0.8, 'DisplayName', 'Theoretical chance');

xlabel('False Positive Rate (1 - Specificity)', 'FontWeight', 'bold', 'FontSize', 14);
ylabel('True Positive Rate (Sensitivity)',       'FontWeight', 'bold', 'FontSize', 14);
title('ROC Curves — Vowel A Detection', 'FontSize', 18);

ax = gca;
ax.FontSize = 14;
ax.FontWeight = 'bold';

lgd = legend('Location', 'SouthEast');
lgd.FontSize = 11;
lgd.FontWeight = 'bold';

grid on; axis square;
hold off;

fprintf(' ROC plot with empirical chance band created.\n\n');
%% REPLACE the FINAL SUMMARY block with:
fprintf('╔════════════════════════════════════════════════════╗\n');
fprintf('║  TRAINING COMPLETE                                 ║\n');
fprintf('╠════════════════════════════════════════════════════╣\n');
fprintf('║  Total time:        %.1f minutes                   ║\n', trainTime/60);
fprintf('║  Test accuracy:     %.2f%%                         ║\n', acc_test*100);
fprintf('║  Test F1-Score:     %.2f%%                         ║\n', F1_test*100);
fprintf('║  Test AUC:          %.4f                           ║\n', AUC_test);
fprintf('╠════════════════════════════════════════════════════╣\n');
fprintf('║  PERMUTATION TEST (n=1000)                         ║\n');
fprintf('║  Chance AUC mean:   %.4f                           ║\n', chanceAUC_mean);
fprintf('║  Chance 95th pct:   %.4f                           ║\n', chanceAUC_95);
fprintf('║  Above chance by:   %.4f                           ║\n', AUC_test - chanceAUC_mean);
if pValue == 0
    fprintf('║  p-value:           <0.001 (significant)           ║\n');
elseif pValue < 0.05
    fprintf('║  p-value:           %.4f  (significant)            ║\n', pValue);
else
    fprintf('║  p-value:           %.4f  (not significant)        ║\n', pValue);
end
fprintf('╚════════════════════════════════════════════════════╝\n');
fprintf('\n All done!\n');
%% FINAL SUMMARY
fprintf('╔════════════════════════════════════════════╗\n');
fprintf('║  TRAINING COMPLETE                         ║\n');
fprintf('╠════════════════════════════════════════════╣\n');
fprintf('║  Total time: %.1f minutes                  ║\n', trainTime/60);
fprintf('║  Test accuracy: %.2f%%                     ║\n', acc_test*100);
fprintf('║  Test F1-Score: %.2f%%                     ║\n', F1_test*100);
fprintf('╚════════════════════════════════════════════╝\n');
fprintf('\n All done!\n');
%% HELPER FUNCTION
function features = extractLSTMFeatures(table, idx, freqMin, freqMax, sampleRate)
    
    % Spectrogram
    spec1 = abs(single(table.EMG_ch1_Spectrogram{idx}));
    spec2 = abs(single(table.EMG_ch2_Spectrogram{idx}));

    
    freqsSpec = linspace(0, sampleRate/2, size(spec1,1));
    freqIdxSpec = (freqsSpec >= freqMin) & (freqsSpec <= freqMax);
    spec1_filtered = spec1(freqIdxSpec, :);
    spec2_filtered = spec2(freqIdxSpec, :);

    currentWidth = size(spec1_filtered, 2);
    
    % Align time
    %minTimeSteps = min([nTimeSamples, size(cwt1_filtered,2), size(spec1_filtered,2)]);
    spec1_resized = imresize(spec1_filtered, [450, currentWidth]);
    spec2_resized = imresize(spec2_filtered, [450, currentWidth]);
    
    % Stack features
    features = [spec1_resized; spec2_resized];
    features = single(features);
end
