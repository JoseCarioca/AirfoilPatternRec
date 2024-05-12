clear all, close all, clc;
addpath("./pattern");

directory       = "./harth";
files           = dir(directory);
TOTAL_CENTROIDS = cell(length(files) -2, 10);
K_CENTROIDS     = 11;

%% Labels
% 1: walking	
% 2: running	
% 3: shuffling
% 4: stairs (ascending)	
% 5: stairs (descending)	
% 6: standing	
% 7: sitting	
% 8: lying	
% 13: cycling (sit)	
% 14: cycling (stand)	
% 130: cycling (sit, inactive)
% 140: cycling (stand, inactive)

%% Procesamos 

data = readtable("ALL_FILES12.csv");

dat = table2array(data(:, :));
dat = dat(randperm(length(dat)), :);
x = dat(:, 1:7)';
y = dat(:, 8)';

y(y==140) = 13;
y(y==130) = 13;
y(y==14)  = 13;
y(y==5)   = 4;

x(:, :) = normalize(x(:, :));

%% Discriminante lineal
%  Nos olvidamos de las caracteristicas menos utiles
lda = fisher( x, y, 3 );
x = lda * x;

PredictMatlab = zeros(1, 10);
PredictKMeans = zeros(1, 10);
PredictGLM    = zeros(1, 10);

for cvIt = 1:10
    [training_x, test_x, training_y, test_y] = crossval(x, y, 10, cvIt );

    % Create a LinearDiscriminantAnalysis object
     lda    = fitcdiscr(training_x', training_y', 'OptimizeHyperparameters','auto',...
     'HyperparameterOptimizationOptions',...
     struct('AcquisitionFunctionName','expected-improvement-plus'));
     y_pred = predict(lda, test_x')';
     
     aciertos   = find(y_pred == test_y);
     lda_perc   = (size(aciertos, 2) / size(test_y, 2) ) * 100;  

    PredictMatlab(1, cvIt) = lda_perc;
    
    [glm_perc, p] = GLM(training_x, training_y, test_x, test_y);

    PredictGLM(1, cvIt) = glm_perc;

    centroides = K_MEANS_PROC(training_x, training_y, test_x, test_y, K_CENTROIDS);

    PredictKMeans(1, cvIt) = centroides{1};

end

save("AciertosModelos.mat", "PredictMatlab", "PredictGLM", "PredictKMeans");
