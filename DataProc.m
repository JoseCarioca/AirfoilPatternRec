clearvars -except data
, close all, clc;
addpath("pattern\");

directory       = "./harth";
files           = dir(directory);
TOTAL_CENTROIDS = cell(length(files) -2, 10);
K_CENTROIDS     = 11;
addvelocity = true;
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
%dat = dat(randperm(length(dat)), :);
x = dat(:, 1:7)';
y = dat(:, 8)';
y(y==140) = 13;
y(y==130) = 13;
y(y==14)  = 13;
y(y==5)   = 4;


if addvelocity % bloque para añadir 6 caracteristicas más: la velocidad inmediata de cada punto
    aux0 = [zeros(size(x,1)-1,1) x(2:end,1:end-1)];
    vel = [x ; (x(2:end,:) - aux0)];
    x = vel;
end
% Normalizar (parece que pierde demasiada precision)
x(:, :) = normalize(x(:, :));
[x, y] = shuffle(x,y);
lda = fisher( x, y, 3 );

x = lda * x;
maxcvIt = 5;
for numCentroides = 1:10
    for cvIt = 1:maxcvIt
        [training_x, test_x, training_y, test_y] = crossval(x, y, 10, cvIt );
    
        % Create a LinearDiscriminantAnalysis object
        lda    = fitcdiscr(training_x', training_y');
        y_pred = predict(lda, test_x')';
    
        aciertos   = find(y_pred == test_y);
        lda_perc   = (size(aciertos, 2) / size(test_y, 2) ) * 100;  
    
        TOTAL_LDA{cvIt} = {lda_perc, lda};
        
        [glm_perc, p] = GLM(training_x, training_y, test_x, test_y);
    
        TOTAL_GLM{cvIt} = {glm_perc, p};
    
        %TOTAL_CENTROIDS{cvIt} = K_MEANS_PROCMAHAL(training_x, training_y, test_x, test_y, K_CENTROIDS);
        TOTAL_CENTROIDS{cvIt} = K_MEANS_PROC(training_x, training_y, test_x, test_y, numCentroides);

        disp(cvIt); 
        TOTAL_LDA_COMPACT12{(numCentroides-1)*maxcvIt+cvIt}{1} = TOTAL_LDA{cvIt}{1};
        TOTAL_LDA_COMPACT12{(numCentroides-1)*maxcvIt+cvIt}{2} = compact(TOTAL_LDA{cvIt}{2});
    end
    for c = 1:10
        %TOTAL_LDA_COMPACT12{c*numCentroides}{1} = TOTAL_LDA{c}{1};
        %TOTAL_LDA_COMPACT12{c*numCentroides}{2} = compact(TOTAL_LDA{c}{2});
    end

end

%eliminando dataset del struct
%TOTAL_LDA_COMPACT = cell(10,2);
porcentaje = zeros(10,maxcvIt);
for c = 1:length( TOTAL_LDA_COMPACT12)
   porcentaje(c) = TOTAL_LDA_COMPACT12{c}{1}; 
   %TOTAL_LDA_COMPACT12{c}{1} = TOTAL_LDA{c}{1};
   %TOTAL_LDA_COMPACT12{c}{2} = compact(TOTAL_LDA{c}{2});
end 

plot(porcentaje); hold on;
hold off;

save("PROCESSED_WEIGHTS12_VELOCITYFISHER2.mat", "TOTAL_CENTROIDS","TOTAL_GLM", "TOTAL_LDA_COMPACT12");
