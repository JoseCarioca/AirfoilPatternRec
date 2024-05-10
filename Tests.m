clear all, close all, clc;

load("PROCESSED_WEIGHTS.mat");

K_CENTROIDS     = 11;

addpath("pattern\");

directory       = "./harth";
files           = dir(directory);

persona_score_max = 0;

for i = 1:size(TOTAL_CENTROIDS, 1)
    nuevo = TOTAL_CENTROIDS{i, :};
    if nuevo{1} > persona_score_max && nuevo{1} < 90
        persona_score_max = nuevo{1};
        mejor_centroide = nuevo{2};
    end
end

centroides = [];
for col = 1:size(mejor_centroide, 2)
    centroides = [centroides mejor_centroide(:, col)];
end

%% Procesamos cada fichero
for f = 3:length(files)
    data = readtable(fullfile(directory, files(f).name));
    % conversion a scalar
    deis = data.timestamp.Hour * 60 * 60 + data.timestamp.Minute * 60 + data.timestamp.Second;
    data.timestamp = deis;

    if any(data.Properties.VariableNames == "Var1")
        data = removevars(data, "Var1");
    end
    if any(data.Properties.VariableNames == "index")
        data = removevars(data, "index");
    end
    dat = table2array(data(:, :));
    dat = dat(randperm(length(dat)), :);
    test_x = dat(:, 1:7)';
    test_y = dat(:, 8)';
    % Normalizar (parece que pierde demasiada precision)
    test_x(:, :) = normalize(test_x(:, :));

    current_labels = unique(test_y);
    valoresCentroides = 1:length(current_labels)*K_CENTROIDS;

    j = 1;
    for row = 1:length(current_labels) * K_CENTROIDS
        valoresCentroides(row) = current_labels(j);
        if mod(row, K_CENTROIDS) == 0
            j = j + 1;
        end
    end

    lda = fisher( test_x, test_y, 3 );

    test_x = lda * test_x;
    yest = zeros(1, size(test_y, 2));

    for i = 1:length(test_y)
        d = d_euclid(test_x(:, i), centroides);
        [~,pos] = sort(d); 
        yest(i) = valoresCentroides(pos(1));
    end
    acierto      = (find(yest == test_y));
    aciertos_mah = (size(acierto, 2) / size(test_y, 2)) * 100;
    disp(aciertos_mah)
end