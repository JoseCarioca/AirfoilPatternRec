
function [TOTAL_CENTROIDS] = K_MEANS_PROC_MAHAL(training_x,training_y, test_x, test_y, K_CENTROIDS)
 %% Utilizar cross validation
    current_labels = unique(training_y);
    c   = {length(current_labels)};
    cov = {length(current_labels)};
    for i = 1:length(current_labels)
        c{i}   = {current_labels(i),kmeans(training_x(:, training_y == current_labels(i)), K_CENTROIDS)};
        cov{i} = covpat(training_x(:, training_y == current_labels(i)));
    end

    valoresCentroides = 1:length(current_labels)*K_CENTROIDS;

    j = 1;
    for row = 1:length(valoresCentroides)
        valoresCentroides(row) = c{j}{1};
        if mod(row, K_CENTROIDS) == 0
            j = j + 1;
        end
    end
    %% Procesar clasificacion
    centroides = [];
    for row = 1:length(c)
        centroides = [centroides c{row}{2}];
    end
    yest = zeros(1, size(test_y, 2));
    for i = 1:length(test_y)
        %d = d_euclid(test_x(:, i), centroides);
        d = d_mahal(test_x(:,i), centroides, cov{i});
        [~,pos] = sort(d); % Aqui se encuentra la posicion de menor distancia
        yest(i) = valoresCentroides(pos(1));
    end
    acierto      = (find(yest == test_y));
    aciertos_mah = (size(acierto, 2) / size(test_y, 2)) * 100;
    %disp(c)
    disp(aciertos_mah)
    TOTAL_CENTROIDS = {aciertos_mah, centroides};
end

