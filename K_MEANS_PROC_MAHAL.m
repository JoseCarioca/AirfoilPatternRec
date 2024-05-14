
function [TOTAL_CENTROIDS] = K_MEANS_PROC_MAHAL(training_x,training_y, test_x, test_y, k, clases)
    nclases = length(clases);
    centroides = zeros(size(training_x,1),k*nclases); %k centroides por cada clase
    %rango de centroides (i-1)*k+1 : i*k
    centr_pertenece = sort(repmat(1:nclases, 1, k)); %pertenencia a cada clase

    for i = clases % de 1 a 8
        x_clase = training_x(:,find(training_y == i));
        centroides(:,(i-1)*k+1:i*k) = kmeans(x_clase,k);
        covarianza{i} = covpat(x_clase);
    end

    for j=1:k*nclases %num total de centroides otra forma size2 centroides
      distMah(j,1:length(test_x)) = d_mahal(test_x,centroides(:,j),covarianza{centr_pertenece(j)});  
      %centr_pertenece(j) devuelve a qué clase pertenece el calculo y por
      %tanto qué covarianza usar
    end

    [~,c] = min(distMah); %c te dice el numero de centroide 
    c_pert = centr_pertenece(c); % c_pert es a qué clase pertenece ese centroide y lo que nos interesa
    aciertos_mahal(cvIt) = (length(find(c_pert(1:length(test_y)) == test_y))/length(test_y))*100; 
    %testx.size != testy.size a veces
    TOTAL_CENTROIDS = {aciertos_mah, c};
end


