
clc, clearvars -except dat data
addpath("pattern")
data = readtable("ALL_FILES12.csv");

fin = size(data,1);

directory       = "./harth12";
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
prueba = false; %toggle para datos reales o prueba

if ~prueba
    %data = readtable("ALL_FILES12.csv");
    dat = table2array(data(1:fin, :));
    %dat = dat(randperm(length(dat)), :);
    x = dat(:, 1:7)';
    y = dat(:, 8)';
    y(y==5)   = 4;
    y(y==13) = 5;
    y(y==140) = 5; 
    y(y==130) = 5;
    y(y==14)  = 5;

    %velocity
    x = [x; ( x(2:end,:) - [zeros(size(x,1)-1,1) x(2:end,1:end-1)] )];
end

%% Labels
% 1: walking	
% 2: running	
% 3: shuffling
% 4: stairs (ascending)	stairs (descending)	
% 5: cycling (sit)	cycling (stand) cycling (sit, inactive) cycling (stand, inactive)
% 6: standing	
% 7: sitting	
% 8: lying	
% 9 cambiar cyling inactive por label 9 en el futuro?
% 14: 	
% 130: 
% 140: 

if prueba 
    %prueba codigo
    A = load("mitbih_train.csv"); B = load("mitbih_test.csv");
    x = [A; B];
    y = x(:,end) + 1; x = x(:,1:size(x,2)/2);
    y = y'; x = x';
end

%% NEW Labels
% 1: walking	
% 2: running	
% 3: shuffling
% 4: stairs (ascending)	& stairs (descending)	
% 5: cycling (sit)	& cycling (stand) cycling (sit, inactive) & cycling (stand, inactive)
% 6: standing	
% 7: sitting	
% 8: lying	

% Normalizar (parece que pierde demasiada precision)
%%  mezclamos y agarramos la mitad de los datos de training y mitad de test (cambiar con CROSSVAL)
x(:, :) = normalize(x(:, :));
[x, y] = shuffle(x, y);
%x_tr = x(1:size(x,1)/2,:); y_tr =  y(1:size(x,1)/2,:); %primera mitad de los datos
%x_ts = x(size(x,1)/2+1:end,:); y_ts = y(size(x,1)/2+1:end,:); %sumar 1 a las clases para iterar en ellas
%x_tr = x(:,1:size(x,2)/2); y_tr =  y(:,1:size(x,2)/2); %primera mitad de los datos
%x_ts = x(:,size(x,2)/2+1:end); y_ts = y(:,size(x,2)/2+1:end); %sumar 1 a las clases para iterar en ellas

%% . Mahalannobis 
K = 10;
aciertos = zeros(1,K); % %aciertos
clases = unique(y);
nclases = length(clases);
aciertos_mahal = zeros(1,K); % %aciertos
lda = fisher(x, y, 2);

x = lda * x;
TOTAL_ACIERTOS = zeros(K,10);
for k = 1:K
    for cvIt = 1:10
        [training_x, test_x, training_y, test_y] = crossval(x, y, 10, cvIt );
        centroides = zeros(size(training_x,1),k*nclases); %k centroides por cada clase
        %rango de centroides (i-1)*k+1 : i*k
        centr_pertenece = sort(repmat(1:nclases, 1, k)); %pertenencia a cada clase
    
        for i = 1:nclases % de 1 a 8
            x_clase = training_x(:,find(training_y == i));
            centroides(:,(i-1)*k+1:i*k) = kmeans(x_clase,k);
            covarianza{i} = covpat(x_clase);
        end
        distMah = zeros(k*nclases, size(test_x,2));
        for j=1:k*nclases %num total de centroides otra forma size2 centroides
          distMah(j,:) = d_mahal(test_x,centroides(:,j),covarianza{centr_pertenece(j)});  
          %centr_pertenece(j) devuelve a qué clase pertenece el calculo y por
          %tanto qué covarianza usar
        end
    
        [~,c] = min(distMah); %c te dice el numero de centroide 
        c_pert = centr_pertenece(c); % c_pert es a qué clase pertenece ese centroide y lo que nos interesa
        aciertos_mahal(cvIt) = (length(find(c_pert == test_y))/length(test_y))*100; 
        %testx.size != testy.size a veces
    end
    TOTAL_ACIERTOS(k,:) = aciertos_mahal;
    %figure(k),
    disp(aciertos_mahal); %legend(cvIt);
    aciertosK(k) = mean(aciertos_mahal);
end
figure(10), plot(aciertosK);
  

%classes = 6; %number of classes
[x,y] = shuffle(x,y);

nErrEuc       = 0; 
nErrManh      = 0; 
nErrMahal     = 0;
classes       = unique(y);
nclasses      = length(classes);
Confmat_manh  = zeros(nclasses,nclasses);
Confmat_euc   = zeros(nclasses,nclasses);
Confmat_mahal = zeros(nclasses,nclasses);
for k =1:10
    [xtrn,xtst,ytrn,ytst] = crossval(x,y,10,k);
    for i=1:nclasses
        index = find(ytrn==classes(i));
        m{i} = meanpat(xtrn(:,index));
        C{i} = covpat(xtrn(:,index));
    end
    %classification
    dist_euclidea = zeros(nclasses,length(xtst));
    dist_manhattan = zeros(nclasses,length(xtst));
    dist_mahal = zeros(nclasses,length(xtst));
    for i=1:nclasses
        dist_manhattan(i,:) = d_manhat(xtst,m{i});
        dist_euclidea(i,:) = d_euclid(xtst,m{i});
        dist_mahal(i,:) = d_mahal(xtst, m{i}, C{i});     
    end
    %Confmat_manh
    [~,cl] = min(dist_manhattan);
    nErrEuc = nErrEuc + length(find (ytst~=classes(cl)) );
    Confmat_manh = Confmat_manh + confusionmat(ytst,classes(cl));

    
    [~,cl] = min(dist_euclidea);
    nErrManh = nErrManh + length(find(ytst~=classes(cl)));
    Confmat_euc = Confmat_euc + confusionmat(ytst,classes(cl));

    [~,cl] = min(dist_mahal);
    nErrMahal = nErrMahal + length(find(ytst~=classes(cl)));
    Confmat_mahal = Confmat_mahal + confusionmat(ytst,classes(cl));
    
    
end

for i= 1:nclasses
    Precision(i) = Confmat_manh(i,i)/(sum(Confmat_manh(i,:)));
    Recall(i) =  Confmat_manh(i,i)/(sum(Confmat_manh(:,i)));
    F1_score(i) = 2*Precision(i)*Recall(i)/(Precision(i)+Recall(i));
end
disp(F1_score);

for i= 1:nclasses
    Precision(i) = Confmat_euc(i,i)/(sum(Confmat_euc(i,:)));
    Recall(i) =  Confmat_euc(i,i)/(sum(Confmat_euc(:,i)));
    F1_score(i) = 2*Precision(i)*Recall(i)/(Precision(i)+Recall(i));
end
disp(F1_score);

for i= 1:classes
    Precision(i) = Confmat_mahal(i,i)/(sum(Confmat_mahal(i,:)));
    Recall(i) =  Confmat_mahal(i,i)/(sum(Confmat_mahal(:,i)));
    F1_score(i) = 2*Precision(i)*Recall(i)/(Precision(i)+Recall(i));
end
disp(F1_score);
%calculo de f1_score

%E_porcentaje = n