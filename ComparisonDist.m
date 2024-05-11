
clc, clear all
addpath("pattern")
data = readtable("ALL_FILES12.csv");

fin = 600000;

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
%data = readtable("ALL_FILES12.csv");

dat = table2array(data(1:fin, :));
dat = dat(randperm(length(dat)), :);
x = dat(:, 1:7)';
y = dat(:, 8)';
y(y==140) = 13;
y(y==130) = 13;
y(y==14)  = 13;
y(y==5)   = 4;
% Normalizar (parece que pierde demasiada precision)
x(:, :) = normalize(x(:, :));

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
