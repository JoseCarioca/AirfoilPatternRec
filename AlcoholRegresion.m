%% estudio GLM

clearvars -except d1 d2 celda,
close all, clc;
addpath("./pattern");

ywork = celda(:,23); ywend = celda(:,24);
x = celda; x(:,23:24) = [];
[x, ywend] = shuffle(x',ywend');
[x(:, :),centering, scaling] = normalize(x(:, :));

%% Discriminante lineal
%  Nos olvidamos de las caracteristicas menos utiles
lda = fisher( x, ywend, 3);
x = lda * x;
plotpat(x,ywend);
%probar con ywork tambien


plotpat(x,ywend);
aux = 0;
for cvIt = 1:10
    [training_x, test_x, training_y, test_y] = crossval(x, ywend, 10, cvIt );

    % Create a LinearDiscriminantAnalysis object

    coeficientes{cvIt} = myGLM(training_x',training_y');
    error(cvIt) = calculaError(test_x',test_y',coeficientes{cvIt});
    
  %Ecv(i) = aux/K;  

end
%Ecv = aux/S;

x_row = xM;
y_row = yM;
trn = int64(0.75*length(x_row));
tst = length(x_row) - trn;
Ers = zeros(1,5);
for i=1:5
     aux = 0;
     for k=1:1000
        [x_row,y_row]=shuffle(x_row,y_row);
        xtrn = x_row(1:trn,:); xtst = x_row(trn+1:end,:);
        ytrn = y_row(1:trn); ytst = y_row(trn+1:end);
        [yestim,p] = polyValFit(i,xtrn',ytrn', xtst);
        Ers(i) = Ers(i) + mean((ytst-yestim).^2);
     end
     Ers(i) = Ers(i)/1000;
     disp("Estimacion del error con Random Sampling para modelo "+i+": "+Ers(i));
end

% Error = zeros(1,5);
% for n=1:5
%     [y_estim,p] = polyValFit(n, xM,yM, xM);
%     Error(n) = mean((yM-y_estim).^2);
%     disp("Estimacion del error con Resusticicion (1M datos) para modelo "+n+": "+Error(n));
% end




function coefficients = myGLM(X, y)
    % Añadir columna de unos para el término de sesgo
    X = [X, ones(size(X, 1), 1)];
    % Calcular coeficientes usando estimación de mínimos cuadrados ordinarios
    coefficients = pinv(X)*y;
end

function er = calculaError(X, y, coefficients)
    % Obtener el número de observaciones
    N = size(X, 1);
    % Añadir columna de unos para el término de sesgo
    X = [X ones(N, 1)];
    % Calcular las predicciones del modelo
    y_pred = X * coefficients;
    % Calcular el error cuadrático medio
    er = mean((y - y_pred).^2);
end

function [yestim,coef] = polyValFit(n, x, y, xi) 
%% x, y vectores columna
if nargin < 4
    xi=linspace(0,1,100);
end
%yestim = zeros(1,100);
    if (n == 1)
        M = [x ones(size(x))];
        coef = pinv(M)*y; 
        yestim  = xi   * coef(1) + coef(2);
    end
    if (n == 2)
        M = [x.^2 x ones(size(x))];
        coef = pinv(M)*y; 
        yestim  = xi.^2 * coef(1) + xi   * coef(2) + coef(3);
    end

    if (n == 3)
        M = [x.^3 x.^2 x ones(size(x))];
        coef = pinv(M)*y; 
        yestim  = xi.^3 * coef(1) +  xi.^2 * coef(2) + xi   * coef(3) + coef(4);
    end

    if (n == 4)
        M = [sin(x) x.^3 x.^2 x ones(size(x))];
        coef = pinv(M)*y; 
        yestim  = sin(xi) * coef(1) + xi.^3 * coef(2) +  xi.^2 * coef(3) + xi   * coef(4) + coef(5);
    end
    if (n == 5)
        M = [cos(x) sin(x) x.^3 x.^2 x ones(size(x))];
        coef = pinv(M)*y; 
        yestim  = cos(xi) * coef(1) + sin(xi) * coef(2) + xi.^3 * coef(3) +  xi.^2 * coef(4) + xi   * coef(5) + coef(6);
    end

end
