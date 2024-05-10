function d = d_mahal(x1, x2, C)
% D_MAHAL  - Evalua la distancia de Mahalanobis entre dos puntos
%
%	d = d_mahal(x1, x2, covar)
%
%         x1    = conjunto de patrones (por columnas)
%         x2    = vector de referencia
%         covar = matriz de covarianza
%         d     = vector fila con las distancias Mahalanobis de cada vector de x1 a x2

%	Copyright (c) Pedro L. Galindo (1998)

R=chol(inv(C));

X = R * (x1-x2*ones(1,size(x1,2)));

d = sum(X .* X);
