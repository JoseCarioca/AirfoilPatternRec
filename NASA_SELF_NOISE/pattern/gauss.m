function y = gauss(media, covar, patrones)
% GAUSS    - Evalua una distribucion Gaussiana multivariada
%
%	y = gauss(media, covar, x) evalua la f. de densidad Gaussiana
%
%   media ha de ser de dimensiones (Dx1)
%   covar ha de ser una matriz     (DxD)
%   x es un conjunto de vectores   (DxN)
%
%     Recordar que los puntos se distribuyen en torno a la media según :
%
%     media +- 1 * stddev = 68 por ciento poblacion
%     media +- 2 * stddev = 95 por ciento     "
%     media +- 3 * stddev = 99.7 por ciento   "
%     media +- 4 * stddev = 99.99 por ciento  "

%	Copyright (c) Pedro L. Galindo (1998)

[nf, nc] = size(patrones);
x = addpat(patrones,-media);
aux = inv(covar)'*x;
if nf==1,
   y = exp(-0.5*aux.*x) / sqrt((2*pi)^nf*det(covar));
else,
   y = exp(-0.5*sum(aux.*x)) / sqrt((2*pi)^nf*det(covar));
end;