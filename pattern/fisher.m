function [W,f,eigenv]=fisher(x,y,n);
% FISHER   - Discriminante Lineal Multiclase de Fisher
%
%   [W,f,eigenv]=fisher(x,y,n)
%
%   W      => discriminante lineal de Fisher (aplicar luego y = W * x)
%   f      => coef. de interes de cada caracteristica
%   eigenv => coef. de interes de cada caracteristica tras la proyecccion. En general
%             habra K coefs. distintos de 0, siendo K el numero de clases menos una.
%
%   x      => patrones de entrada
%   y      => categoria de cada patron de entrada
%   n      => numero de coefs. deseados tras la proyeccion

%	Copyright (c) Pedro L. Galindo (1998)


[Sb,Sw,Sm]=scatter(x,y);

NC=size(x,2);

if nargin==3,
   n = min(n,NC);
else,
   n = NC;
end;

[W eigenv]=pca(inv(Sw)*Sb,n)
f = diag(Sb) ./ diag(Sw);
