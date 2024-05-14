function d = norma(x)
% NORMA    - Calcula la norma (o longitud) de un vector o conjunto de vectores.
%
%	d = norma(x) 

%	Copyright (c) Pedro L. Galindo (1998)

if size(x,1)==1,
   x=x';
end;

d=sqrt(sum(x.*x));