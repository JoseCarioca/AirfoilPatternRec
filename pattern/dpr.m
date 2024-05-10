function d = dpr(x,hiperplano)
% DPR      - Distancia de un punto n-dimensional a un hiperplano de dimension n-1
%
%	d = dpr(x,hiperplano)
%
%     El hiperplano debe estar definido como 
%        [A B C]   ==> A * x + B * y + C = 0;
%        [A B C D] ==> A * x + B * y + C * z + D = 0
%        etc.
%
%     x puede ser un conjunto de vectores
%     tanto x como hiperplano deben estar en formato columna
%
%     El signo puede ser negativo o positivo, segun en que semiespacio se 
%     encuentre el punto.

%	Copyright (c) Pedro L. Galindo (1998)

if length(hiperplano)~=size(x,1)+1,
   error('Longitud de parametros erronea');
end;

x=[x;ones(1,size(x,2))];
d = hiperplano' * x / norma(hiperplano);
