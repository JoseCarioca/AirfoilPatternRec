function d=chernoff(m1,c1,m2,c2,s);
% CHERNOFF - Calcula la distancia de Chernoff entre dos normales multivariadas
%            en funcion del parametro s
%
%           d=chernoff(m1,c1,m2,c2,s)
%

%	Copyright (c) Pedro L. Galindo (1998)

K1 = s*(1-s)/2 * ( (m2-m1)' * inv(s*c1+(1-s)*c2) * (m2-m1));
K2 = 1/2 * log( det(s*c1+(1-s)*c2) / ((det(c1)^s)*(det(c2)^(1-s))));
d = K1 + K2;