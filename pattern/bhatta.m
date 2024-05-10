function d=bhatta(m1,c1,m2,c2);
% BHATTA   - Calcula la distancia de Bhattacharayya entre dos normales multivariadas
%
%           d=bhatta(m1,c1,m2,c2)
%

%	Copyright (c) Pedro L. Galindo (1998)

d = chernoff(m1,c1,m2,c2,0.5);