function [y,i1,i2] = intersec(x1,x2,t)
% INTERSEC - Conjunto interseccion entre dos poblaciones
%            Calcula la media y covarianza de cada poblacion, y aplica test chi2
%
%        [y,indices1,indices2] = intersec(x1,x2,t)
%
%        x1       : poblacion 1
%        x2       : poblacion 2
%        t        : parametro a utilizar en el test chi2
%
%        y        : poblacion interseccion
%        indices1 : indices de x1 pertenecientes a la interseccion
%        indices2 : indices de x2 pertenecientes a la interseccion

%	Copyright (c) Pedro L. Galindo (1998)

m1 = meanpat(x1);
c1 = covpat(x1);
m2 = meanpat(x2);
c2 = covpat(x2);

[aux,indicesaux] = chitest(x1,m1,c1,t);
[y1,indicesaux2]  = chitest(aux,m2,c2,t);
i1 = indicesaux(indicesaux2);

[aux,indicesaux] = chitest(x2,m1,c1,t);
[y2,indicesaux2]  = chitest(aux,m2,c2,t);
i2 = indicesaux(indicesaux2);

y=[y1 y2];