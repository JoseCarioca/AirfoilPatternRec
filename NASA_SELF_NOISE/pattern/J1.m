function d = J1(x,y)
% J1       - Criterio de clustering (a minimizar) : suma errores al cuadrado
%            Coincide con la traza de Sw
%
%	J = J1(x,y)

%	Copyright (c) Pedro L. Galindo (1998)

clases=allclass(y);
d = 0;
for i=clases,
   xaux = myclass(x,y,i);
   media = meanpat(xaux);
   d = d + sum(d_eucli2(xaux,media));
end;