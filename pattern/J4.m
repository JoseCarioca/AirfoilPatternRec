function d = J4(x, y)
% J4       - Criterio de clustering (a minimizar) : det(Sw)
%            tambien denominado criterio de Wilks
%	J = J4(x,y)

%	Copyright (c) Pedro L. Galindo (1998)

[Sb,Sw]=scatter(x,y);
d=det(Sw);