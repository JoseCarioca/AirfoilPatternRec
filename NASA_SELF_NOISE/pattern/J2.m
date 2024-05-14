function d = J2(x, y)
% J2       - Criterio de clustering (a minimizar) : (-1) * traza(inv(Sw)*Sb)
%            tambien denominado criterio de Hotelling
%	J = J2(x,y)

%	Copyright (c) Pedro L. Galindo (1998)

[Sb,Sw]=scatter(x,y);
if abs(det(Sw))<0.01,
   d = 1e10;
else,
   d=-trace(inv(Sw)*Sb);
end;