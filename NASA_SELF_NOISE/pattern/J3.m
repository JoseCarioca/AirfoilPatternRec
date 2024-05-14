function d = J3(x, y)
% J3       - Criterio de clustering (a minimizar) : log(|C1|^n1 * |C2|^n2 * ...)
%     
%	J = J3(x,y)

%	Copyright (c) Pedro L. Galindo (1998)

d = 0;
for i=allclass(y),
   xaux = myclass(x,y,i);
   d = d + size(xaux,2) * log(det(covpat(xaux)));
end;