function suma=sumpat(x);
% SUMPAT   - Suma de un conjunto de patrones
  
%	Copyright (c) Pedro L. Galindo (1998)

if size(x,2)==1,
   suma=x;
else,
   suma = sum(x')';
end;
