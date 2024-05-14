function y=maxpat(x);
% MAXPAT   - Maximo de un conjunto de patrones
  
%	Copyright (c) Pedro L. Galindo (1998)

if size(x,2)==1,
   y=x;
else,
   y = max(x')';
end;
