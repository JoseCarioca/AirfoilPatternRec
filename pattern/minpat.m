function y=minpat(x);
% MINPAT   - Minimo de un conjunto de patrones
  
%	Copyright (c) Pedro L. Galindo (1998)

if size(x,2)==1,
   y=x;
else,
   y = min(x')';
end;
