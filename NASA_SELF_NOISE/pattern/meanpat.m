function media=meanpat(x);
% MEANPAT  - Media de un conjunto de patrones
  
%	Copyright (c) Pedro L. Galindo (1998)

if size(x,2)==1,
   media=x;
else,
   media = mean(x')';
end;
