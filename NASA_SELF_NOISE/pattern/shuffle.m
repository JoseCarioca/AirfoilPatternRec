function [x,y] = shuffle(x,y)
% SHUFFLE  - Desordena un conjunto de patrones de entrenamiento
%
%    [x,y]=shuffle(x,y);
%  

%	Copyright (c) Pedro L. Galindo (1998)

[dummy,orden] = sort(rand(1,size(x,2)));
x = x(:,orden);

if nargin==2,
  y = y(:,orden);
end;