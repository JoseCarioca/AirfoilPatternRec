function y = autocpat(x)
% AUTOCPAT - Matriz de autocorrelacion de un conjunto de patrones
%
%       y = autocpat(x)
%

%	Copyright (c) Pedro L. Galindo (1998)

y=(1/size(x,2)*x*x');
