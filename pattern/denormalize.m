function y=denormalize(x,maximo,minimo);
% NORMALIZE - Realiza la operacion inversa a Normalize
%
%     y=denormalize(x,maximo,minimo);
%

%	Copyright (c) Pedro L. Galindo (1998)

x = mulpat(x,maximo-minimo);
y = addpat(x,minimo);