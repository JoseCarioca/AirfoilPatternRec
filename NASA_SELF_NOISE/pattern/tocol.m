function y = tocol(x)
% TOCOL    - Convierte el vector x en un vector columna
%
%	y = tocol(x)

%	Copyright (c) Pedro L. Galindo (1998)

[a,b]=size(x);
y=reshape(x,a*b,1);
