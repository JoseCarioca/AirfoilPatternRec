function y = torow(x)
% TOROW    - Convierte el vector x en un vector fila
%
%	y = torow(x) convierte un vector en una fila

%	Copyright (c) Pedro L. Galindo (1998)

[a,b]=size(x);
y=reshape(x,1,a*b);
