function x=circulo(centro,radio,N);
% CIRCULO  - Devuelve N puntos del circulo de centro y radio dados
%
%           x = circulo(centro,radio)
%           x = circulo(centro,radio,N)
%
%               centro : vector columna de dos coordenadas
%               radio  : radio del circulo
%               N      : numero de puntos
%
%               x      : vector de 2 filas y N columnas (N=100, por defecto) 

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==2,
   N=100;
elseif nargin~=3,
   help circulo
   break;
end;

x = linspace(0,2*pi,N);
x = radio * [cos(x);sin(x)];
x = x + tocol(centro) * ones(1,N);
