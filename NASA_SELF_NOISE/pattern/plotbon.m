function plotbon(m1,C1,m2,C2,cadena);
% PLOTBON  - Dibuja la interseccion de dos normales 2D
%
%            PLOTBON(media1,cova1,media2,cova2);
%            PLOTBON(media1,cova1,media2,cova2,'.g');
%            PLOTBON(media1,cova1,media2,cova2,-5:0.1:5);
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==4,
   cadena='g';
elseif nargin~=5,
   help plotbon
   return
end;

W = bayesbon(m1,C1,m2,C2);

plotquad(W,cadena);
