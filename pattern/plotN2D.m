function plotN2D(media,cova,valorpdf,cadena);
% PLOTN2D  - Dibuja la elipse de los puntos de una normal 2D de pdf constante (0.01 por defecto)
%
%            PLOTN2D(media,cova);
%            PLOTN2D(media,cova,0.01);
%            PLOTN2D(media,cova,'.g');
%            PLOTN2D(media,cova,0.01,'.r');
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==2,
   valorpdf=0.01;
   cadena='g';
elseif nargin==3,
   if isstr(valorpdf),
      cadena=valorpdf;
      valorpdf=0.01;
   else,
      cadena='g';
   end;  
elseif nargin~=4,
   help plotN2D
   return
end;

m=media(1);
n=media(2);
invcova=inv(cova);

a=invcova(1,1);
b=invcova(1,2);
c=invcova(2,1);
d=invcova(2,2);

distancia = 2*log(valorpdf*(2*3.14159)*sqrt(det(cova)));

    plotquad([a,                ... 
             d,                ...
             b+c,              ...
             -2*a*m - (b+c)*n, ...
             -2*d*n - (b+c)*m, ...
             a*m*m + (b+c)*m*n + d*n*n + distancia],cadena);

