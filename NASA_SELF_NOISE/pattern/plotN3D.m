function plotN3D(media,cova);
% PLOTN3D  - Dibuja una gaussiana 3D 
%
%            PLOTN3D(media,cova);
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin~=2,
   help plotN3D
   return
end;

N=5;
np=5*sqrt(N)+15;
ancho = sqrt(N) + 1;

x=linspace(media(1)-ancho*sqrt(cova(1,1)),media(1)+ancho*sqrt(cova(1,1)),np);
y=linspace(media(2)-ancho*sqrt(cova(2,2)),media(2)+ancho*sqrt(cova(2,2)),np);

invcova=inv(cova);

x1=x-media(1);
y1=y-media(2);

for i=1:np,
   for j=1:np,
      X=[x1(i);y1(j)];
      Z(j,i)= X'*invcova*X;
   end;
end;
Z=1/(2*pi*sqrt(det(cova)))*exp(-0.5*Z);

contour(x,y,Z,0.01:0.02:0.2);