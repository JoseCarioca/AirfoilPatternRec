function centroides=kmeans(x,n,c0);
% KMEANS   - Algoritmo de las K-medias
%
%    centroides = kmeans(x,n)
%    centroides = kmeans(x,n,c0)
%
%    x  : patrones 
%    n  : numero de clases a localizar
%    c0 : centroides iniciales

%	Copyright (c) Pedro L. Galindo (1998)

if nargin==2,
   % Inicializamos los centroides a n patrones de forma aleatoria
   % ============================================================
   [sizepat numpat]=size(x);
   [valores indices]=sort(rand(1,numpat));
   indices=indices(1:n);
   centroides=x(:,indices);
elseif nargin==3,
   centroides=c0;
else,
   help kmeans;
   return;
end;

% Algoritmo de las K-medias
% =========================
suma=1;
sumaant = 0;
contador=1;
while (sumaant ~= suma) & (contador<100),
   sumaant = suma;
   [ganadores,distancias]=mycenter(x,centroides);

   suma = sum(distancias);
   for i=1:n,
      centroides(:,i)=meanpat(myclass(x,ganadores,i));
   end;
   contador=contador+1;
   pause(0.1);
end;
