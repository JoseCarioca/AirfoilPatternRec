function centroide = cluster3(x,k)
% CLUSTER3 - Algoritmo de clustering MAX-MIN
%
%       centroide = cluster1(x,k)
%
%         x         = conjunto de patrones
%         k         = numero de clusters a localizar
%         centroide = centros encontrados
%

%	Copyright (c) Pedro L. Galindo (1998)


[numdim,numpatrones] = size(x);

%---------------------------------------------------------------------------
% Buscar los dos puntos más lejanos, que serán los dos primeros centroides
%---------------------------------------------------------------------------
dmax = -1e20;
for i=1:numpatrones-1,
  d = d_euclid(x(:,i+1:numpatrones),x(:,i));
  [valor,indice]=max(d);
  if valor>dmax,
     dmax=valor;
     imax=i;
     jmax=i+indice;
  end
end
centroide=zeros(numdim,k);
centroide(:,1)=x(:,imax);
centroide(:,2)=x(:,jmax);
nk=2;

while nk<k,
  %-------------------------------------------------------------------------------------
  % Crear un centroide en aquel punto cuya distancia al centroide mas cercano sea maxima
  %-------------------------------------------------------------------------------------
  dmax = -1e20;  
  d=zeros(nk,numpatrones);
  for i=1:nk,
     d(i,:) = d_euclid(x,centroide(:,i));
  end;
  d=min(d);
  [valor imax]=max(d);
  nk = nk + 1;
  centroide(:,nk)=x(:,imax);
end
