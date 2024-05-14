function [y,d]=mycenter(patrones,centroides);
% MYCENTER - Asigna cada patron a su centroide mas cercano. 
%            Devuelve asimismo el cuadrado de dicha distancia.
%
%    [y,d] = mycenter(patrones,centroides)
%
%    y = vector de centroides
%    d = vector de distancias
  
%	Copyright (c) Pedro L. Galindo (1998)

[basura,numpatrones]=size(patrones);
[basura,numcentros]=size(centroides);

filaaunos = ones(1,numpatrones);

diferencias(numcentros,numpatrones)=0; % Sera la distancia de cada patron al centroide i-esimo   
for i=1:numcentros,
   d=patrones-centroides(:,i)*filaaunos;
   if size(d,1)==1,
      diferencias(i,:)=d.*d;
   else,
      diferencias(i,:)=sum(d.*d);
   end;
end;

[d,y]=min(diferencias);
