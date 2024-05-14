function [N,m,v] = clstats(x,c);
% CLSTATS  - Calcula el numero de puntos, la media y la matriz de covarianza de cada cluster
%
%     [n,m,v] = clstats(x,y)
%	x  - vectores de entrada
%	y  - cluster de cada vector de entrada
%
%	N  - Numero de muestras de cada cluster
%	m  - Media de cada cluster
%	v  - Matriz de covarianza de cada cluster

clases=allclass(c);
nc = length(clases);
v=[];
N(nc)=0;
m(size(x,1),nc)=0;

for i=1:nc,
   clase=clases(i);
   aux=x(:,find(c==clase));
   m(:,i)=meanpat(aux);
   v=[v covpat(aux)];  
   N(i) = size(aux,2);
end

