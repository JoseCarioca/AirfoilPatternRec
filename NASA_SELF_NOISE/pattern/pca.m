function [W,eigval]=pca(x,n)
% PCA      - Principal Component Analysis
%         
%    [W,eigval]=pca(x,n)
%  
%       W      = matriz cuyas filas son los autovectores de la matriz de covarianza 
%       eigval = vector columna de autovalores ordenados descendentemente
%
%       x = matriz de patrones
%       n = numero de dimensiones proyectadas
%  
%       Para proyectar, hacer y=W*x

%	Copyright (c) Pedro L. Galindo (1998)

C=covpat(x);

[W,B]=eig(C);
B=diag(B);
[valores,orden]=sort(-B);  % Ordena B de forma descendente

eigval = B(orden);         % Ordena los eigenvalores descendentemente
W = W(:,orden)';           % Ordena la matriz de igual forma

if nargin==2,              % Si existe el parametro n, recortar W y eigenv
   n = min(n,size(C,1));
   W=W(1:n,:);
   eigval=eigval(1:n);
end;
