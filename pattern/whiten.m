function [y,W]=whiten(x)
% WHITEN   - Whitening transformation
%         
%    [y,W]=whiten(x)
%  
%       Obtiene una transformacion, tal que y tiene media 0 y covaria
%       igual a la matriz unidad.
%
%       y = vector de patrones de salida
%       W = matriz de transformacion
%       x = vector de patrones de entrada
%  

%	Copyright (c) Pedro L. Galindo (1998)

[W,eigval]=pca(x);
eigval=diag(eigval);
eig2=eigval^(-0.5);
W=eig2 * W;
y=W*x;

