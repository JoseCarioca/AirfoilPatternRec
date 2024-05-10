function W = simudiag(W1,W2)
% SIMUDIAG - Diagonalizacion simultanea de dos matrices simetricas
%         
%    W = simudiag(W1,W2)
%  
%       Obtiene una transformacion que hace diagonales a ambas matrices.
% 
%       W * W1 * W' = matriz unidad
%       W * W2 * W' = matriz diagonal
%

%	Copyright (c) Pedro L. Galindo (1998)


[eigvec,eigval]=eig(W1);
eigaux = eigval^(-0.5);
W = eigaux * eigvec';

K=eigaux * eigvec' * W2 * eigvec * eigaux;

[eigvec2,eigval2] = eig(K);

W = eigvec2' * eigaux * eigvec';