function x=randnorm(media,covaria,n)
% RANDNORM - Genera n puntos de una gaussiana de media y matriz de covarianza dadas
%         
%    x = randnorm(media,covaria,n)
%  

%	Copyright (c) Pedro L. Galindo (1998)

[eigvect,eigval] = eig(covaria);
W = eigvect * (eigval^0.5);
y = randn(length(media),n);
x = W * y + tocol(media) * ones(1,n);