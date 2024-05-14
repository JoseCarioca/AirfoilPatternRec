function [y,indices] = chitest(x,media,covaria,t)
% CHITEST  - Test Chi-cuadrado
%            Determina las muestras de una poblacion cuya distancia de 
%            Mahalanobis es menor que el parametro de test
%
%       [y,indices] = chitest(x,media,covaria,test_parameter)
%
%        y       : muestras que cumplen el test
%        indices : indices de dichas muestras en x
%        
%        x : poblacion
%        t : parametro de test

%	Copyright (c) Pedro L. Galindo (1998)

distancias = d_mahal(x,media,covaria);
indices = find(distancias<t);
y = x(:,indices);