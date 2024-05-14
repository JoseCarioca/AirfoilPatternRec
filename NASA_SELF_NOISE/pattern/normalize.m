function [y,maximo,minimo]=normalize(x);
% NORMALIZE - Normaliza a valores 0..1 de forma independiente
%
%     [y,maximo,minimo]=normalize(x);
%
%     Para recomponer los datos originales, aplicar "denormalize"
%
%	Copyright (c) Pedro L. Galindo (1998)

maximo = maxpat(x);
minimo = minpat(x);
x=subpat(x,minpat(x));
resta = maximo-minimo;
y = divpat(x,resta);
