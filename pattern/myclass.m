function [xi,ni] = myclass(x,c,i);
% MYCENTER - Extrae de x los patrones de la clase i-esima, segun se indica en c
%
%    clases = myclass(x,c,indice)
%
%    xi = vector de patrones de la clase i-esima
%    ni = numero de elementos de dicha clase
%    x = vector de patrones
%    c = vector de clases
%    i = clase deseada
  
%	Copyright (c) Pedro L. Galindo (1998)

xi = x(:,find(c==i));
ni = size(xi,2);