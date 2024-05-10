function y=zeromean(x);
% ZEROMEAN - Resta a un conjunto de patrones la media, haciendola cero
%

%	Copyright (c) Pedro L. Galindo (1998)

y=addpat(x,-meanpat(x));
