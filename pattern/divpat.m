function y=divpat(x,patron);
% DIVPAT   - Divide a un conjunto de patrones un vector dado
%

%	Copyright (c) Pedro L. Galindo (1998)

y=x./(patron*ones(1,size(x,2)));
