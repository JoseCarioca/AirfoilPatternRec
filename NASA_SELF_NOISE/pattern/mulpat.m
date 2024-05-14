function y=mulpat(x,patron);
% MULPAT   - Multiplica a un conjunto de patrones por un vector dado
%

%	Copyright (c) Pedro L. Galindo (1998)

y=x.*(patron*ones(1,size(x,2)));
