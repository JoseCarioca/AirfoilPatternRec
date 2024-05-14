function y=parzen(x)
% PARZEN   - Ventana de Parzen

%	Copyright (c) Pedro L. Galindo (1998)

x=abs(x)-0.5;
x=(x>0);
y=(sum(x)==0);

