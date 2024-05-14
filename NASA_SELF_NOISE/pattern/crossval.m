function [datatrn,dataval,outtrn,outval]=crossval(data,out,N,i);
% CROSSVAL - Extrae el sunbconjunto de patrones i-esimo de los N posibles
%
%      [datatrn,dataval,outtrn,outval]=crossval(data,out,N,i);

%
%	Copyright (c) Pedro L. Galindo (1998)

umbrales = round(linspace(1,size(data,2)+1,N+1));
inicios = umbrales(1:N);
finales = umbrales(2:N+1)-1;

datatrn = [data(:,1:inicios(i)-1) data(:,finales(i)+1:size(data,2))];
outtrn  = [out(:,1:inicios(i)-1)  out(:,finales(i)+1:size(data,2))];

dataval = data(:,inicios(i):finales(i));
outval  = out(:,inicios(i):finales(i));

