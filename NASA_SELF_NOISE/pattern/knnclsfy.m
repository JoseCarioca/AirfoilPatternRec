function outtst=knnclsfy(intrn,outtrn,intst,k);
% KNNCLSFY - Realiza una clasificacion con el metodo K-Nearest neighbour
%
%          outtst = knnclsfy(intrn,outtrn,intst,k)
%

%	Copyright (c) Pedro L. Galindo (1998)

if nargin~=4, 
   error('Numero de argumentos invalido');
end;

valorminimo = min(outtrn) - 1;

outtrn = outtrn - valorminimo;

filaunos = ones(1,size(intrn,2));
for i=1:size(intst,2), % para cada patron de test
   patron = intst(:,i) * filaunos;
   distancias = intrn - patron;
   distancias = sum(distancias .* distancias);
   [distancias,ganadores]=sort(distancias);

   votos = zeros(1,max(outtrn));

   for j=1:k,
      indice = outtrn(ganadores(j));
      votos(indice)=votos(indice) + 1;
   end;
   [valor, posicion] = max(votos);

   outtst(i)=posicion;
end;

outtst = outtst + valorminimo;