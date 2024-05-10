function c = cluster2(x,k,particion,criterio)
% CLUSTER2 - Algoritmo de clustering jerarquico   !! O(n^2) !!
%            "Stepwise-Optimal Hierarchical Clustering"

%
%       clases = cluster1(x,k)
%       clases = cluster1(x,k,particion) 
%       clases = cluster1(x,k,'J2');
%       clases = cluster1(x,k,particion,'J2')
%
%         x      = conjunto de patrones
%         particion     = particion inicial
%         k      = numero de clusters a localizar
%         clases = categoria de cada patron de x
%         criterio = 'J4'  (por defecto)
%
%       Como criterio es posible usar J1, J2, J3, ...

%	Copyright (c) Pedro L. Galindo (1998)


if nargin==2,
   criterio='J4';
   particion=1:length(x);   % Cada patron es una clase
elseif nargin==3,
   if isstr(particion),
      criterio=particion;
      particion=1:length(x);
   else,      
      criterio='J4';
   end;
elseif nargin~=4,
   help cluster1;
   error('Parametros incorrectos')
end;


[d,n] = size(x);
nc = k;
c = particion;
nk = n;

while nk>nc,
  %-----------------------------------------------------
  % Buscar los dos clusters más cercanos
  %-----------------------------------------------------
  dmin = 1e20;
  for i=1:nk-1,
    i
    indicesi = find(c==i);
    filadeunos = ones(1,length(indicesi));
    for j=i+1:nk,
    
     
      caux = c;
      caux(indicesi)= j * filadeunos;

      d = feval(criterio,x,caux);
      if d<dmin, 
         dmin=d; 
         imin=i; 
         jmin=j; 
      end
    end
  end
  %-----------------------------------------------------
  % Mezclar los clusters imin y jmin
  %-----------------------------------------------------
  for i=1:n,
    if c(i)==jmin, 
       c(i)=imin; 
    end
    if c(i)==nk, 
       c(i)=jmin; 
    end
  end
  %-----------------------------------------------------
  % Decrementar el numero de clases
  %-----------------------------------------------------
  nk = nk - 1;
  [nk feval(criterio,x,c)]
end

return


if nargin==2,
   criterio='J4';
   p0=1:length(x);   % Cada patron es una clase
elseif nargin==3,
   if isstr(p0),
      criterio=p0;
      p0=1:length(x);
   else,      
      criterio='J4';
   end;
elseif nargin~=4,
   help cluster1;
   error('Parametros incorrectos')
end;

N = length(allclass(p0));

for numclases=N:-1:k+1,
   numclases
   minimoglobal = 1e10;

   categorias = allclass(p0);
   M = length(categorias);

   % Para cada pareja de clases, buscar la fusion que reduce en mayor medida el criterio
   for i=1:M-1,
      indice1=find(p0==categorias(i));
      clase1=x(:,indice1);
      for j=i+1:M,
         indice2=find(p0==categorias(j));
         paux = p0;
         paux(indice2)=categorias(i)*ones(1,size(indice2,2));
         minimoparcial=feval(criterio,x,paux);
         if minimoparcial<minimoglobal,
            patron1=categorias(i);
            patron2=categorias(j);
            minimoglobal=minimoparcial;
         end;
      end;
   end

   % Fusionar las clases del patron1 y patron2
   indice2=find(p0==patron2);
   p0(indice2)=patron1*ones(1,length(indice2));   
end;

% Renumerar las clases de 1 hasta numclases
% =========================================
y=p0;
j=1;
for i=allclass(p0),
   indices=find(p0==i);
   y(indices)=j*ones(1,length(indices));
   j=j+1;
end;
