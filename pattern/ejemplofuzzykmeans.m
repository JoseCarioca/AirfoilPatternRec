clear all
close all
N = 1000;
K = 7;

% Generacion de x
r = 4+rand(1,N);
alpha = 2*pi*rand(1,N);
x=[r.*cos(alpha);r.*sin(alpha)];

% Determinacion inicial de prototipos
c=x(:,1:K)*0.999;

for rep=1:100,
   plot(x(1,:),x(2,:),'.'), axis equal, hold on;
   plot(c(1,:),c(2,:),'or','LineWidth',3);hold off;
   
   % Determinacion del prototipo mas cercano a cada dato
   for k=1:K,
       d(k,:)=d_euclid(x,c(:,k));
   end;       
   a=1./d;
   suma = sum(a);   
   u = a ./ (ones(K,1)*suma);
   
   % Actualizacion de los prototipos     
   for k=1:K,
       aux = ones(2,1) * u(k,:);
       c(:,k) = sumpat(x.*aux)./sumpat(aux);
   end;

   pause
end;
