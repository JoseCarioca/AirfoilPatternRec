function error=rmse(x,t),
%
% RMSE error function
%
%    error=rmse(x,t)
%
%        x = salida obtenida
%        t = salida deseada
%

error = sumsqr(t-x)/sumsqr(t-mean(t));
