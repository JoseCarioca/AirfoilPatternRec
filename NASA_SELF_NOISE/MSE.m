function [error] = MSE(y, yest)
    error = sqrt(sum(abs(y - yest)) / length(yest));
end

