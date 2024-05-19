function [error] = MSE(y, yest)
    %% Absolute minimum error
    error = sum(abs(y - yest)) / length(yest);
end

