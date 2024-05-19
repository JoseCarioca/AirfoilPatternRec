function [error] = RMSE(y, yest)
    % Minimum square error
    error = sqrt(sum((y - yest).^2) / length(yest));
end

