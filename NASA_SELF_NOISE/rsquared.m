function sol = rsquared(ts_y,ypredict)
    media = mean(ts_y);
    sol = 1- sum((ts_y - ypredict).^2 ) / sum((ts_y-media).^2);
end