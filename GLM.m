function [porcentaje,p] = GLM(training_x, training_y, test_x, test_y)
    current_labels = unique(training_y);
    for i = 1:size(training_y, 2)
        maped_label = find(current_labels == training_y(1, i));
        training_y(1, i) = maped_label;
    end
    A = [training_x(3, :)' training_x(2, :)' training_x(1, :)' ones(size(training_x, 2), 1)];
    p = pinv(A) * training_y';

    y_pred = test_x(3, :) .* p(1) + test_x(2, :) .* p(2) + test_x(1, :) .* p(3) + p(4);

    y_pred = round(y_pred);

    for i = 1:size(y_pred, 2)
        if( y_pred(1, i) > length(current_labels))
            y_pred(1, i) = current_labels(end);
        else
            if( y_pred(1, i) <= 0)
                y_pred(1, i) = current_labels(1);
            else
                y_pred(1, i) = current_labels(y_pred(1, i));
            end
        end
    end

    aciertos   = find(y_pred == test_y);
    porcentaje = (size(aciertos, 2) / size(test_y, 2)) * 100;
end

