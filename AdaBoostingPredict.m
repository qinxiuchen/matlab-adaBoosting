function [ target, error_rate ] = AdaBoostingPredict( model, X, y )
% predict the target value by adaboosting train model
gt = model.gt;
T = model.T;
N = length(X);
target = zeros(N, 1);
for j = 1:N
    for i = 1:T
        s = gt(i, 1);
        d = gt(i, 2);
        threshold = gt(i, 3);
        error_rate = gt(i, 4);
        delta_t = ((1-error_rate)/error_rate)^0.5;
        if X(j, d) >= threshold
            result = s*1;
        else
            result = s*(-1);
        end
        target(j) = target(j) + log(delta_t)*result;
    end
    if target(j) >= 0
        target(j) = 1;
    else
        target(j) = -1;
    end
end

error_count = 0;
for i = 1:N
    if target(i) ~= y(i)
        error_count = error_count + 1;
    end
end
error_rate = error_count / N;

end

