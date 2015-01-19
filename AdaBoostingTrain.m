function model = AdaBoostingTrain( X, y, T )
% use ada boosting for classification
% X: input data
% y: output data
% T: iteration count
N = size(X, 1);
d = size(X, 2);
% for every dimension for X order esc
Sort_X = sort(X);
model.T = T;
gt = zeros(T, 4);
U = zeros(T, N);
u = ones(N, 1);

for t = 1:T
    best_s = 0;
    best_d = 0;
    best_threshold = 0;
    best_error_rate = -1;
    best_predict_info = ones(N, 1);
    for i = 1:d
        for s = [1,-1]
            % select different threshold
            for j = 1:N-1
                threshold = (Sort_X(j, i) + Sort_X(j+1, i))/2;
                error_amount = 0;
                predict_info = ones(N, 1);
                for n = 1:N
                    if X(n, i) >= threshold && s*y(n) == -1
                        error_amount = error_amount + u(n);
                        predict_info(n) = -1;
                    elseif X(n, i) < threshold && s*y(n) == 1
                        error_amount = error_amount + u(n);
                        predict_info(n) = -1;
                    end
                end
                error_rate = error_amount/sum(u);

                if best_error_rate == -1 || error_rate < best_error_rate
                    best_s = s;
                    best_d = i;
                    best_threshold = threshold;
                    best_error_rate = error_rate;
                    best_predict_info = predict_info;
                end
                
            end
        end
    end
    gt(t, :) = [best_s, best_d, best_threshold, best_error_rate];
    U(t, :) = u';
    % caculate the new u
    delta_t = ((1-best_error_rate)/best_error_rate)^0.5;
    for i = 1:N
        if best_predict_info(i) == 1
            u(i) = u(i) / delta_t;
        else
            u(i) = u(i) * delta_t;
        end
    end
end

model.U = U;
model.gt = gt;

end

