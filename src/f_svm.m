function [f, g] = f_svm(alpha, X)
    % docstring
    f = -(sum(alpha) - (1/2)*alpha'*X*X'*alpha);
    g = -(ones(size(alpha)) - X*X'*alpha);
end