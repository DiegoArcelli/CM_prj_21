function [f, g] = f_svm(alpha, X)
    % objective function of the svm quadratic optimization problem
    % input arguments:
    % - X (matrix) the matrix of the training samples
    % - alpha (vector) the lagrangian multiplier, element that has to be
    % minimized
    % output arguments:
    % - f (scalar) the value of the function
    % - g (vector) the vector of the gradient of the function
    
    f = -(sum(alpha) - (1/2)*alpha'*X*X'*alpha);
    g = -(ones(size(alpha)) - X*X'*alpha);
end