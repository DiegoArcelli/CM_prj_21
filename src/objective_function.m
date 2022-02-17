function [f_x, g_x] = objective_function(Q, q, x)
    % evaluate the value and the gradient of the
    % function f(x) = x'Qx + q'x in a given point x
    %
    % input arguments:
    % - Q: a n x n positive semi-definite matrix
    % - q: a n dimensional vector
    % - x: a n dimensional vector in which evaluate the function 
    %
    % output arguments:
    % - f_x: the value of f in the point x
    % - g_x: the value of the gradient of x in the point x

    f_x = x'*Q*x+ + q'*x;
    g_x = 2*Q*x + q;
end