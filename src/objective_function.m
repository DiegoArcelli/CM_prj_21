function [f_x, g_x] = objective_function(Q, q, x)
% docstring
f_x = x'*Q*x+ + q'*x;
g_x = 2*Q*x + q;
end