function [x_star_real, f_star_real] = minimize_matlab_kqp(x_start, Q, q, l, u, a, b, max_iters, compute_optimum)
    % solve the minimization problem f(x) = x'Qx + q'x given the
    % constraints a'x >= b and l <= x <= u using matlab built-in solver
    % fmincon
    % 
    % input arguments:
    % - x_start: the starting point of the algorithm
    % - Q (a n x n positive semi-definite matrix) and q (a n dimensional vector) to
    % represent the quadratic function to minimize
    % - l, u, a (n dimensional vectors) and b (scalar) to define the feasible region
    % - max_iters: the maximum number of iterations to compute the solution
    % - compute_optimum: flag to 
    %
    % outputs:
    % - x_star_real: the optimal point computed by fmincon
    % - f_star_real: the value of function in the optimium point
    % - copmpute_optimum: flag to set the global variables to 
    %
    % golabal variables (only if compute_optimum = false):
    % - x_s_fmincon: the sequence of points computed by fmincon at each
    % iteration
    % - f_s_fmincon: the value of the function in the points computed at each
    % iteration

    if compute_optimum
        options = optimoptions('quadprog', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point-convex', ... % algorithm used
            'OptimalityTolerance', 1e-15, ... % For some large-scale problems with only linear equalities, the first-order optimality measure is the infinity norm of the projected gradient. In other words, the first-order optimality measure is the size of the gradient projected onto the null space of Aeq.
            'ConstraintTolerance', 1e-15, ... % Tolerance on the constraint violation, a positive scalar
            'MaxIterations', 3000); % Maximum number of iterations allowed, a positive integer
    else
        options = optimoptions('quadprog', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point-convex', ... % algorithm used
            'OptimalityTolerance', 1e-6, ... % For some large-scale problems with only linear equalities, the first-order optimality measure is the infinity norm of the projected gradient. In other words, the first-order optimality measure is the size of the gradient projected onto the null space of Aeq.
            'ConstraintTolerance', 1e-15, ... % Tolerance on the constraint violation, a positive scalar
            'MaxIterations', max_iters); % Maximum number of iterations allowed, a positive integer
    end
    
    [x_star_real, f_star_real] = quadprog(2*Q, q, -a', -b, [], [], l, u, x_start, options);
end