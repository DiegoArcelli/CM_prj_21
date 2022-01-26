function [x_star_real, f_star_real] = minimize_matlab_kqp(x_start, Q, q, l, u, a, b, max_iters, compute_optimum)
    % docstring
    if compute_optimum
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'Interior-Point', ... % algorithm used
            'OptimalityTolerance', 1e-15, ... % For some large-scale problems with only linear equalities, the first-order optimality measure is the infinity norm of the projected gradient. In other words, the first-order optimality measure is the size of the gradient projected onto the null space of Aeq.
            'ConstraintTolerance', 1e-15, ... % Tolerance on the constraint violation, a positive scalar
            'MaxIterations', 3000, ... % Maximum number of iterations allowed, a positive integer
            'SpecifyObjectiveGradient', true); % specify that i'll pass the gradient to so not needed to compute by differentiation
    else
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'Interior-Point', ... % algorithm used
            'OptimalityTolerance', 1e-6, ... % For some large-scale problems with only linear equalities, the first-order optimality measure is the infinity norm of the projected gradient. In other words, the first-order optimality measure is the size of the gradient projected onto the null space of Aeq.
            'ConstraintTolerance', 1e-6, ... % Tolerance on the constraint violation, a positive scalar
            'MaxIterations', max_iters, ... % Maximum number of iterations allowed, a positive integer
            'OutputFcn',@outfun, ... 
            'SpecifyObjectiveGradient', true); % specify that i'll pass the gradient to so not needed to compute by differentiation
    end
    
    [x_star_real, f_star_real] =  fmincon(@(x) objective_function(Q, q, x), x_start, -a', -b, [], [], l, u, [], options);
end

function [stop] = outfun(x,optimValues, states)
    global x_s_fmincon;
    global f_s_fmincon;
        
    if states == "iter"
        x_s_fmincon = [x_s_fmincon, x];
        f_s_fmincon = [f_s_fmincon, optimValues.fval];
    end
    
    stop = false;
end