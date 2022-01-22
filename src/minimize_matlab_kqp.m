function [x_star_real, f_star_real] = minimize_matlab_kqp(n, x_start, Q, q, l, u, a, b)
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'active-set', 'OptimalityTolerance', 10e-15, 'ConstraintTolerance', 10e-15, 'MaxIterations', 1000, 'FunctionTolerance', 10e-16);
    
    [x_star_real, f_star_real] =  fmincon(@(x) objective_function(Q, q, x), x_start, -a', -b, [], [], l, u, [], options);
end

