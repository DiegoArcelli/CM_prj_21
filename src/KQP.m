function [x_star, f_star, x_s, f_s, g_s, y_s] = KQP(Q, q, l, u, a, b, x_start, eps, eps_prime, max_iterations, stepsize, stepsize_args, verbose)
    % docstring

    f = @(x) objective_function(Q,q,x);
    prj = @(y) projection(l, u, a, b, y, eps_prime);

    x_i = x_start;

    [f_i, g_i] = f(x_i);

    x_s = x_start;
    f_s = f_i;
    g_s = g_i;
    y_s = x_start;
    f_best = f_i;

    iteration = 1;

    while iteration <= max_iterations
        
        if verbose
            fprintf("iterata %d \n", iteration);
        end
        
        d = -g_i;

        if stepsize == "fixed"
            alpha = stepsize_args;
        elseif stepsize == "diminishing"
            alpha = stepsize_args(iteration);
        elseif stepsize == "polyak"
            alpha = polyak_stepsize(f_i, g_i, f_best, stepsize_args(iteration));
        elseif stepsize == "armijo"
            alpha = armijo_stepsize(f, prj, x_i, g_i, d, stepsize_args{:});
        end

        y = x_i + alpha*d;
        x_prev = x_i;
        x_i = prj(y);

        if norm(x_i - x_prev) < eps
            break
        end

        [f_i, g_i] = f(x_i);

        x_s = [x_s, x_i];
        f_s = [f_s, f_i];
        g_s = [g_s, g_i];
        y_s = [y_s, y];

        if f_i < f_best
            f_best = f_i;
        end

        iteration = iteration + 1;
    end

    x_star = x_i;
    f_star = f_i;
end


function [stepsize] = polyak_stepsize(f_i, g_i, f_best, gamma)
    % docstring
    stepsize = (f_i - f_best + gamma)/(norm(g_i)^2);
end


% find best parameters of beta and delta !!
function [stepsize] = armijo_stepsize(f, prj, x_i, g_i, d, beta, s, delta)
    % docstring

    m = 0;
    x_beta = prj(x_i + (beta * 2^(-m))*s*d);

    while f(x_i) - f(x_beta) < delta*g_i'*(x_i - x_beta) && m < 500
        m = m + 1;
        x_beta = prj(x_i + (beta * 2^(-m))*d);
    end

    stepsize = (beta * 2^(-m));
end