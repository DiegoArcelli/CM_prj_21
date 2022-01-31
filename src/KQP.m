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
        x_prev = x_i;

        if stepsize == "fixed"
            alpha = stepsize_args;
            y = x_i + alpha*d;
            x_i = prj(y);
        elseif stepsize == "diminishing"
            alpha = stepsize_args(iteration);
            y = x_i + alpha*d;
            x_i = prj(y);
        elseif stepsize == "polyak"
            alpha = polyak_stepsize(f_i, g_i, f_best, stepsize_args(iteration));
            y = x_i + alpha*d;
            x_i = prj(y);
        elseif stepsize == "armijo"
            alpha = armijo_stepsize_i(f, prj, x_i, g_i, d, stepsize_args{:});
            y = x_i + alpha*d;
            x_i = prj(y);
        elseif stepsize == "armijo_ii"
            [beta, sigma] = stepsize_args{:};
            y = prj(x_i + beta*d);
            gamma = armijo_stepsize_ii(f, x_i, y, g_i, sigma);
            x_i = x_i + gamma*(y - x_i);
        end

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


function [stepsize] = armijo_stepsize_i(f, prj, x_i, g_i, d, beta, sigma)
    % docstring

    m = 0;
    x_beta = prj(x_i + (beta * 2^(-m))*d);

    while f(x_i) - f(x_beta) < sigma*g_i'*(x_i - x_beta) && m < 500
        m = m + 1;
        x_beta = prj(x_i + (beta * 2^(-m))*d);
    end

    stepsize = (beta * 2^(-m));
end

function [gamma] = armijo_stepsize_ii(f, x_i, z, g_i, sigma)
    % docstring

    m = 0;
    while f(x_i) - f(x_i + 2^(-m)*(z - x_i)) < sigma*2^(-m)*g_i'*(x_i - z) && m < 500
        m = m + 1;
    end

    gamma = 2^(-m);
end