function [x_star, f_star] = KQP(Q, q, x_start,a, l, u, b, eps, eps_prime, stepsize, stepsize_args)
    % docstring

    stepsize_args = num2cell(stepsize_args);

    f = @(x) objective_function(Q,q,x);
    prj = @(y) projection(a, l, u, b, y, eps_prime);

    x_i = x_start;
    x_prev = Inf(size(x_start));

    [f_i, g_i] = f(x_i);

    x_s = x_start;
    f_s = f_i;
    g_s = g_i;

    while norm(x_i - x_prev) >= eps
        d = -g_i;

        if stepsize == "fixed"
            alpha = stepsize_args{:};
        elseif stepsize == "diminishing"
            alpha = diminishing_stepsize(stepsize_args{:});
        elseif stepsize == "polyak"
            alpha = polyak_stepsize(f_i, g_i, f_s, stepsize_args{:});
        elseif stepsize == "armijo"
            alpha = armijo_stepsize(f, prj, x_i, g_i, d, stepsize_args{:});
        end

        y = x_i + alpha*d;
        x_prev = x_i;
        x_i = prj(y);

        [f_i, g_i] = f(x_i);

        x_s = [x_s, x_i];
        f_s = [f_s, f_i];
        g_s = [g_s, g_i];
    end
    x_star = x_i;
    f_star = f_i;
end


% maybe can be replaced with anonimus function passed as args to KQP
function [stepsize] = diminishing_stepsize()
    % docstring
    stepsize = 0.1;
end


function [stepsize] = polyak_stepsize(f_i, g_i, f_s, delta)
    stepsize = (f_i - min(f_s) + delta)/(norm(g_i)^2);
end


function [stepsize] = armijo_stepsize(f, prj, x_i, g_i, d, beta, s, delta)
    % docstring

    m = 0;
    x_beta = prj(x_i + (beta^m)*s*d);

    while f(x_i) - f(x_beta) < delta*g_i'*(x_i - x_beta)
        m = m + 1;
        x_beta = prj(x_i + (beta^m)*s*d);
    end

    stepsize = (beta^m)*s;
end