function [x_star, f_star, x_s, f_s, g_s, y_s] = KQP(Q, q, l, u, a, b, x_start, eps, eps_prime, max_iterations, stepsize, stepsize_args, verbose)
    % minimize the quadratic function f(x) = x'Qx + q'x given the constraints
    % a'x >= b and l <= x <= u, by applying the gradient projection method
    % at each step the next point is computed as in the normal gradient
    % descent algorithm y_i = x_i + alpha_i*d_i (where d_i is the
    % anti-gradient of f(x_i)), and then y_i is projected into the feasible
    % region to obtain x_{i+1) = proj(y_i)
    %
    % input arguments:
    % - Q (a nxn positive semi-definite matrix) and q (a n dimensional vector) to
    % represent the quadratic function to minimize
    % - l, u, a (n dimensional vectors) and b (scalar) to define the feasible region
    % - x_start: the starting point of the algorithm
    % - eps: the precision required from the algorithm
    % - eps_prime: the precision required to projection sub-problem
    % - max_iterations: the maximum number of iterations to compute the
    % solution
    % - stepsize and stepsize_args: a string which allows to select how to
    % compute the step size, the possible values are:
    %   - "fixed": then stepsize_args has to be the value for the stepsize
    %   - "diminishing" and "polyak": then stepsize_args has to be a function which
    %   specifies how to compute the step size with respect to the current
    %   iteration (eg. @(i) 1/i)
    %   - "armijo" and "armijo_ii": then stepsize_args consists of the two
    %   parameters of the algorithm beta and sigma
    % - verbose: boolean variable to show or suppress the output of the
    % algorithm
    %
    % output arguments:
    % - x_star: the optimal point computed by the algorithm
    % - f_star: the value of function in the optimium point
    % - x_s: the sequence of points computed by the algorithm at each
    % iteration
    % - f_s: the value of the function in the points computed at each
    % iteration
    % - g_s: the value of the gradient of the function in the points
    % computed at each iteration
    % - y_s: the sequence of points computed by the algorithm at each
    % iteration before being projected

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
    % modified polyak stepsize
    % 
    % input arguments:
    % - f_i: the value of the function at the i-th iteration
    % - g_i: the value of the gradient at the i-th iteration
    % - f_best: the best value of f found up until the i-th iteration
    % - gamma: the value of gamma for the i-th iteration
    %
    % output arguments:
    % - stepsize: the value of the stepsize for the i-th stepsize computed
    % with the modified polyak stepsize

    stepsize = (f_i - f_best + gamma)/(norm(g_i)^2);
end


function [stepsize] = armijo_stepsize_i(f, prj, x_i, g_i, d, beta, sigma)
    % first version of armijo stepsize
    % 
    % input arguments:
    % - f: the objective function
    % - prj: the projection function
    % - x_i: the point at the i-th iteration
    % - g_i: the value of the gradient at the i-th iteration
    % - d: the value of the anti-gradient at the i-th iteration
    % - beta and sigma: the values of the two parameters of the algorithm
    %
    % outputs:
    % - stepsize: the value of the stepsize for the i-th stepsize computed
    % as with the amijo algorithm

    m = 0;
    x_beta = prj(x_i + (beta * 2^(-m))*d);

    while f(x_i) - f(x_beta) < sigma*g_i'*(x_i - x_beta) && m < 500
        m = m + 1;
        x_beta = prj(x_i + (beta * 2^(-m))*d);
    end

    stepsize = (beta * 2^(-m));
end

function [gamma] = armijo_stepsize_ii(f, x_i, z, g_i, sigma)
    % second version of armijo stepsize
    % 
    % input arguments:
    % - f: the objective function
    % - x_i: the point at the i-th iteration
    % - z: 
    % - g_i: the value of the gradient at the i-th iteration
    % - d: the value of the anti-gradient at the i-th iteration
    % - sigma: the values of the two parameters of the algorithm
    %
    % outputs:
    % - stepsize: the value of the stepsize for the i-th stepsize computed
    % as with the amijo algorithm

    m = 0;
    while f(x_i) - f(x_i + 2^(-m)*(z - x_i)) < sigma*2^(-m)*g_i'*(x_i - z) && m < 500
        m = m + 1;
    end

    gamma = 2^(-m);
end